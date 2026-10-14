"""B04A bounded runner; measurements and resumable evidence are stage local."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import chess
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from nemesis.data.common import digest,write_json
from .assets import PLAN,verify_plan_config
from .guards import require_new_output
from .engine import EngineSession,build_candidate_bundle,audit_bundle,reply_leaves,hash_object,rebuild_root
from .panels import DEV,sample_panel,read_development_states,root_context,write_roots,forecast_nodes
from .select import select_move,branch_context,mix_probabilities,matched_selection
from .evaluate import seal_selections,false_weakness_controls
from .types import StageResult,Selection

class BudgetStop(RuntimeError): pass


def write_selection_output(selections,traces,output,seal):
    """The production writer seals the complete trace before any scoring access."""
    from .evaluate import validate_traces
    if Path(output).exists(): raise FileExistsError('Selection output exists')
    validate_traces(selections,traces)
    seal_selections(selections,seal,traces=traces)
    pq.write_table(pa.Table.from_pylist([dict(asdict(s),**t) for s,t in zip(selections,traces)]),output)

def canonical_input(row):
    from nemesis.data.prepare import input_fingerprint
    b=rebuild_root(root_context(row)); stack=[]; prior=b.copy()
    for _ in range(8):
        stack.append(prior.copy())
        if not prior.move_stack: break
        prior.pop()
    row['input_hash']=input_fingerprint(b,list(reversed(stack)),1900,1900,0)

class TablePolicy:
    access_role='selector'
    def __init__(self,identity,tables,hashes):
        self.identity=identity; self.tables=tables; self.model_hash=hash_object(hashes)
    def probabilities(self,root,move): return [t[(root.root_id,move)] for t in self.tables]

def run_selector(roots,bundles,registry,local,check,*,capture_tables=False,traces=None):
    # Only T/G/G0 weights in this process. Compute full distributions once per model.
    import torch
    from nemesis.models.personal_predictor import TwinPredictor,registry_record
    records=json.loads(Path(registry).read_text())
    mids=['G0','G']+[f'T_{p}_{s}' for p in DEV for s in (1729,2718,3141)]
    for mid in mids: registry_record(records,mid,'selector')
    tables={}; hashes={}; calls=[]
    keys=[]; contexts=[]
    for root in roots:
        for m in bundles[root.root_id].reply_values:
            if '__terminal__' not in bundles[root.root_id].reply_values[m]:
                keys.append((root.root_id,m)); contexts.append(branch_context(root,m))
    gpu_start=time.perf_counter()
    for n,mid in enumerate(mids):
        check(estimated_cpu=10,estimated_gpu=15)
        tick=time.perf_counter(); cpu=time.process_time(); model=TwinPredictor(registry,mid,'selector')
        table={}
        for start in range(0,len(contexts),32):
            for key,d in zip(keys[start:start+32],model.predict_many(contexts[start:start+32])):
                table[key]=dict(zip(d.moves_uci,d.probabilities))
        tables[mid]=table; hashes[mid]=model.model_hash
        del model; torch.cuda.empty_cache()
        calls.append(dict(model_id=mid,cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-tick))
        print('selector',mid,'seconds',round(calls[-1]['wall_seconds'],2),flush=True)
    gpu_seconds=time.perf_counter()-gpu_start
    write_json(local/'selector_role_costs.json',dict(calls=calls,gpu_seconds=gpu_seconds))
    if capture_tables:
        write_json(local/'probability_tables.json',dict(model_hashes=hashes,
          tables={mid:{'|'.join(key):value for key,value in table.items()} for mid,table in tables.items()}))
    return select_from_tables(roots,bundles,tables,hashes,traces=traces),gpu_seconds


def select_from_tables(roots,bundles,tables,hashes,*,traces=None):
    """Same selection path for fresh model inference and bounded frozen-table replay."""
    keys=set(tables['G'])
    policies={'GENERIC':TablePolicy('GENERIC',[tables['G']],[hashes['G']]),
      'GENERIC_G0':TablePolicy('GENERIC_G0',[tables['G0']],[hashes['G0']])}
    personal=[]
    for p in DEV:
        ids=[f'T_{p}_{s}' for s in (1729,2718,3141)]
        policies[p]=TablePolicy(p,[tables[i] for i in ids],[hashes[i] for i in ids]); personal.extend(ids)
    # MIX8 averages each person's three final probability distributions, uniformly across identities.
    mixed={key:mix_probabilities([tables[i][key] for i in personal]) for key in keys}
    policies['MIX8']=TablePolicy('MIX8',[mixed],[hashes[i] for i in personal])
    selections=[]
    for root in roots:
        b=bundles[root.root_id]
        for method in ('GENERIC','GENERIC_G0','MIX8','SF_BEST','SAFE_RANDOM'):
            policy=policies[method] if method in policies else policies['GENERIC']
            selections.append(select_move(root,b,policy,method,0,traces=traces))
        for p in DEV:
            for method,lam in [('PERSONAL_PLAIN',0),('PERSONAL_ROBUST',1)]:
                selections.append(select_move(root,b,policies[p],method,lam,traces=traces))
    return selections

def run_screen(config_path,assets_path,out):
    require_new_output(out); verify_plan_config(config_path)
    # External approval is intentionally separate from the pre-review false flag.
    approval=Path('workflow/reviews/lw-nemesis-stage1-B03R-R01/response.md')
    text=approval.read_text(encoding='utf8')
    base='4e2f631f46f57dd9918b1ed119912dfe993994fa'
    if base not in text or 'APPROVED' not in text or 'SUFFICIENT' not in text: raise PermissionError('Exact B03R approval missing')
    assets=json.loads(assets_path.read_text()); registry=Path('artifacts/B03/model_registry.json')
    if digest(registry)!=assets['registry_hash'] or assets['plan_version']!=PLAN: raise ValueError('Assets binding mismatch')
    if out.exists() and any(p.name not in ('validation','minimal_unit.json','minimal_unit.raw.log','COST.preflight.parquet','screen.raw.log','failed-attempt01') for p in out.iterdir()):
        raise FileExistsError('B04A output already contains a run; use explicit new attempt')
    out.mkdir(parents=True,exist_ok=True); local=out/'local'; local.mkdir(); (out/'roots').mkdir()
    cfg=yaml.safe_load(config_path.read_text(encoding='utf8'))
    start=time.perf_counter(); cpu_start=time.process_time(); engine_cpu=0.; child_cpu=0.; gpu=0.
    work=[]; allowances=300.; inherited=json.loads(Path('artifacts/V2/B03R/resource_ledger.json').read_text())['project_actual']
    inherited_disk=44.47238283511251
    # Charge the prior minimum unit exactly, plus setup/test/CLI/finalization allowance.
    minimal=json.loads((out/'minimal_unit.json').read_text()) if (out/'minimal_unit.json').exists() else None
    if minimal:
        engine_cpu+=minimal['engine_cpu_seconds']; child_cpu+=minimal['python_cpu_seconds']
        work.append(dict(work_unit_id='minimal-engine-unit',cpu_seconds=engine_cpu+child_cpu,gpu_seconds=0,
          evidence='minimal_unit.json',status='COMPLETE_ENGINE_UNIT'))
    failed=out/'failed-attempt01/interrupted_attempt_costs.json'
    if failed.exists():
        failed_cost=json.loads(failed.read_text()); child_cpu+=failed_cost['cpu_seconds']
        work.append(dict(work_unit_id='failed-attempt01',cpu_seconds=failed_cost['cpu_seconds'],gpu_seconds=0,
          reason=failed_cost['reason'],evidence=str(failed.relative_to(out))))
    session=None; panel_rows={}; logs=[]; cost_measurements=[]; bundles={}; audit_cp={}; audit_values={}; selections=[]; gpu_active_start=None
    status='DONE'; research='ENGINE_AND_PANELS_VALIDATED'; reason='B04A completed; no utility conclusion'
    def consumed():
        active=session.cpu_seconds()-session.cpu_start if session else 0
        return (time.process_time()-cpu_start+engine_cpu+child_cpu+active+allowances)/3600
    def check(estimated_cpu=0,estimated_gpu=0):
        total=consumed()
        if total+1.25*estimated_cpu/3600>1.9 or inherited['cpu_core_hours']+total+1.25*estimated_cpu/3600>48:
            raise BudgetStop('CPU forecast exceeds stage/cumulative cap with 5% closure reserve')
        active_gpu=(time.perf_counter()-gpu_active_start)/3600 if gpu_active_start else 0
        if gpu+active_gpu+1.25*estimated_gpu/3600>.2375 or inherited['gpu_hours']+gpu+active_gpu+1.25*estimated_gpu/3600>20:
            raise BudgetStop('GPU forecast exceeds stage/cumulative cap with 5% closure reserve')
    def screen(row):
        check(estimated_cpu=5)
        try:
            b=build_candidate_bundle(root_context(row),dict(cfg,_engine_session=session,_screen_only=True))
            row['engine_bundle_hash']=hash_object(asdict(b)); return b.status
        except (chess.engine.EngineError,chess.engine.EngineTerminatedError,TimeoutError) as exc:
            row['eligibility_reason']=type(exc).__name__; return 'MISSING'
    try:
        rows=read_development_states('artifacts/B01/samples.parquet')
        write_json(out/'access_log.json',dict(actor='B04A_SELECTOR_SCREEN',path='artifacts/B01/samples.parquet',
          player_filter=list(DEV),split_filter=['calibration','H1','H2'],columns=__import__('nemesis.v2.panels',fromlist=['STATE_COLUMNS']).STATE_COLUMNS,
          input_sha256=digest('artifacts/B01/samples.parquet'),row_count=len(rows),confirmation_H_reads=0,S_compare_outcome_reads=0))
        used=set(); seen=set()
        panel_rows['COST'],l=sample_panel(rows,'COST',2,used,None,transform=canonical_input); logs.extend(l)
        panel_rows['N_dev'],l=sample_panel(rows,'N_dev',16,used,None,transform=canonical_input); logs.extend(l)
        for panel in ('COST','N_dev'): write_roots(out/'roots'/f'{panel}.parquet',panel_rows[panel])
        session=EngineSession(cfg)
        # First full root then bounded full COST32: no root skipped on effect or eligibility.
        for index,row in enumerate(panel_rows['COST']):
            estimate=max([m['cpu_seconds'] for m in cost_measurements],default=60)*2
            check(estimated_cpu=estimate)
            root=root_context(row); before=session.cpu_seconds(); tick=time.perf_counter(); calls_before=len(session.calls)
            b=build_candidate_bundle(root,dict(cfg,_engine_session=session)); bundles[root.root_id]=b
            online_nodes=sum(c['actual_nodes'] for c in session.calls[calls_before:])
            # Actual total online engine nodes, not depth or the 20k reply budget alone.
            matched=session.analyse(rebuild_root(root),root.root_color,online_nodes,namespace='sf_compute_matched')[0]
            extra=matched['move'] if matched['move'] not in b.safe_moves else None
            measure=dict(root_id=root.root_id,status=b.status,candidates=len(b.moves),safe=len(b.safe_moves),
              legal_root=rebuild_root(root).legal_moves.count(),replies=sum(len(v) for v in b.reply_values.values()),
              extra_reply_count=0,online_nodes=online_nodes,matched_call=dict(session.calls[-1]),
              sf_compute_matched_move=matched['move'],sf_compute_matched_cp=matched['cp'],
              cpu_seconds=session.cpu_seconds()-before,wall_seconds=time.perf_counter()-tick)
            cost_measurements.append(measure)
            write_json(local/'cost_checkpoint.json',dict(measurements=cost_measurements,bundles={k:asdict(v) for k,v in bundles.items()},
              audit_cp=audit_cp,audit_values=audit_values,calls=session.calls))
            print('COST',index+1,'/32',b.status,'safe',len(b.safe_moves),'CPU',round(measure['cpu_seconds'],2),flush=True)
            if index==0:
                # Remaining COST32 conservative 5-safe projection from real complete unit branch cost.
                ratio=5/max(1,len(b.safe_moves)); check(estimated_cpu=31*measure['cpu_seconds']*ratio+31*10)
        # Required deterministic S streams; model/effect outcomes have not been loaded.
        for panel,count in [('S_dev',8),('S_compare',4),('TUNE',4)]:
            panel_rows[panel],l=sample_panel(rows,panel,count,used,screen,seen_inputs=seen,transform=canonical_input)
            logs.extend(l); write_roots(out/'roots'/f'{panel}.parquet',panel_rows[panel])
            print(panel,len(panel_rows[panel]),'scanned',len(l),flush=True)
        # Retain all N roots, including no branches/out of range and missing states.
        for row in panel_rows['N_dev']:
            row['status']=screen(row); row['eligibility_reason']=row['status']; logs.append(dict(row))
        write_roots(out/'roots/N_dev.parquet',panel_rows['N_dev'])
        write_json(local/'engine_calls.json',session.calls)
        engine_cpu+=session.cpu_seconds()-session.cpu_start; session.close(); session=None
        roots=[root_context(r) for r in panel_rows['COST']]
        # Preflight GPU based on approved asset replay cost, then refine per model.
        check(estimated_cpu=150,estimated_gpu=180)
        gpu_active_start=time.perf_counter()
        traces=[]
        selections,selector_gpu=run_selector(roots,bundles,registry,local,check,traces=traces)
        # CUDA remains resident during the CPU audit and child E phase. Charge the
        # complete one-device residency window, including idle time.
        for measure in cost_measurements:
            selections.append(matched_selection(bundles[measure['root_id']],measure,traces))
        write_selection_output(selections,traces,out/'smoke_selections.parquet',local/'selection_seal.json')
        # New process/session and namespace, after the selection seal; benchmark audits are never scoring inputs.
        session=EngineSession(cfg)
        for root,measure in zip(roots,cost_measurements):
            check(estimated_cpu=60,estimated_gpu=60)
            before=session.cpu_seconds(); tick=time.perf_counter()
            cp,av=audit_bundle(root,bundles[root.root_id],session,selection_seal=local/'selection_seal.json',selections=selections,traces=traces)
            audit_cp[root.root_id]=cp; audit_values[root.root_id]=av
            extra=measure['sf_compute_matched_move']
            measure['extra_reply_count']=len(av.get(extra,{})) if extra not in bundles[root.root_id].safe_moves else 0
            measure['audit_cpu_seconds']=session.cpu_seconds()-before
            measure['cpu_seconds']+=measure['audit_cpu_seconds']; measure['wall_seconds']+=time.perf_counter()-tick
            if int(hash_object(['sensitivity',root.root_id])[:16],16)%10==0:
                before=session.cpu_seconds()
                scp,sv=audit_bundle(root,bundles[root.root_id],session,nodes=400000,
                  selection_seal=local/'selection_seal.json',selections=selections,traces=traces)
                write_json(local/f'sensitivity_{root.root_id}.json',dict(cp=scp,values=sv))
                measure['sensitivity_cpu_seconds']=session.cpu_seconds()-before
                measure['cpu_seconds']+=measure['sensitivity_cpu_seconds']
            print('audit',root.root_id[:8],'seconds',round(measure['audit_cpu_seconds'],2),flush=True)
        write_json(local/'audit_engine_calls.json',session.calls)
        engine_cpu+=session.cpu_seconds()-session.cpu_start; session.close();session=None
        request=dict(roots=panel_rows['COST'],selections=[asdict(s) for s in selections],registry=str(registry),
          config={'_selection_seal':str(local/'selection_seal.json'),'_selection_traces':traces,'_selector_pid':os.getpid(),'_audit_values':audit_values,'_audit_cp':audit_cp,
            '_original_candidates':{rid:list(b.moves) for rid,b in bundles.items()},
            '_evaluator_ids':['E3_group','E2_base','E2_group']+[f'E3_{p}_4242' for p in DEV]+[f'E2_{p}_6161' for p in DEV]})
        write_json(local/'evaluator_request.json',request)
        check(estimated_cpu=150,estimated_gpu=180)
        with (out/'evaluator.raw.log').open('w',encoding='utf8') as log:
            completed=subprocess.run([sys.executable,'-m','nemesis.v2.evaluate',str(local/'evaluator_request.json'),str(local/'evaluator_output.json')],stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode: raise RuntimeError('Independent evaluator failed; see evaluator.raw.log')
        evaluated=json.loads((local/'evaluator_output.json').read_text()); child_cpu+=evaluated['cpu_seconds']
        work.append(dict(work_unit_id='independent-COST-evaluator',cpu_seconds=evaluated['cpu_seconds'],gpu_seconds=evaluated['gpu_seconds']))
        write_json(local/'cost_checkpoint.json',dict(measurements=cost_measurements,bundles={k:asdict(v) for k,v in bundles.items()},
          audit_cp=audit_cp,audit_values=audit_values,calls=json.loads((local/'engine_calls.json').read_text())+json.loads((local/'audit_engine_calls.json').read_text())))
        write_json(out/'smoke_audit_summary.json',dict(records=len(evaluated['records']),roots=len(roots),identities=list(DEV),
          E3_models=9,E2_models=10,selection_seal_sha256=digest(local/'selection_seal.json'),utility_claim=False))
    except BudgetStop as exc:
        status='RESOURCE_LIMIT'; research='BUDGET_BLOCKED'; reason=str(exc)
    except Exception as exc:
        status='FAILED'; research='INSUFFICIENT_EVIDENCE'; reason=f'{type(exc).__name__}: {exc}'
        import traceback
        (out/'runtime_error.raw.log').write_text(traceback.format_exc(),encoding='utf8')
    finally:
        if gpu_active_start: gpu+=(time.perf_counter()-gpu_active_start)/3600
        if session:
            engine_cpu+=session.cpu_seconds()-session.cpu_start
            write_json(local/'engine_calls.json',session.calls); session.close(); session=None
    # Never fabricate unfinished panels, outcomes or zero research effects.
    for panel in ('COST','N_dev','S_dev','S_compare','TUNE'):
        if not (out/'roots'/f'{panel}.parquet').exists(): write_roots(out/'roots'/f'{panel}.parquet',panel_rows.get(panel,[]))
    write_roots(out/'screening_log.parquet',logs)
    if not (out/'smoke_selections.parquet').exists():
        pq.write_table(pa.Table.from_pylist([],schema=pa.schema([(k,pa.string()) for k in ('root_id','method_id','identity','move_uci','selection_hash','source_policy_hash')])),out/'smoke_selections.parquet')
    expected={'COST':32,'N_dev':512,'S_dev':256,'S_compare':128,'TUNE':64}
    counts={k:len(panel_rows.get(k,[])) for k in expected}
    if status=='DONE' and counts!=expected: research='INSUFFICIENT_COVERAGE'; reason='Fixed sampling rules did not fill all strata'
    write_json(out/'panels_manifest.json',dict(stage='B04A',run_id='B04A-N01',plan_version=PLAN,base_commit=base,
      assets_sha256=digest(assets_path),registry_sha256=digest(registry),approval_response_sha256=digest(approval),
      expected_counts=expected,actual_counts=counts,execution_status=status,research_status=research,
      S_compare_outcome_access_count=0,confirmation_H_reads=0,sampling='sha256(seed|panel|game_id), then sha256(seed|panel|game_id|ply|input_hash); game first; no resample in rejected game',
      game_overlap_count=sum(len(rs) for rs in panel_rows.values())-len({r['source_game_id'] for rs in panel_rows.values() for r in rs}),
      panels={k:dict(path=f'roots/{k}.parquet',sha256=digest(out/'roots'/f'{k}.parquet'),local_only=True) for k in expected}))
    write_json(out/'controls.json',false_weakness_controls())
    node_forecast=forecast_nodes(5,max([m['replies']+m['extra_reply_count'] for m in cost_measurements],default=0),
      max([m['legal_root'] for m in cost_measurements],default=0))
    # Sampling screens are not part of COST32's CPU numerator. Exclude their
    # nodes from the throughput denominator, or downstream cost is understated.
    cost_calls=[]; matched_count=0
    if cost_measurements and (local/'engine_calls.json').exists():
        for call in json.loads((local/'engine_calls.json').read_text()):
            cost_calls.append(call)
            if call['namespace']=='sf_compute_matched':matched_count+=1
            if matched_count==len(cost_measurements):break
    if (local/'audit_engine_calls.json').exists():cost_calls+=json.loads((local/'audit_engine_calls.json').read_text())
    total_nodes=sum(c['actual_nodes'] for c in cost_calls)
    seconds_per_node=sum(m['cpu_seconds'] for m in cost_measurements)/max(1,total_nodes)
    cpu_forecast={k:1.25*v*seconds_per_node/3600 for k,v in node_forecast.items()}
    write_json(out/'cost_forecast.json',dict(run_id='B04A-N01',cost_roots_measured=len(cost_measurements),
      complete_COST32=len(cost_measurements)==32,measurements=cost_measurements,node_components=node_forecast,
      cpu_hours_pessimistic=cpu_forecast,seconds_per_node=seconds_per_node,forecast_multiplier=1.25,
      closure_reserve_fraction=.05,neural_costs=None if not selections else json.loads((local/'selector_role_costs.json').read_text()),
      neural_cost_reason='Unmeasured downstream training/confirmation; requires own fixed microbenchmark',
      training_challenge=None,confirmation_training=None,risk_cost_scope='all legal moves, 1600 dev / 4800 confirm, 200000 nodes each',
      cold_warm='Cold engine hash cleared per search; in-run audit cache keys bind complete history, nodes and namespace',
      B04B_authorized=False,methods_not_equal_compute=True,reuse_compute_match=False))
    actual_cpu=(time.process_time()-cpu_start+engine_cpu+child_cpu+allowances)/3600
    ledger=dict(stage='B04A',attempt_id='B04A-N01',inherited_actual=inherited,
      inherited_ledger_sha256=digest('artifacts/V2/B03R/resource_ledger.json'),
      stage_actual=dict(cpu_core_hours=actual_cpu,gpu_hours=gpu),
      project_actual=dict(cpu_core_hours=inherited['cpu_core_hours']+actual_cpu,gpu_hours=inherited['gpu_hours']+gpu),
      caps=dict(stage_cpu_core_hours=2,stage_gpu_hours=.25,project_cpu_core_hours=48,project_gpu_hours=20,disk_gib=80,external_spend_usd=0),
      remaining=dict(stage_cpu_core_hours=2-actual_cpu,stage_gpu_hours=.25-gpu,project_cpu_core_hours=48-inherited['cpu_core_hours']-actual_cpu,project_gpu_hours=20-inherited['gpu_hours']-gpu),
      work_units=work+[dict(work_unit_id='screen-python',cpu_seconds=time.process_time()-cpu_start,gpu_seconds=0),
        dict(work_unit_id='screen-stockfish',cpu_seconds=engine_cpu-(minimal['engine_cpu_seconds'] if minimal else 0),gpu_seconds=0)],
      allowances=[dict(work_unit_id='all-test-setup-cli-git-finalization',cpu_seconds=allowances,gpu_seconds=0)],
      measurement_method='all-thread process_time for Python; psutil process user+system seconds for Stockfish; evaluator process reported separately; CUDA resident wall',
      disk=dict(inherited_peak_gib=inherited_disk,new_stage_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
        peak_upper_bound_gib=inherited_disk+1),external_spend_usd=0)
    write_json(out/'resource_ledger.json',ledger)
    exit_code=0 if status=='DONE' else 4 if status=='RESOURCE_LIMIT' else 5
    write_json(out/'stage_disposition.json',dict(stage='B04A',execution_status=status,research_status=research,review_status='PENDING_REVIEW',reason=reason,exit_code=exit_code,
      permitted_downstream_uses=[] if status!='DONE' else ['Independent B04A review only; B04B requires Router acceptance']))
    return StageResult('B04A',status,research,out,exit_code,reason)
