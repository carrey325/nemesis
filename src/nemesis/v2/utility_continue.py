"""Explicit continuation of immutable decision/selector checkpoints after diagnosis."""
import json
from pathlib import Path
import time
import subprocess
import sys
import chess
import pyarrow.parquet as pq
import yaml
from nemesis.data.common import write_json,digest
from .assets import verify_plan_config,A03_PLAN
from .resources import load_a03_inherited,validate_ledger
from .types import CandidateBundle,Selection
from .engine import hash_object
from .utility_compute import ComputeMeter,compute_phase,root_nodes,ResourceLimit
from .evaluate import verify_selection_seal
from .panels import DEV

def continue_utility(config,assets,panels,out):
    out=Path(out);binding=json.loads((out/'run_binding.json').read_text())
    if verify_plan_config(config)!=A03_PLAN or binding['config_sha256']!=digest(config) or binding['assets_sha256']!=digest(assets) or binding['panels_sha256']!=digest(panels):raise ValueError('Continuation lock mismatch')
    prior=json.loads((out/'resource_ledger.json').read_text());validate_ledger(prior)
    inherited=load_a03_inherited()
    if prior['inherited_actual']!=inherited['project_actual']:raise ValueError('Continuation would repeat overlay or reset history')
    meter=ComputeMeter(out,inherited);meter.prior_cpu=prior['stage_actual']['cpu_core_hours']*3600-meter.allowance
    meter.gpu=prior['stage_actual']['gpu_hours']*3600;meter.units=prior['work_units'][:]
    # Only decision and selector phases are eligible for this diagnosed continuation.
    if any(u['work_unit_id'].startswith(('audit-','risk-','GPU-evaluator','GPU-risk')) for u in meter.units):raise ValueError('Unexpected downstream checkpoint; cannot silently resume')
    if not (out/'local/checkpoint_stop.request').exists():raise ValueError('No explicit diagnosed checkpoint')
    snapshot=out/'checkpoint_history';snapshot.mkdir()
    for name in ('resource_ledger.json','stage_disposition.json','cost_forecast.json','failure_traceback.log'):
        path=out/name
        if path.exists():(snapshot/name).write_bytes(path.read_bytes())
    base=Path(panels).parent;pm=json.loads(Path(panels).read_text());rows={}
    for panel in ('S_dev','TUNE','N_dev'):
        record=pm['panels'][panel];path=base/record['path']
        if digest(path)!=record['sha256']:raise ValueError('Continuation panel changed')
        rows[panel]=pq.read_table(path).to_pylist()
    screens={r['root_id']:r for r in pq.read_table(base/'screening_log.parquet').to_pylist() if r['panel']=='N_dev'}
    for r in rows['N_dev']:r.update(status=screens[r['root_id']]['status'])
    decisions={};units={u['work_unit_id']:u for u in meter.units}
    for row in rows['S_dev']+rows['TUNE']:
        rid=row['root_id'];path=out/'local/decision'/(hash_object(rid)+'.json')
        if digest(path)!=units['decision-'+rid]['output_sha256']:raise ValueError('Decision checkpoint hash mismatch')
        body=json.loads(path.read_text());decisions[rid]=body;row['cold_screen_actual_nodes']=body['cold_screen_actual_nodes']
    complete={rid:CandidateBundle(**dict(b['bundle'],moves=tuple(b['bundle']['moves']),safe_moves=tuple(b['bundle']['safe_moves']))) for rid,b in decisions.items()}
    if digest(out/'local/selector/selector_output.json')!=units['GPU-selector']['output_sha256']:raise ValueError('Selector output checkpoint hash mismatch')
    selected=json.loads((out/'local/selector/selector_output.json').read_text());sels=[Selection(**s) for s in selected['selections']];seal=out/'local/selector/selection_seal.json'
    verify_selection_seal(sels,seal,traces=selected['traces'],require_trace=True)
    if selected['selection_seal_sha256']!=digest(seal):raise ValueError('Selector output seal mismatch')
    risk=pq.read_table(out/'local/risk_states.parquet').to_pylist();lock=json.loads((out/'risk_sample.lock.json').read_text())
    if digest(out/'risk_sample.lock.json')!=json.loads((out/'input_verification.json').read_text())['risk_sample_lock_sha256']:raise ValueError('D6 lock changed')
    if hash_object(risk)!=lock['sample_state_hash']:raise ValueError('D6 frozen states changed')
    cfg=yaml.safe_load(Path(config).read_text(encoding='utf8'));old=json.loads((out/'cost_forecast.json').read_text())
    cfg['_seconds_per_node']=old['seconds_per_node']
    audit_nodes=sum(root_nodes(row,complete[row['root_id']])['audit']+root_nodes(row,complete[row['root_id']])['sensitivity'] for row in rows['S_dev']+rows['TUNE'])
    risk_nodes=sum(chess.Board(r['fen6']).legal_moves.count()*200000 for r in risk)
    forecast=(meter.actual_cpu()+meter.allowance)/3600+1.25*(audit_nodes+risk_nodes)*cfg['_seconds_per_node']/3600+.2842456427015251
    write_json(out/'continuation_binding.json',dict(config_sha256=digest(config),prior_ledger_sha256=digest(snapshot/'resource_ledger.json'),
        risk_sample_lock_sha256=digest(out/'risk_sample.lock.json'),selector_seal_sha256=digest(seal),
        source_sha256={str(p):digest(p) for p in Path('src/nemesis/v2').glob('*.py')},
        diagnosed_change='Restore original B04A sensitivity hash salt and16-digit prefix;33 S roots,5 TUNE roots; no choices/audits changed',
        forecast_full_stage_cpu_hours=forecast,original_14_77_preserved=True,prior_cpu_recharged=False,overlay_reapplied=False))
    def gpu(phase,request,output,estimate):
        meter.gate(cpu=estimate,gpu=estimate);request_path=out/'local'/(phase+'_request.json');write_json(request_path,request)
        tick=time.time()
        with (out/(phase+'.raw.log')).open('w',encoding='utf8') as log:
            child=subprocess.run([sys.executable,'-m','nemesis.v2.utility_actor',phase,str(request_path),str(output)],stdout=log,stderr=subprocess.STDOUT)
        receipt_path=request_path.with_suffix('.cost.json')
        receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else dict(cpu_seconds=12*(time.time()-tick),resident_gpu_seconds=time.time()-tick+5,reason='Failed child inclusive full12-thread wall bound')
        receipt['gpu_seconds']=max(receipt['resident_gpu_seconds'],time.time()-tick)
        meter.child_cpu+=receipt['cpu_seconds'];meter.gpu+=receipt['gpu_seconds']
        meter.units.append(dict(work_unit_id='GPU-'+phase,owner_stage='B04B',input_hash=digest(request_path),**receipt,exit_code=child.returncode));meter.save()
        if child.returncode:raise RuntimeError('CUDA '+phase+' failed; see raw log')
        return json.loads(output.read_text())
    status='DONE';reason='B04B complete';research='DEV_PILOT_COMPLETE'
    try:
        if forecast>10.45:raise ResourceLimit('Corrected frozen remaining full-stage forecast exceeds CPU reserve gate')
        audits=compute_phase(rows['S_dev']+rows['TUNE'],complete,cfg,meter,'audit',seal=seal,selections=sels,traces=selected['traces'])
        evaluation=gpu('evaluator',dict(roots=rows['S_dev']+rows['TUNE'],audits=audits,selections=selected['selections'],traces=selected['traces'],
          selector_pid=selected['selector_pid'],seal=str(seal),registry='artifacts/B03/model_registry.json'),out/'local/evaluator_output.json',600)
        risk_values=compute_phase(risk,{},cfg,meter,'risk')
        selected_ids=[r['sample_id'] for r in risk]
        targets=pq.read_table('artifacts/B01/samples.parquet',filters=[('player_id','in',list(DEV)),('split','in',['H1','H2']),('sample_id','in',selected_ids)]).to_pylist()
        if {r['sample_id'] for r in targets}!=set(selected_ids):raise ValueError('D6 target coverage mismatch')
        rp=gpu('risk',dict(rows=targets,values=risk_values,registry='artifacts/B03/model_registry.json'),out/'local/risk_predictions.json',200)
        from .utility_report import build_outputs
        build_outputs(out,rows,complete,selected,audits,evaluation,targets,risk_values,rp,meter)
    except ResourceLimit as exc:status='RESOURCE_LIMIT';reason=str(exc);research='BUDGET_BLOCKED'
    except Exception as exc:
        status='FAILED';reason=repr(exc);research='NOT_TESTED'
        import traceback
        (out/'continuation_failure_traceback.log').write_text(traceback.format_exc(),encoding='utf8')
    finally:meter.save()
    code=0 if status=='DONE' else 4 if status=='RESOURCE_LIMIT' else 5
    write_json(out/'stage_disposition.json',dict(stage='B04B',execution_status=status,research_status=research,reason=reason,exit_code=code,review_status='PENDING_REVIEW',permitted_downstream_uses=['Router independent review only']))
    print(status+': '+reason,flush=True)
    return code

if __name__=='__main__':
    raise SystemExit(continue_utility(Path('nemesis-stage1-plan/configs/experiment.v2.proposed.yaml'),Path('artifacts/V2/B03R/approved_assets.json'),Path('artifacts/V2/B04A/panels_manifest.json'),Path('artifacts/V2/B04B')))
