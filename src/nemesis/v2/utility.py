"""Authorized B04B phased utility run; no downstream training or holdout access."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
import chess
import pyarrow.parquet as pq
import yaml
from nemesis.data.common import digest,write_json
from .guards import require_new_output
from .utility_inputs import load_inputs
from .utility_compute import ComputeMeter,compute_phase,root_nodes,ResourceLimit
from .engine import hash_object
from .types import CandidateBundle,Selection,StageResult

def run_utility(config,assets,panels,out):
    require_new_output(out);out=Path(out)
    if (out/'run_binding.json').exists():raise FileExistsError('Existing run requires explicit verified checkpoint continuation; cannot reset ledger')
    out.mkdir(exist_ok=True,parents=True);(out/'local').mkdir(exist_ok=True)
    rows,bundles,risk,inherited=load_inputs(config,assets,panels,out)
    cfg=yaml.safe_load(Path(config).read_text(encoding='utf8'))
    meter=ComputeMeter(out,inherited)
    binding=dict(attempt_id='B04B-N01',base_commit='18d356107a70af279431a2e9839c3c20996f3e6d',
        plan_version='sha256:f2ffc68dffe1a238cd284a58b96801eaaf89332afea54fa3bdda98bf4d188a7d',
        config_sha256=digest(config),assets_sha256=digest(assets),panels_sha256=digest(panels),
        source_hashes={str(p):digest(p) for p in sorted(Path('src/nemesis/v2').glob('*.py'))},start_unix=time.time())
    write_json(out/'run_binding.json',binding)
    ordered=rows['S_dev']+rows['TUNE'];ns=[root_nodes(r,bundles[r['root_id']]) for r in ordered]
    cost=json.loads(Path('artifacts/V2/B04A/cost_forecast.json').read_text())
    # Preserve original14.77; updated forecast appends frozen exact branching.
    rate=cost['seconds_per_node']*(4.734375/4.4375)
    cfg['_seconds_per_node']=rate
    risk_nodes=sum(chess.Board(r['fen6']).legal_moves.count()*200000 for r in risk)
    engine_nodes=sum(n['decision']+n['audit']+n['sensitivity'] for n in ns)+risk_nodes
    forecast=1.25*engine_nodes*rate/3600+.2842456427015251+meter.allowance/3600
    write_json(out/'cost_forecast.json',dict(original_B04A_forecast_cpu_hours=14.767714640228304,
        append_A03_frozen_exact_cpu_hours=forecast,stage_cap=11,reserve=.55,engine_nodes_upper_scenario=engine_nodes,
        risk_exact_nodes=risk_nodes,risk_sample_lock_sha256=digest(out/'risk_sample.lock.json'),
        seconds_per_node=rate,source_cost_sha256=digest('artifacts/V2/B04A/cost_forecast.json'),
        ratio_from_short_two_process_benchmark=4.734375/4.4375,not_long_run_guarantee=True,
        forecast_multiplier=1.25,neural_cpu_with_margin=.2842456427015251,per_root_nodes=ns))
    status='DONE';reason='Complete eight-player utility pilot';research='DEV_PILOT_COMPLETE'
    def gpu_actor(phase,request,output,estimated=300):
        meter.gate(cpu=estimated,gpu=estimated)
        path=out/'local'/(phase+'_request.json');write_json(path,request)
        tick=time.time()
        with (out/(phase+'.raw.log')).open('w',encoding='utf8') as log:
            child=subprocess.run([sys.executable,'-m','nemesis.v2.utility_actor',phase,str(path),str(output)],stdout=log,stderr=subprocess.STDOUT)
        receipt_path=path.with_suffix('.cost.json')
        if receipt_path.exists():receipt=json.loads(receipt_path.read_text())
        else:receipt=dict(cpu_seconds=(time.time()-tick)*12,resident_gpu_seconds=time.time()-tick+5,start_unix=tick,end_unix=time.time(),reason='Failed child conservative full12-core wall bound')
        meter.child_cpu+=receipt['cpu_seconds'];meter.gpu+=max(receipt['resident_gpu_seconds'],time.time()-tick)
        meter.units.append(dict(work_unit_id='GPU-'+phase,owner_stage='B04B',input_hash=digest(path),**receipt,exit_code=child.returncode))
        meter.save()
        if child.returncode:raise RuntimeError(phase+' CUDA actor failed; see raw log')
        return json.loads((output/'selector_output.json' if phase=='selector' else output).read_text(encoding='utf8'))
    try:
        if forecast>10.45 or inherited['project_actual']['cpu_core_hours']+forecast>52:
            raise ResourceLimit('Frozen exact full remaining forecast exceeds stage/project gate')
        meter.gate(cpu=900,gpu=1200)
        decisions=compute_phase(ordered,bundles,cfg,meter,'decision')
        complete={rid:CandidateBundle(**dict(b['bundle'],moves=tuple(b['bundle']['moves']),safe_moves=tuple(b['bundle']['safe_moves']))) for rid,b in decisions.items()}
        selected=gpu_actor('selector',dict(roots=ordered,bundles={rid:asdict(b) for rid,b in complete.items()},
          matched={rid:b['measure'] for rid,b in decisions.items()},registry='artifacts/B03/model_registry.json'),out/'local/selector',estimated=600)
        sels=[Selection(**s) for s in selected['selections']];seal=out/'local/selector/selection_seal.json'
        audits=compute_phase(ordered,complete,cfg,meter,'audit',seal=seal,selections=sels,traces=selected['traces'])
        evaluation=gpu_actor('evaluator',dict(roots=ordered,audits=audits,selections=selected['selections'],traces=selected['traces'],
          selector_pid=selected['selector_pid'],seal=str(seal),registry='artifacts/B03/model_registry.json'),out/'local/evaluator_output.json',estimated=600)
        risk_values=compute_phase(risk,{},cfg,meter,'risk')
        # Targets and real Elo are opened only for the exact frozen sample IDs.
        selected_ids={r['sample_id'] for r in risk}
        targets=pq.read_table('artifacts/B01/samples.parquet',filters=[('player_id','in',list(__import__('nemesis.v2.panels',fromlist=['DEV']).DEV)),('split','in',['H1','H2']),('sample_id','in',list(selected_ids))]).to_pylist()
        if {r['sample_id'] for r in targets}!=selected_ids:raise ValueError('Frozen D6 targets missing')
        rp=gpu_actor('risk',dict(rows=targets,values=risk_values,registry='artifacts/B03/model_registry.json'),out/'local/risk_predictions.json',estimated=200)
        from .utility_report import build_outputs
        build_outputs(out,rows,complete,selected,audits,evaluation,targets,risk_values,rp,meter)
    except ResourceLimit as exc:status='RESOURCE_LIMIT';research='BUDGET_BLOCKED';reason=str(exc)
    except Exception as exc:
        status='FAILED';research='NOT_TESTED';reason=repr(exc)
        import traceback
        (out/'failure_traceback.log').write_text(traceback.format_exc(),encoding='utf8')
    finally:meter.save()
    code=0 if status=='DONE' else 4 if status=='RESOURCE_LIMIT' else 5
    write_json(out/'stage_disposition.json',dict(stage='B04B',execution_status=status,research_status=research,reason=reason,
        review_status='PENDING_REVIEW',exit_code=code,permitted_downstream_uses=['Router independent review only']))
    return StageResult('B04B',status,research,out,code,reason)
