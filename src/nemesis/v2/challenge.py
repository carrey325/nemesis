"""B05A immutable fair protocol and bounded challenge phases."""
import json
import math
from pathlib import Path
import time
import yaml
from nemesis.data.common import digest, write_json
from .types import StageResult
from .guards import require_new_output
from .engine import hash_object
from .matched_training import role_protocol, role_budgets, reference_reuse_audit

BASE='ebde118057f8760f7b9de4583b90e469818170af'
PLAN='sha256:f2ffc68dffe1a238cd284a58b96801eaaf89332afea54fa3bdda98bf4d188a7d'
DEV=('P006','P007','P013','P014','P025','P029','P030','P032')


def candidate_decision(*, fair, nll=None, windows=None, d=None, ds=None, safety=None, own_gain=None):
    if not fair: return 'METHOD_BUDGET_INCONCLUSIVE'
    values=[nll,*(windows or []),*(d or []),*(ds or []),*(own_gain or [])]
    if (len(windows or [])!=2 or len(d or [])!=2 or len(ds or [])!=2
        or len(own_gain or [])!=2 or any(v is None or not math.isfinite(v) for v in values)):
        return 'METHOD_INCONCLUSIVE'
    prediction=nll>=.005 and min(windows)>=0
    utility=min(d)>=.002 and min(ds)>=0 and safety is True and min(own_gain)>0
    if prediction and utility: return 'C01_UTILITY_UPGRADE'
    if prediction: return 'PREDICTION_ONLY'
    if utility: return 'TRADEOFF_NOT_AUTOMATIC_UPGRADE'
    return 'METHOD_INCONCLUSIVE'


def direct_effect(values, candidate_move, reference_move, evaluator_family):
    return values[evaluator_family][candidate_move]-values[evaluator_family][reference_move]


def verify_phase(out, phase):
    out=Path(out)
    lock_path=out/'challenge.lock.json'
    seal=json.loads((out/'lock_seal.json').read_text())
    if digest(lock_path)!=seal['lock_sha256']: raise ValueError('Challenge lock hash mismatch')
    lock=json.loads(lock_path.read_text())
    for path,expected in {**lock['inputs'],**lock['source_hashes']}.items():
        if digest(path)!=expected: raise ValueError('Challenge source/input drift: '+path)
    if phase=='evaluate':
        train=out/'train/phase_seal.json'
        if not train.exists(): raise PermissionError('Evaluation requires sealed train phase')
        record=json.loads(train.read_text())
        if record['lock_sha256']!=digest(lock_path): raise ValueError('Training phase belongs to other lock')
        for name in ('candidate_registry.json','matched_reference_registry.json'):
            if digest(out/name)!=record['registries'][name]: raise ValueError('Registry seal mismatch')
        for name,sha in record['outputs'].items():
            if digest(out/name)!=sha: raise ValueError('Training closure evidence mismatch')
        if record['research_status']=='METHOD_BUDGET_INCONCLUSIVE':
            # This authorizes only the null closure, never opening H or S_compare.
            return lock
        if not record.get('registries_complete'): raise PermissionError('Incomplete fair registries')
    return lock


def lock_challenge(config, assets, panels, out):
    from nemesis.models.train import load_rows, rows_hash
    cfg=yaml.safe_load(Path(config).read_text(encoding='utf8'))
    from .assets import verify_plan_config
    if verify_plan_config(config)!=PLAN: raise ValueError('Exact A03 plan/config required')
    if cfg['challenge']['candidate_id']!='C01' or cfg['challenge']['rank']!=4 or cfg['challenge']['transformer_layer_indices']!=[6,7]:
        raise ValueError('Challenge config differs from approved C01')
    if cfg['resources']['stage_caps']['B05A']!={'cpu_core_hours':5,'gpu_hours':4.5}:
        raise ValueError('B05A fixed cap mismatch')
    decision=Path('artifacts/V2/B04B/pilot_decision.json')
    if digest(decision)!='0acceba3de2a9bbbaebe4fa1d05efd4d4801ca3a6410727b59987becd9d1e22f':
        raise ValueError('Approved B04B pilot decision mismatch')
    if json.loads(decision.read_text())['challenge']!='BOUNDED_CHALLENGE_NEEDED':
        raise PermissionError('No challenge admission')
    inherited_path=Path('artifacts/V2/B04B/resource_ledger.json')
    inherited=json.loads(inherited_path.read_text())
    from .resources import validate_ledger
    validate_ledger(inherited)
    if inherited['project_actual']!={'cpu_core_hours':15.139727918042077,'gpu_hours':4.955804963630877}:
        raise ValueError('Inherited B04B ledger differs')
    manifest=json.loads(Path(panels).read_text())
    compare=Path(panels).parent/manifest['panels']['S_compare']['path']
    if digest(compare)!=manifest['panels']['S_compare']['sha256']: raise ValueError('S_compare panel hash mismatch')
    if digest('artifacts/B03/model_registry.json')!=json.loads(Path(assets).read_text())['registry_hash']:
        raise ValueError('Approved legacy registry hash mismatch')
    # Protocol freeze needs lineage/label metadata, not all million full histories.
    # Column projection preserves the exact B03 row order and draws hash.
    import pyarrow.parquet as pq
    split=json.loads(Path('artifacts/B01/split_manifest.json').read_text())
    from nemesis.models.train import group_players
    players=group_players(split)
    cols=['sample_id','game_id','player_id','split','input_hash','target_move_uci']
    b01=json.loads(Path('artifacts/B01/artifact_manifest.json').read_text())
    for name in ('samples.parquet','split_manifest.json'):
        expected=next(r for r in b01['files'] if Path(r['path']).name==name)['sha256']
        if digest(Path('artifacts/B01')/name)!=expected: raise ValueError('B01 source drift')
    def read_protocol(partition):
        rows=pq.read_table('artifacts/B01/samples.parquet',columns=cols,filters=[
            ('player_id','in',players),('split','=',partition),('main_eligible','=',True)]).to_pylist()
        from nemesis.data.common import append_json
        append_json(out/'access.jsonl',dict(actor='B05A-lock',purpose='same game bootstrap/row hashes',
            split=partition,cohort='all32',rows=len(rows),columns=cols,time_unix=time.time(),
            path='artifacts/B01/samples.parquet',input_hash=digest('artifacts/B01/samples.parquet')))
        return rows
    fit=read_protocol('fit_a');cal=read_protocol('calibration')
    if tuple(sorted(split['dev_players']))!=DEV: raise ValueError('Fixed development identities differ')
    protocols=[role_protocol(fit,None,1729)]+[role_protocol(fit,p,s) for p in DEV for s in (1729,2718,3141)]
    for r in protocols:
        selected=[x for x in cal if r['player'] is None or x['player_id']==r['player']]
        r['calibration_rows']=len(selected);r['calibration_sha256']=rows_hash(selected)
    budgets=role_budgets(protocols)
    inputs={str(p):digest(p) for p in (Path(config),Path(assets),Path(panels),decision,inherited_path,
        Path('artifacts/B01/samples.parquet'),Path('artifacts/B01/split_manifest.json'),
        Path('artifacts/B00/source_lock.json'),Path('weights/maia3-23m.pt'),Path('artifacts/B03/model_registry.json'),compare)}
    for path in ('artifacts/V2/B04A/cost_forecast.json','artifacts/V2/B04A/state_audit_extract.json',
                 'artifacts/V2/B05A/shared_forecast_probe.json',
                 'artifacts/V2/B05A/engineering02/group_engineering_benchmark.json',
                 'artifacts/V2/B05A/engineering02/personal_engineering_benchmark.json'):
        inputs[path]=digest(path)
    sources={str(p):digest(p) for p in sorted(Path('src/nemesis').rglob('*.py'))}
    vendor=Path('vendor/maia3-1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py')
    sources[str(vendor)]=digest(vendor)
    lock=dict(stage='B05A',run_id='B05A-N01',lock_revision=out.name,base_commit=BASE,plan_version=PLAN,
        inputs=inputs,source_hashes=sources,config=cfg,created_unix=time.time(),
        structures={'A00_MATCHED':'final_square_residual_rank16_no_base_training',
            'C01':'packed_QV_blocks6_7_rank4_scale1_no_residual'},
        role_protocols=protocols,role_budgets=budgets,base_checkpoint=digest('weights/maia3-23m.pt'),
        evaluator_registry_sha256=digest('artifacts/B03/model_registry.json'),
        compare_panel_sha256=digest(compare),compare_outcomes_open=False,
        confirmation_H_open=False,recipe_count_each_arm=1,primary_lambda=0,
        group_scope='all32 Fit-A; equal game then player',
        cache_strategy='stream full FP32 inputs; optional per-role prefix bounded2GiB; no personalized sharing')
    write_json(out/'challenge.lock.json',lock)
    write_json(out/'role_budgets.json',budgets)
    write_json(out/'reference_reuse_audit.json',reference_reuse_audit(dict(
        registry_sha256=inputs[str(Path('artifacts/B03/model_registry.json'))],historical_per_role_costs=None)))
    write_json(out/'lock_seal.json',dict(lock_sha256=digest(out/'challenge.lock.json'),run_id=lock['run_id']))
    return StageResult('B05A','DONE','PROTOCOL_LOCKED',out,0,'Fair protocol frozen; no challenge outcomes opened')


def _run_challenge(config:Path, phase:str, assets:Path, panels:Path, out:Path)->StageResult:
    require_new_output(out);out=Path(out)
    if phase not in ('lock','train','evaluate'): raise ValueError('Unknown challenge phase')
    if phase=='lock':
        if (out/'challenge.lock.json').exists(): raise FileExistsError('Existing challenge lock cannot be overwritten')
        out.mkdir(exist_ok=True,parents=True)
        return lock_challenge(config,assets,panels,out)
    lock=verify_phase(out,phase)
    if any(lock['inputs'][str(p)]!=digest(p) for p in (config,assets,panels)):
        raise ValueError('CLI inputs differ from challenge lock')
    folder=out/phase
    if folder.exists(): raise FileExistsError('Completed or failed phase cannot be overwritten')
    folder.mkdir()
    if phase=='train':
        from .challenge_benchmark import run_benchmark
        return run_benchmark(out,lock,folder)
    from .challenge_report import close_inconclusive
    return close_inconclusive(out,lock,folder)


def run_challenge(config:Path, phase:str, assets:Path, panels:Path, out:Path)->StageResult:
    tick=time.time();cpu=time.process_time()
    try:
        return _run_challenge(config,phase,assets,panels,out)
    except (FileExistsError,FileNotFoundError,ValueError,PermissionError,RuntimeError):
        raise
    except Exception as exc:
        raise RuntimeError('unexpected challenge failure: '+repr(exc)) from exc
    finally:
        # A rejected retry adds a new receipt and cannot reset prior expenses.
        if Path(out).resolve().is_relative_to(Path('artifacts/V2').resolve()):
            from nemesis.data.common import append_json
            append_json(Path(out)/'phase_resource_receipts.jsonl',dict(work_unit_id=f'B05A-{phase}-{tick}',
                owner_stage='B05A',start_unix=tick,end_unix=time.time(),cpu_seconds=time.process_time()-cpu,
                gpu_seconds=0,phase=phase,measurement='CPU-only phase; CUDA children/engineering separately charged'))
