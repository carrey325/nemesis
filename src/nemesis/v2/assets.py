"""Read-only B03 asset audit; evidence is a candidate, never an approval."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import zipfile
import yaml
from nemesis.data.common import digest,write_json
from .types import StageResult
from .guards import require_new_output

PLAN='sha256:592b95df1e6414dbd3fbbe40fb280ccf02ecd2b4c10912d80c06dc016e9118f4'
A03_PLAN='sha256:f2ffc68dffe1a238cd284a58b96801eaaf89332afea54fa3bdda98bf4d188a7d'
SOURCE='65fe54e1af379c2b7dbcb116f21a786d4607ba87'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def verify_plan_config(config):
    root=Path('nemesis-stage1-plan'); manifest=read(root/'MANIFEST.json')
    hashes={r['path']:r['sha256'] for r in manifest['files']}
    for path,sha in hashes.items():
        if digest(root/path)!=sha: raise ValueError('Bound plan bytes differ: '+path)
    effective=''.join(p+'\t'+hashes[p]+'\n' for p in sorted(manifest['effective_plan_files']))
    actual='sha256:'+hashlib.sha256(effective.encode()).hexdigest()
    if actual not in (PLAN,A03_PLAN) or manifest['plan_version']!=actual: raise ValueError('V2 plan digest differs')
    if digest(config)!=hashes['configs/experiment.v2.proposed.yaml']: raise ValueError('Unapproved V2 config bytes')
    return actual

def check_inventory(registry,split):
    models=registry['models']; ids=[r['model_id'] for r in models]
    dev=split['dev_players']; confirm=split['confirm_players']
    expected={'G','E3_group','E2_group','G0','E2_base'}
    expected|={f'{role}_{p}_{s}' for p in dev+confirm for role,seeds in
        [('T',[1729,2718,3141]),('E3',[4242]),('E2',[6161])] for s in seeds}
    if len(ids)!=165 or set(ids)!=expected or len(set(ids))!=len(ids): raise ValueError('Incomplete/unknown model inventory')
    counts=Counter(r['status'] for r in models)
    if counts!={'TRAINED':43,'BASE':2,'NOT_TRAINED':120}: raise ValueError('Unexpected trained/base/confirmation counts')
    for r in models:
        if r['cohort']=='confirm' and r['status']!='NOT_TRAINED': raise ValueError('Confirmation must remain untrained')
        if r['cohort']=='dev' and r['status'] not in ('TRAINED','BASE'): raise ValueError('Missing dev asset')
    if registry.get('confirmation_H_opened') is not False: raise ValueError('Holdout seal missing')
    return dict(trained=43,bases=2,confirm_trained=0,confirm_not_trained=120)

def check_record_lineage(record,registry,root):
    root=Path(root); m=record['metadata']; mid=record['model_id']
    compatibility=read(root/'source_compatibility.json')
    expected_group={'T':'G','E3':'E3_group','E2':'E2_group'}
    role=record['role'] if record['identity'] else {'G':'T','E3_group':'E3','E2_group':'E2'}[mid]
    base=next(r for r in registry['models'] if r['model_id']==('E2_base' if record['family']=='maia2' else 'G0'))
    if m['base_sha']!=base['base_sha'] or m['split']!=('fit_a' if role=='T' else 'fit_b'):
        raise ValueError('Wrong base/split lineage: '+mid)
    if m['code_sha'] not in {compatibility['old_code_sha'],compatibility['new_code_sha']}:
        raise ValueError('Unknown training code lineage: '+mid)
    if m['model_id']!=mid or m['family']!=record['family'] or m['player_id']!=record['identity']:
        raise ValueError('Wrong model/identity lineage: '+mid)
    if m['config_sha']!=registry['config_sha'] or m['source_sha']!=read(root/'pooled_lineage_audit.json')['source_sha256']:
        raise ValueError('Wrong config/source lineage: '+mid)
    if record['identity'] and record['group']!=expected_group[role]: raise ValueError('Wrong group lineage: '+mid)
    calibration=read(root/'calibration.json')
    expected_alpha=calibration['models'][mid]['alpha'] if record['identity'] else 1.
    if record['temperature']!=calibration['unified_temperature'] or record['alpha']!=expected_alpha:
        raise ValueError('Wrong Calibration temperature/alpha lineage: '+mid)
    if not record['base_unchanged'] or record['gradient_max']<=0: raise ValueError('Invalid freeze/gradient record')
    import torch
    checkpoint=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
    if checkpoint['metadata']!=m: raise ValueError('Checkpoint metadata differs: '+mid)
    if not all(torch.isfinite(t).all() for t in checkpoint['parameters'].values()): raise ValueError('Nonfinite parameters')
    training=read(Path(record['checkpoint']).parent/'training.json')
    if training['metadata']!=m or training['base_state_before']!=training['base_state_after']:
        raise ValueError('Training receipt differs: '+mid)
    return dict(model_id=mid,metadata=m,checkpoint_sha256=digest(record['checkpoint']),
        training_receipt_sha256=digest(Path(record['checkpoint']).parent/'training.json'),
        base_unchanged=True,parameters_finite=True,group=record.get('group'),
        training_players=record['training_players'],temperature=record['temperature'],alpha=record['alpha'])

def replay_metrics(root,out):
    """Independent game-weighted aggregation of every frozen identity/window cell."""
    import pyarrow.parquet as pq
    root=Path(root); matrix=pq.read_table(root/'validation/dev_prediction_matrix.parquet').to_pylist()
    split=read('artifacts/B01/split_manifest.json'); players=sorted(split['dev_players'])
    mids={r['model_id'] for r in matrix}
    if len(matrix)!=848 or len(mids)!=53: raise ValueError('Incomplete development matrix')
    for mid in mids:
        cells=[r for r in matrix if r['model_id']==mid]
        if len(cells)!=16 or {(r['player_id'],r['split']) for r in cells}!={(p,w) for p in players for w in ['H1','H2']}:
            raise ValueError('Missing/duplicate matrix cell: '+mid)
    def nll(mid,p):
        cells=[r for r in matrix if r['model_id']==mid and r['player_id']==p]
        return sum(r['nll']*r['games'] for r in cells)/sum(r['games'] for r in cells)
    per=[]
    for p in players:
        per.append(dict(player_id=p,T=nll('G',p)-nll('T_ensemble_'+p,p),
            E3_group=nll('E3_group',p)-nll('E3_'+p+'_4242',p),
            E2_group=nll('E2_group',p)-nll('E2_'+p+'_6161',p),
            E2_base=nll('E2_base',p)-nll('E2_'+p+'_6161',p),
            E2_group_vs_base=nll('E2_base',p)-nll('E2_group',p)))
    result=dict(matrix_rows=len(matrix),models=len(mids),players=per,confirmation_H_reads=0,
        t_mean_nll_improvement=sum(r['T'] for r in per)/8,
        E3_vs_group=sum(r['E3_group'] for r in per)/8,E2_vs_group=sum(r['E2_group'] for r in per)/8,
        E2_vs_base=sum(r['E2_base'] for r in per)/8,
        input_sha256=digest(root/'validation/dev_prediction_matrix.parquet'),
        scope='cached development matrix arithmetic; hardware probability replay recorded separately')
    gate=read(root/'dev_gate.json')
    for key in ['t_mean_nll_improvement','E3_vs_group','E2_vs_group','E2_vs_base']:
        if abs(result[key]-gate[key])>1e-10: raise ValueError('Gate arithmetic mismatch: '+key)
    if abs(result['t_mean_nll_improvement']-.04682818933853605)>1e-10: raise ValueError('Bound development result differs')
    return result

def validate_readonly(root,out):
    from nemesis.models.train import load_rows,rows_hash
    root=Path(root); out=Path(out)
    rows,split=load_rows(Path('artifacts/B01'),'dev',('H1','H2'),out/'access_log.jsonl')
    if rows_hash(rows)!=read(root/'lineage_audit.json')['row_hashes']['dev_H']: raise ValueError('Development row lineage mismatch')
    metrics=replay_metrics(root,out)
    if len(rows)!=122593: raise ValueError('Development decision count differs')
    return dict(status='PASS',players=len(split['dev_players']),decisions=len(rows),splits=['H1','H2'],
        matrix_rows=metrics['matrix_rows'],confirmation_H_reads=0,rows_sha256=rows_hash(rows),
        original_B03_writes=0,probability_replay='validation/registry_runtime_replay.json')

def reconcile_cost(root):
    old=read(Path(root)/'resource_ledger.json')
    return dict(inherited_actual=dict(cpu_core_hours=old['project_cpu_core_hours'],gpu_hours=old['project_gpu_hours'],disk_gib=old['disk_gib']),
        historical_B03=dict(cpu_core_hours=old['cpu_core_hours'],gpu_hours=old['gpu_hours']),
        report_cpu_core_hours=3.902651909722444,ledger_tail_cpu_core_hours=old['cpu_core_hours']-3.902651909722444,
        roles={r:dict(cpu_core_hours=None,gpu_hours=None,status='unattributed',reason='Aggregate runner/test receipts do not separate encoding/training/validation per role')
            for r in ['public_encoding','features','group','T','E3','E2','validation']},
        reuse_compute_match=False,reason='No per-model capped stopping protocol or reliable cost attribution; use A00_MATCHED only after B05A budget gate',
        history_charged_again=False,source_sha256=digest(Path(root)/'resource_ledger.json'))

def candidate_disposition():
    return dict(stage_id='B03R',execution_status='DONE',research_status='ASSET_CANDIDATE_ONLY',review_status='PENDING_REVIEW',
        original_B03='INCOMPLETE_NOT_ACCEPTED',assets_approved=False,permitted_downstream_use='B04A only after independent B03R approval',
        confirmation_models='NOT_TRAINED',confirmation_H='SEALED',old_AC05_migrated_to='B06A')

def verify_source(root):
    """Verify original archive bytes and legacy training core against current source."""
    from nemesis.models.source_compatibility import definitions
    root=Path(root); receipt=read(root/'run_receipt.json'); compat=read(root/'source_compatibility.json')
    archives=[root/'training_source.zip',*sorted((root/'history').rglob('training_source.zip'))]
    old_archive=next((p for p in archives if digest(p)==compat['archive_sha256']),None)
    if old_archive is None: raise ValueError('Original training archive missing/hash mismatch')
    exceptions={'src/nemesis/__main__.py':{'main'},'src/nemesis/models/twins.py':{'run_training','run_validation','train_confirmation','evaluate_registry'},
        'src/nemesis/models/train.py':{'fit_adapter'},'src/nemesis/models/personal_predictor.py':{'registry_record'}}
    checks=[]
    with zipfile.ZipFile(old_archive) as archive:
        hashes=json.loads(archive.read('source_hashes.json'))
        if hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()!=compat['old_code_sha']: raise ValueError('Archive source digest mismatch')
        for name,sha in hashes.items():
            path=Path(name); old=archive.read(path.as_posix()); new=path.read_bytes()
            if hashlib.sha256(old).hexdigest()!=sha: raise ValueError('Archived bytes differ')
            before=definitions(old); after=definitions(new)
            for fn,body in before.items():
                if fn not in exceptions.get(path.as_posix(),set()) and after.get(fn)!=body:
                    raise ValueError('Training core changed: '+name+':'+fn)
            checks.append(dict(path=path.as_posix(),archived_sha256=sha,current_sha256=digest(path),
                unchanged_core=True,excluded=sorted(exceptions.get(path.as_posix(),set()))))
    with zipfile.ZipFile(root/'training_source.zip') as latest:
        hashes=json.loads(latest.read('source_hashes.json'))
        if hashes!=receipt['source_hashes'] or hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()!=receipt['code_sha']:
            raise ValueError('Latest source archive differs from frozen run receipt')
        for name,sha in hashes.items():
            if hashlib.sha256(latest.read(Path(name).as_posix())).hexdigest()!=sha: raise ValueError('Latest archived bytes mismatch')
    return dict(status='PASS',archive_sha256=digest(root/'training_source.zip'),original_archive_path=str(old_archive),
        original_archive_sha256=digest(old_archive),source_commit=SOURCE,
        legacy_compatibility_sha256=digest(root/'source_compatibility.json'),checks=checks,
        exceptions_reason='V2 CLI/permission guards, existing audited public-fit and gate orchestration repairs; training math unchanged')

def audit_assets(config:Path,registry:Path,out:Path)->StageResult:
    require_new_output(out)
    verify_plan_config(config)
    out=Path(out); root=Path(registry).parent
    # Logs may already live in the stage root; audit owns a new named phase.
    phase=out/'audit'
    if phase.exists(): raise FileExistsError('Audit phase already exists')
    c=yaml.safe_load(Path(config).read_text(encoding='utf8'))
    if c['schema_version']!=2 or c['lineage']['source_commit']!=SOURCE: raise ValueError('Unknown V2 binding')
    if digest(c['lineage']['legacy_config_path'])!=c['lineage']['legacy_config_sha256']: raise ValueError('Legacy config differs')
    start=time.process_time(); wall=time.time(); phase.mkdir(parents=True)
    value=read(registry); split=read('artifacts/B01/split_manifest.json')
    inventory=check_inventory(value,split)
    # Hash every original B03 file including partial caches; never load sealed H.
    original=[dict(path=p.as_posix(),sha256=digest(p),bytes=p.stat().st_size,local_only=p.suffix in ('.pt','.npy','.parquet','.jsonl'))
        for p in sorted(root.rglob('*')) if p.is_file()]
    write_json(phase/'legacy_snapshot.json',original)
    lineage=[]
    for r in value['models']:
        if r['status']=='TRAINED':
            if digest(r['checkpoint'])!=r['checkpoint_sha256']: raise ValueError('Checkpoint hash mismatch')
            item=check_record_lineage(r,value,root)
            if r['identity'] is None and (r['training_players']!=sorted(split['dev_players']+split['confirm_players']) or r['metadata']['cohort']!='all_targets'):
                raise ValueError('Group must preserve all32 training scope')
            lineage.append(item)
    runtime=read('vendor/runtime_lock.json')
    for mid,key in [('G0','maia3_23m'),('E2_base','maia2_blitz')]:
        record=next(r for r in value['models'] if r['model_id']==mid)
        base=runtime['models'][key]
        if base['sha256']!=record['base_sha'] or digest(base['path'])!=record['base_sha']: raise ValueError('Base hash mismatch')
    costs=reconcile_cost(root)
    write_json(out/'resource_reconciliation.json',costs)
    write_json(out/'cost_by_role.json',costs['roles'])
    write_json(out/'asset_lineage.json',dict(models=lineage,source=verify_source(root),
        independent_split_audit=read(root/'pooled_lineage_audit.json'),legacy_config_sha256=digest(c['lineage']['legacy_config_path'])))
    write_json(out/'reuse_registry.json',dict(registry_path=str(registry),registry_hash=digest(registry),source_commit=SOURCE,
        legacy_plan_version=value['plan_version'],plan_version=PLAN,models=value['models'],
        aliases={'E3_G':'E3_group','E2_G':'E2_group','E2_BASE':'E2_base'},mutation_of_original=False))
    write_json(out/'cache_inventory.json',dict(files=[r for r in original if '/cache/' in r['path'] or '/logits/' in r['path']],
        partial_confirmation_is_not_training=True,cold_equivalent_cost=None,cold_equivalent_reason='unattributed historical aggregate'))
    write_json(out/'dev_metrics_replay.json',replay_metrics(root,out))
    write_json(out/'approved_assets.json',dict(**inventory,assets_approved=False,review_status='PENDING_REVIEW',registry_hash=digest(registry),
        source_commit=SOURCE,plan_version=PLAN,candidate_asset_ids=[r['model_id'] for r in value['models'] if r['status'] in ('BASE','TRAINED')]))
    write_json(out/'stage_disposition.json',candidate_disposition())
    write_json(phase/'receipt.json',dict(status='PASS',start_unix=wall,end_unix=time.time(),cpu_seconds=time.process_time()-start,
        gpu_seconds=0,confirmation_H_reads=0,original_B03_writes=0,input_sha256=digest(registry),config_sha256=digest(config)))
    return StageResult('B03R','DONE','ASSET_CANDIDATE_ONLY',out,0,'Read-only audit complete; independent approval required')
