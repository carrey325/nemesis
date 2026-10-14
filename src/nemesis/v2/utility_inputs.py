"""Hash-bound B04A cache admission and D6 freeze, without evaluator outcomes."""
from dataclasses import asdict
import json
from pathlib import Path
import pyarrow.parquet as pq
from nemesis.data.common import digest,write_json
from .assets import PLAN,A03_PLAN,verify_plan_config
from .engine import hash_object,rebuild_root
from .panels import DEV,STATE_COLUMNS,root_context
from .resources import load_a03_inherited
from .risk import freeze_risk_rows
from .types import CandidateBundle

def load_inputs(config,assets,panels,out):
    plan=verify_plan_config(config)
    if plan!=A03_PLAN:raise ValueError('B04B requires A03 authorization')
    inherited=load_a03_inherited()
    approval=Path('workflow/reviews/lw-nemesis-stage1-B04A-R03/response.md').read_text(encoding='utf8')
    if not all(x in approval for x in ('a09984926f28bde5f8e0758d87db56f58550f02b',PLAN,'APPROVED','SUFFICIENT')):
        raise PermissionError('Exact B04A R03 acceptance missing')
    base=Path(panels).parent;manifest=json.loads(Path(panels).read_text(encoding='utf8'))
    asset=json.loads(Path(assets).read_text(encoding='utf8'))
    if manifest['plan_version']!=PLAN or asset['plan_version']!=PLAN or digest(assets)!=manifest['assets_sha256']:
        raise ValueError('Historical asset binding mismatch')
    if digest('artifacts/B03/model_registry.json')!=manifest['registry_sha256']:raise ValueError('Registry changed')
    historical=json.loads((base/'artifact_manifest.json').read_text(encoding='utf8'))
    hashes={r['path']:r['sha256'] for r in historical['files']}
    for name in ('state_audit_extract.json','screening_log.parquet','local/cost_checkpoint.json','local/engine_calls.json'):
        if digest(base/name)!=hashes[name]:raise ValueError('B04A evidence changed: '+name)
    panel_rows={}
    for panel,record in manifest['panels'].items():
        path=base/record['path']
        if digest(path)!=record['sha256']:raise ValueError('Panel hash mismatch: '+panel)
        # Only states allowed for S_compare; never outcomes.
        panel_rows[panel]=pq.read_table(path).to_pylist()
        if len(panel_rows[panel])!=manifest['expected_counts'][panel]:raise ValueError('Panel coverage mismatch')
    ids={p:{r['root_id'] for r in rows} for p,rows in panel_rows.items()}
    if any(ids['COST']&ids[p] for p in ('N_dev','S_dev','TUNE')):raise ValueError('COST cache overlaps formal roots')
    extract=json.loads((base/'state_audit_extract.json').read_text(encoding='utf8'));records={r['root_id']:r for r in extract['root_records']}
    calls={c['cache_key']:c for c in json.loads((base/'local/engine_calls.json').read_text(encoding='utf8'))}
    bundles={}
    for panel in ('S_dev','TUNE'):
        for row in panel_rows[panel]:
            rec=records[row['root_id']]
            b=CandidateBundle(row['root_id'],tuple(rec['cp_root']),tuple(rec['safe_moves']),rec['cp_root'],{},extract['engine_hash'],extract['budget_hash'],rec['status'])
            if hash_object(asdict(b))!=row['engine_bundle_hash'] or rec['bundle_hash']!=row['engine_bundle_hash']:
                raise ValueError('Screen bundle hash mismatch')
            if b.status!='ELIGIBLE':raise ValueError('Conditional panel unqualified')
            board=rebuild_root(root_context(row));bundles[b.root_id]=b
            import chess
            requests=[(100000,None,min(5,board.legal_moves.count()))]+[(50000,[m],1) for m in b.moves]
            screen_nodes=0
            for nodes,moves,multipv in requests:
                key=hash_object(['decision',board.root().fen(),[m.uci() for m in board.move_stack],board.turn,nodes,moves,multipv,b.engine_hash])
                call=calls[key]
                if call['requested_nodes']!=nodes:raise ValueError('Screen cold-node provenance mismatch')
                screen_nodes+=call['actual_nodes']
            row['cold_screen_actual_nodes']=screen_nodes
    screens={r['root_id']:r for r in pq.read_table(base/'screening_log.parquet').to_pylist() if r['panel']=='N_dev'}
    for row in panel_rows['N_dev']:
        rec=screens[row['root_id']]
        if rec['engine_bundle_hash']!=row['engine_bundle_hash']:raise ValueError('N state binding mismatch')
        row.update(status=rec['status'],eligibility_reason=rec['eligibility_reason'])
    # D6 hash selection happens with state-only columns before any legal cost/target access.
    path=Path('artifacts/B01/samples.parquet')
    cols=STATE_COLUMNS+['sample_id']
    rows=pq.read_table(path,columns=cols,filters=[('player_id','in',list(DEV)),('split','in',['H1','H2']),('main_eligible','=',True)]).to_pylist()
    risk=freeze_risk_rows(rows)
    if len(risk)!=1600:raise ValueError('D6 must retain all 16 strata')
    lock=dict(sample_ids=[r['sample_id'] for r in risk],sample_state_hash=hash_object(risk),
        source_sha256=digest(path),count=1600,seed=1107,panel_salt='risk',
        rule='D6 game-first fixed hash; inherited root seed1107; hash(seed|risk|game), then hash(seed|risk|game|ply|input_hash); 100/person/window',
        outcomes_read_before_freeze=False,costs_read_before_freeze=False)
    write_json(out/'risk_sample.lock.json',lock)
    pq.write_table(__import__('pyarrow').Table.from_pylist(risk),out/'local/risk_states.parquet')
    write_json(out/'input_verification.json',dict(plan_version=plan,panels_sha256=digest(panels),
        assets_sha256=digest(assets),bundles_verified=len(bundles),natural_roots=len(panel_rows['N_dev']),
        COST_formal_intersection=0,S_compare_outcome_reads=0,confirmation_H_reads=0,
        risk_sample_lock_sha256=digest(out/'risk_sample.lock.json'),inherited=inherited,
        state_extract_sha256=digest(base/'state_audit_extract.json'),
        screening_log_sha256=digest(base/'screening_log.parquet')))
    return panel_rows,bundles,risk,inherited
