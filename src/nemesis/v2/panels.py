"""Outcome-blind, game-first sampling and bounded B04A orchestration."""
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import chess
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from nemesis.data.common import digest,write_json
from .engine import hash_object,rebuild_root
from .types import RootContext,StageResult

DEV=('P006','P007','P013','P014','P025','P029','P030','P032')
STATE_COLUMNS=['game_id','player_id','split','mover_color','ply','clock_before','initial_fen','fen6','history_uci','input_hash']
ROOT_SCHEMA=pa.schema([(k,t) for k,t in [('root_id',pa.string()),('panel',pa.string()),('source_game_id',pa.string()),
 ('source_player_id',pa.string()),('source_window',pa.string()),('root_color',pa.string()),('ply',pa.int32()),
 ('initial_fen',pa.string()),('fen6',pa.string()),('history_uci',pa.list_(pa.string())),('clock_before',pa.float64()),
 ('input_hash',pa.string()),('sampling_stratum',pa.string()),('sampling_rank',pa.int32()),('eligibility_reason',pa.string()),
 ('engine_bundle_hash',pa.string()),('duplicate_status',pa.string()),('status',pa.string())]])

def read_development_states(path,players=DEV):
    if not players or not set(players)<=set(DEV): raise PermissionError('B04A cannot read confirmation states')
    if Path(path).name.startswith('sealed'): raise PermissionError('B04A refuses sealed confirmation files')
    return pq.read_table(path,columns=STATE_COLUMNS,filters=[('player_id','in',list(players)),
      ('split','in',['calibration','H1','H2']),('ply','>=',20),('ply','<=',80),('clock_before','>=',30)]).to_pylist()

def root_context(row):
    return RootContext(row['root_id'],row['source_game_id'],tuple(row['sampling_stratum'].split('|')),
      row['initial_fen'],row['fen6'],tuple(row['history_uci']),row['root_color']=='white',row['input_hash'])

def require_outcome_access(panel,locks):
    if panel=='S_compare':
        raise PermissionError('S_compare evaluator outcomes require B05A accepted lock; unavailable to B04A')
    if panel.endswith('confirm'): raise PermissionError('Confirmation H is sealed')

def sample_panel(rows,panel,per_stratum,used,screen,seen_inputs=None,transform=None):
    seed={'COST':3101,'TUNE':3102}.get(panel,1107); groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        window=r['split']; is_cal=panel in ('COST','TUNE')
        if (window=='calibration')!=is_cal: continue
        if r['game_id'] in used or not 20<=r['ply']<=80 or r['clock_before'] is None or r['clock_before']<30: continue
        h=(r['player_id'],window,r['mover_color']); groups[h][r['game_id']].append(r)
    selected=[]; logs=[]; seen_inputs=seen_inputs if seen_inputs is not None else set(); scanned=0
    for h,games in sorted(groups.items()):
        count=0
        for rank,gid in enumerate(sorted(games,key=lambda g:hash_object_seed(seed,panel,g)),1):
            if count>=per_stratum or scanned>=50000: break
            scanned+=1
            r=min(games[gid],key=lambda r:hash_object_seed(seed,panel,gid,r['ply'],r['input_hash']))
            # Ignore target, Elo, effects, results. Bind full rule state and real prefix.
            ih=hash_object([r['initial_fen'],r['history_uci'],r['fen6'],1900,1900])
            row=dict(root_id=hash_object([gid,r['initial_fen'],r['history_uci'],r['fen6']]),panel=panel,
              source_game_id=gid,source_player_id=h[0],source_window=h[1],root_color=h[2],ply=r['ply'],
              initial_fen=r['initial_fen'],fen6=r['fen6'],history_uci=r['history_uci'],clock_before=r['clock_before'],
              input_hash=ih,sampling_stratum='|'.join(h),sampling_rank=rank,eligibility_reason='UNSCREENED',
              engine_bundle_hash=None,duplicate_status='UNIQUE',status='UNSCREENED')
            if transform:
                transform(row)
                ih=row['input_hash']
            if screen:
                if ih in seen_inputs and panel not in ('N_dev','COST'): row.update(status='DUPLICATE',duplicate_status='DUPLICATE')
                else: row['status']=screen(row)
                row['eligibility_reason']=row['status']
            logs.append(dict(row))
            if panel in ('N_dev','COST') or row['status']=='ELIGIBLE':
                selected.append(row); used.add(gid); seen_inputs.add(ih); count+=1
    return selected,logs

def hash_object_seed(*parts):
    import hashlib
    return hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()

def forecast_nodes(candidates,replies,legal,*,extra_candidates=0,extra_replies=0):
    online=100000+candidates*50000+replies*20000
    audit=(candidates+extra_candidates+replies+extra_replies)*200000
    sensitivity=(candidates+extra_candidates+replies+extra_replies)*400000
    full=2*online+audit+.1*sensitivity
    return dict(online_per_root=online,audit_per_root=audit,sensitivity_per_root=sensitivity,
      sf_compute_matched_per_root=online,screen_per_root=100000+candidates*50000,
      risk_dev=1600*legal*200000,risk_confirm=4800*legal*200000,
      N_dev_screen=512*(100000+candidates*50000),S_dev=full*256,
      S_compare_two_methods=full*128,N_confirm_screen=960*(100000+candidates*50000),
      S_confirm_480=full*480,S_confirm_960=full*960,
      COST32=full*32,TUNE64=full*64)

def write_roots(path,rows):
    pq.write_table(pa.Table.from_pylist(rows,schema=ROOT_SCHEMA),path,compression='zstd')

def prepare_panels(config,assets,out):
    from .screen import run_screen
    return run_screen(Path(config),Path(assets),Path(out))
