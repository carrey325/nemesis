from datetime import datetime, timezone
from pathlib import Path
import pyarrow.parquet as pq
from .common import append_json

def split_for(utc):
    dt=datetime.fromisoformat(utc.replace('Z','+00:00')).astimezone(timezone.utc)
    day=dt.strftime('%Y-%m-%d')
    if '2025-08-01'<=day<'2026-04-01': return 'fit_a' if dt.isocalendar().week%2==0 else 'fit_b'
    for name,start,end in [('calibration','2026-04-01','2026-05-01'),('H1','2026-05-01','2026-08-01'),('H2','2026-08-01','2026-10-01')]:
        if start<=day<end: return name
    return None

def deduplicate(rows):
    seen=set(); result=[]
    for row in rows:
        if row['game_id'] not in seen: result.append(row); seen.add(row['game_id'])
    return result

def exclude_cross_cohort(game, players):
    return game['white'].casefold() in players and game['black'].casefold() in players

def read_samples(path, *, cohort, split, log, freeze=None):
    # An explicit validated B05 lock is required; a boolean bypass is unsupported.
    sealed=(cohort=='confirm' and split in {'H1','H2'}) or Path(path).name.startswith('sealed_')
    append_json(log,{'cohort':cohort,'split':split,'action':'REJECTED' if sealed else 'READ','path':Path(path).name})
    if sealed: raise PermissionError('Confirmation labels require B05 accepted lock; gate implemented in B05')
    return pq.read_table(path)

def assert_split_integrity(samples_path,manifest_path):
    import json
    rows=pq.read_table(samples_path).to_pylist(); games={}; ids=set()
    for r in rows:
        if r['sample_id'] in ids: raise ValueError('duplicate sample')
        ids.add(r['sample_id'])
        if split_for(r['utc'])!=r['split']: raise ValueError('UTC partition mismatch')
        if r['game_id'] in games and games[r['game_id']]!=r['split']: raise ValueError('cross partition game')
        games[r['game_id']]=r['split']
    m=json.loads(Path(manifest_path).read_text())
    if set(m['dev_players']) & set(m['confirm_players']): raise ValueError('cohort overlap')
    return {'game_partition_overlap':0,'dev_confirm_overlap':0,'fit_a_b_game_overlap':0,'samples_checked':len(rows)}
