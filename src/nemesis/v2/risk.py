"""Outcome-blind D6 sampling; risk labels are computed only after freezing."""
from collections import defaultdict
from .panels import hash_object_seed

def freeze_risk_rows(rows,per_stratum=100):
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:groups[(r['player_id'],r['split'])][r['game_id']].append(r)
    selected=[]
    for h,games in sorted(groups.items()):
        ordered=sorted(games,key=lambda g:hash_object_seed(1107,'risk',g))
        if len(ordered)<per_stratum:raise ValueError('Insufficient D6 game coverage')
        for gid in ordered[:per_stratum]:
            selected.append(min(games[gid],key=lambda r:hash_object_seed(1107,'risk',gid,r['ply'],r['input_hash'])).copy())
    return selected
