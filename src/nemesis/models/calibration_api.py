"""Public Calibration-only API using the temperature already chosen on dev."""
from pathlib import Path
import json
import pyarrow.parquet as pq
import numpy as np
from .calibrate import require_calibration,choose_alpha
from .train import equal_weights,load_rows

def calibrate(checkpoint,calibration_path,config):
    # Inspect split names before any labels. Arbitrary H-labelled shards cannot
    # become calibration data by passing a different config value.
    if Path(calibration_path).resolve()!=Path('artifacts/B01/samples.parquet').resolve():
        table=pq.read_table(calibration_path,columns=['split'])
        require_calibration(table.to_pylist())
        raise ValueError('Calibration must be bound to approved B01 sample artifact')
    from .personal_predictor import TwinPredictor
    from .common import context_from_row
    rows,_=load_rows(Path(calibration_path).parent,config['cohort'],('calibration',),Path(config['out'])/'calibration_access.jsonl')
    rows=[r for r in rows if r['player_id']==config['player_id']]; require_calibration(rows)
    predictor=TwinPredictor(config['registry'],config['model_id'],'evaluator')
    if Path(predictor.record['checkpoint']).resolve()!=Path(checkpoint).resolve(): raise ValueError('Wrong calibration checkpoint')
    group=TwinPredictor(config['registry'],predictor.record['group'],'evaluator')
    # Evaluate the raw personal distribution at fixed T before selecting alpha.
    predictor.record={**predictor.record,'alpha':1.}
    personal=[]; generic=[]; targets=[]
    for start in range(0,len(rows),64):
        part=rows[start:start+64]; ctx=[context_from_row(r) for r in part]
        for row,p,g in zip(part,predictor.predict_many(ctx),group.predict_many(ctx)):
            personal.append(p.probabilities); generic.append(g.probabilities)
            targets.append(p.moves_uci.index(row['target_move_uci']))
    # Legal counts vary by position: pad probability vectors with exact zeros.
    width=max(map(len,personal)); p=np.zeros((len(rows),width)); g=np.zeros_like(p)
    for i,(a,b) in enumerate(zip(personal,generic)): p[i,:len(a)]=a; g[i,:len(b)]=b
    return {**choose_alpha(p,g,np.array(targets),equal_weights(rows),[0,.25,.5,.75,1]),
            'temperature':predictor.record['temperature'],'selection_split':'calibration','checkpoint_sha256':predictor.record['checkpoint_sha256']}
