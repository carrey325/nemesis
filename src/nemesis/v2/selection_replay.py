"""Bounded B04A trace correction using frozen COST inputs, never panel resampling."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import pyarrow.parquet as pq

from nemesis.data.common import digest, write_json
from .engine import hash_object, rebuild_root
from .evaluate import verify_selection_seal
from .panels import root_context
from .screen import select_from_tables, write_selection_output
from .select import matched_selection
from .types import CandidateBundle


def replay(source, output):
    source=Path(source); output=Path(output)
    if output.exists(): raise FileExistsError('Replay output exists')
    start=time.time(); cpu=time.process_time()
    # Only frozen decision candidates/policies and original matched-search measurements
    # enter the selection phase. Audit/E inputs are read below, after the new seal.
    paths=[source/'local/cost_checkpoint.json',source/'local/warm/probability_tables.json',
           source/'roots/COST.parquet',source/'local/selection_seal.json',
           source/'local/evaluator_request.final.json']
    bindings={str(p):digest(p) for p in paths}
    checkpoint=json.loads(paths[0].read_text())
    policy=json.loads(paths[1].read_text())
    rows=pq.read_table(paths[2]).to_pylist()
    roots=[root_context(r) for r in rows]
    bundles={}
    for rid,b in checkpoint['bundles'].items():
        b=dict(b,moves=tuple(b['moves']),safe_moves=tuple(b['safe_moves']))
        bundles[rid]=CandidateBundle(**b)
    tables={mid:{tuple(k.split('|')):v for k,v in table.items()} for mid,table in policy['tables'].items()}
    traces=[]
    selections=select_from_tables(roots,bundles,tables,policy['model_hashes'],traces=traces)
    calls=[c for c in checkpoint['calls'] if c['namespace']=='sf_compute_matched']
    if len(calls)!=len(roots) or len(roots)!=32: raise ValueError('Incomplete frozen COST32 match provenance')
    for root,m,call in zip(roots,checkpoint['measurements'],calls):
        bundle=bundles[root.root_id]; b=rebuild_root(root)
        key=hash_object(['sf_compute_matched',b.root().fen(),[move.uci() for move in b.move_stack],
                         root.root_color,m['online_nodes'],None,1,bundle.engine_hash])
        if m['root_id']!=root.root_id or call['cache_key']!=key:
            raise ValueError('Frozen matched root request provenance mismatch')
        selections.append(matched_selection(bundle,dict(m,matched_call=call),traces))
    prior=json.loads(paths[3].read_text())['selections']
    choices=lambda records:[(r['root_id'],r['method_id'],r['identity'],r['move_uci']) for r in records]
    if choices([asdict(s) for s in selections])!=choices(prior):
        raise ValueError('Correction changed frozen choices')
    output.mkdir(parents=True)
    write_selection_output(selections,traces,output/'smoke_selections.parquet',output/'selection_seal.json')
    verify_selection_seal(selections,output/'selection_seal.json',traces=traces,require_trace=True)
    sealed_unix=time.time()
    request=json.loads(paths[4].read_text())
    request['selections']=[asdict(s) for s in selections]
    request['config'].update(_selection_seal=str(output/'selection_seal.json'),
                             _selection_traces=traces,_selector_pid=os.getpid())
    write_json(output/'evaluator_request.json',request)
    write_json(output/'selection_replay.json',dict(input_sha256=bindings,selection_count=len(selections),
      unchanged_choices=len(selections),matched_count=len(calls),source_sha256={str(p):digest(p)
        for p in sorted(Path('src/nemesis/v2').glob('*.py'))},
      selection_seal_sha256=digest(output/'selection_seal.json'),
      evaluator_request_sha256=digest(output/'evaluator_request.json'),
      start_unix=start,sealed_unix=sealed_unix,end_unix=time.time(),cpu_seconds=time.process_time()-cpu,
      gpu_seconds=0,scope='Frozen COST32 selection replay; no engines/models/panel resampling',
      audit_access='Original checkpoint physically contains audit fields; only decision fields used before seal; final evaluator request read after seal'))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('source'); parser.add_argument('output')
    args=parser.parse_args(); replay(args.source,args.output)
