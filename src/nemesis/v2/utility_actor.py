"""Short-lived CUDA actors; selectors and evaluators have separate processes."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import numpy as np
from nemesis.data.common import write_json,digest
from .engine import hash_object,expected_value
from .panels import DEV,root_context
from .select import branch_context,select_move,bind_selection
from .screen import TablePolicy,select_from_tables,write_selection_output
from .evaluate import verify_selection_seal
from .types import CandidateBundle,Selection

def predict_variants(model,contexts,keys,factors):
    tables={str(f):{} for f in factors};original=dict(model.record)
    for f in factors:
        model.record=dict(original,temperature=original['temperature']*f)
        for start in range(0,len(contexts),32):
            for key,d in zip(keys[start:start+32],model.predict_many(contexts[start:start+32])):
                p=dict(zip(d.moves_uci,d.probabilities))
                if abs(sum(p.values())-1)>1e-6 or any(not np.isfinite(x) or x<0 for x in p.values()):raise ValueError('Invalid full policy mass')
                tables[str(f)][key]=p
    model.record=original
    return tables

def selector(request,out):
    from nemesis.models.personal_predictor import TwinPredictor
    import torch
    rows=request['roots'];roots=[root_context(r) for r in rows]
    bundles={rid:CandidateBundle(**dict(b,moves=tuple(b['moves']),safe_moves=tuple(b['safe_moves']))) for rid,b in request['bundles'].items()}
    mids=['G0','G']+[f'T_{p}_{s}' for p in DEV for s in (1729,2718,3141)]
    tables={str(f):{} for f in (.9,1.,1.1)};hashes={};costs=[];keys=[];contexts=[]
    for root in roots:
        for move,leaves in bundles[root.root_id].reply_values.items():
            if '__terminal__' not in leaves:keys.append((root.root_id,move));contexts.append(branch_context(root,move))
    support=[]
    for mid in mids:
        cpu=time.process_time();tick=time.time();model=TwinPredictor(request['registry'],mid,'selector')
        predicted=predict_variants(model,contexts,keys,(.9,1.,1.1))
        for f in tables:tables[f][mid]=predicted[f]
        hashes[mid]=model.model_hash
        if mid=='G0':
            index=json.loads(Path('artifacts/B03/support_index.json').read_text(encoding='utf8'))
            # Frozen original mean-square features; same1000 and Cal-only threshold.
            support_contexts=[branch_context(r,bundles[r.root_id].safe_moves[0]) for r in roots]
            from nemesis.types import PositionContext
            support_contexts=[PositionContext(r.fen6,r.history_uci,r.initial_fen,r.root_color,1900,1900,180,0,None,r.input_hash) for r in roots]
            with torch.no_grad():
                for start in range(0,len(roots),32):
                    cs=support_contexts[start:start+32];enc=[model.predictor.encode(c) for c in cs]
                    token=torch.stack([e[0] for e in enc]).cuda();elo=torch.full((len(cs),),1900,device='cuda')
                    h,_=model.policy.extract(token,elo,elo);means=h.mean(1).cpu().numpy()
                    for j,p in enumerate(DEV):
                        rec=index['players'][p] if 'players' in index else index[p]
                        bank=np.load(rec['path'])
                        if digest(rec['path'])!=rec['sha256']:raise ValueError('Support index hash mismatch')
                        similarity=(means@bank.T)/(np.linalg.norm(means,axis=1)[:,None]*np.linalg.norm(bank,axis=1)[None,:]).clip(1e-12)
                        distances=1-similarity.max(axis=1)
                        threshold=rec['calibration_quantiles']['q50']
                        for root,d in zip(roots[start:start+32],distances):support.append(dict(root_id=root.root_id,target_id=p,distance=float(d),threshold=threshold,high_support=bool(d<=threshold)))
        costs.append(dict(model_id=mid,cpu_seconds=time.process_time()-cpu,wall_seconds=time.time()-tick))
        del model;torch.cuda.empty_cache();print('selector',mid,flush=True)
    disagreement=[]
    for root in roots:
        for p in DEV:
            ids=[f'T_{p}_{seed}' for seed in (1729,2718,3141)]
            for move,leaves in bundles[root.root_id].reply_values.items():
                qs=[leaves['__terminal__'] if '__terminal__' in leaves else expected_value(tables['1.0'][mid][root.root_id,move],leaves) for mid in ids]
                disagreement.append(dict(root_id=root.root_id,target_id=p,move=move,seed_q=qs,std=float(np.std(qs)),mean=float(np.mean(qs))))
    traces=[];sels=select_from_tables(roots,bundles,tables['1.0'],hashes,traces=traces)
    from .select import matched_selection
    for root in roots:sels.append(matched_selection(bundles[root.root_id],request['matched'][root.root_id],traces))
    for root,row in zip(roots,rows):
        for label,f in [('09',.9),('11',1.1)]:
            policy=TablePolicy('GENERIC',[tables[str(f)]['G']],[hash_object([hashes['G'],f])])
            choice=select_move(root,bundles[root.root_id],policy,'GENERIC',0)
            sels.append(bind_selection(root.root_id,'TEMP_GENERIC_'+label,'GENERIC',choice.move_uci,bundles[root.root_id],policy.model_hash,0,None,traces))
        for p in DEV:
            ids=[f'T_{p}_{seed}' for seed in (1729,2718,3141)]
            variants=[('TEMP_09',.9,0),('TEMP_11',1.1,0)]
            if row['panel']=='TUNE':variants += [('LAMBDA_05',1.,.5),('LAMBDA_2',1.,2)]
            for method,f,lam in variants:
                policy=TablePolicy(p,[tables[str(f)][mid] for mid in ids],[hash_object([hashes[mid],f]) for mid in ids])
                choice=select_move(root,bundles[root.root_id],policy,'PERSONAL_ROBUST',lam)
                sels.append(bind_selection(root.root_id,method,p,choice.move_uci,bundles[root.root_id],policy.model_hash,lam,None,traces))
    out.mkdir(exist_ok=True)
    write_selection_output(sels,traces,out/'selections.parquet',out/'selection_seal.json')
    write_json(out/'selector_output.json',dict(selections=[asdict(s) for s in sels],traces=traces,model_hashes=hashes,
       role_costs=costs,support=support,disagreement=disagreement,selector_pid=os.getpid(),selection_seal_sha256=digest(out/'selection_seal.json')))

def evaluator(request,out):
    from nemesis.models.personal_predictor import TwinPredictor
    import torch
    sels=[Selection(**s) for s in request['selections']]
    verify_selection_seal(sels,request['seal'],traces=request['traces'],require_trace=True)
    if request['selector_pid']==os.getpid():raise PermissionError('Independent evaluator required')
    roots=[root_context(r) for r in request['roots']];values=request['audits'];moves=[];contexts=[]
    for root in roots:
        for move,leaves in values[root.root_id]['values'].items():
            if '__terminal__' not in leaves:moves.append((root.root_id,move));contexts.append(branch_context(root,move))
    records=[];costs=[]
    for mid in ['E3_group','E2_base','E2_group']+[f'{family}_{p}_{seed}' for family,seed in [('E3',4242),('E2',6161)] for p in DEV]:
        tick=time.time();cpu=time.process_time();model=TwinPredictor(request['registry'],mid,'evaluator')
        predictions=predict_variants(model,contexts,moves,(.9,1.,1.1))
        probability_path=Path(out).parent/'evaluator_probabilities'/(mid+'.json')
        write_json(probability_path,dict(model_id=mid,model_hash=model.model_hash,
            request_seal_sha256=digest(request['seal']),
            tables={f:{'|'.join(key):p for key,p in table.items()} for f,table in predictions.items()}))
        family='E3' if model.record['family']=='maia3' else 'E2';target=model.record['identity'] or mid
        for rid,body in values.items():
            for move,leaves in body['values'].items():
                q={f:(leaves['__terminal__'] if '__terminal__' in leaves else expected_value(table[(rid,move)],leaves)) for f,table in predictions.items()}
                sensitivity=body.get('sensitivity',{}).get('values',{}).get(move)
                sq=None if sensitivity is None else (sensitivity['__terminal__'] if '__terminal__' in sensitivity else expected_value(predictions['1.0'][(rid,move)],sensitivity))
                records.append(dict(root_id=rid,evaluator_family=family,target_id=target,model_id=mid,move_uci=move,q=q['1.0'],q_temp09=q['0.9'],q_temp11=q['1.1'],q_sensitivity=sq,model_hash=model.model_hash))
        costs.append(dict(model_id=mid,cpu_seconds=time.process_time()-cpu,wall_seconds=time.time()-tick))
        del model;torch.cuda.empty_cache();print('evaluator',mid,flush=True)
    write_json(out,dict(values=records,role_costs=costs,selection_seal_sha256=digest(request['seal']),evaluator_pid=os.getpid()))

def risk_predict(request,out):
    from nemesis.models.personal_predictor import TwinPredictor
    from nemesis.models.common import context_from_row
    import torch
    rows=request['rows'];contexts=[context_from_row(r) for r in rows];records=[];costs=[]
    for mid in ['G']+[f'T_{p}_{seed}' for p in DEV for seed in (1729,2718,3141)]:
        selected=list(range(len(rows))) if mid=='G' else [i for i,r in enumerate(rows) if r['player_id']==mid.split('_')[1]]
        tick=time.time();cpu=time.process_time();model=TwinPredictor(request['registry'],mid,'selector')
        for start in range(0,len(selected),32):
            indices=selected[start:start+32]
            for i,d in zip(indices,model.predict_many([contexts[i] for i in indices])):
                values=request['values'][rows[i]['sample_id']]['values'];best=max(values.values())
                probs=dict(zip(d.moves_uci,d.probabilities))
                if set(probs)!=set(values):raise ValueError('Risk complete legal vocabulary mismatch')
                risk=sum(p for m,p in probs.items() if best-values[m]>=.05)
                records.append(dict(sample_id=rows[i]['sample_id'],model_id=mid,risk=risk,model_hash=model.model_hash))
        costs.append(dict(model_id=mid,cpu_seconds=time.process_time()-cpu,wall_seconds=time.time()-tick))
        del model;torch.cuda.empty_cache()
    write_json(out,dict(predictions=records,role_costs=costs))

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase');p.add_argument('request');p.add_argument('output');a=p.parse_args()
    request=json.loads(Path(a.request).read_text(encoding='utf8'))
    if any(r.get('panel')=='S_compare' or (r.get('panel') or '').endswith('confirm') or r.get('source_player_id',r.get('player_id')) not in DEV for r in request.get('roots',request.get('rows',[]))):raise PermissionError('Sealed/unknown inputs denied before models')
    # B04B consumes the unchanged B03R/B04A registry; names alone do not
    # authorize swapped role/checkpoint metadata in a resumed request.
    if digest(request['registry'])!='7938aa2665659d2ea87f832ad33341b17f759a2d5d098a6978676de13cf8f111':
        raise ValueError('Actor registry differs from the accepted B04A asset binding')
    tick=time.time();cpu=time.process_time()
    try:
        {'selector':selector,'evaluator':evaluator,'risk':risk_predict}[a.phase](request,Path(a.output))
    finally:
        output_path=Path(a.output)/'selector_output.json' if a.phase=='selector' else Path(a.output)
        write_json(Path(a.request).with_suffix('.cost.json'),dict(cpu_seconds=time.process_time()-cpu+10,
          resident_gpu_seconds=time.time()-tick+5,start_unix=tick,end_unix=time.time(),
          cpu_allowance=10,gpu_boundary_allowance=5,phase=a.phase,request_sha256=digest(a.request),
          output_sha256=digest(output_path) if output_path.is_file() else None,
          source_sha256={str(p):digest(p) for p in Path('src/nemesis/v2').glob('*.py')}))
    # Explicit safe checkpoint for a diagnosed scheduling correction; never audit
    # under the old in-memory schedule. Completed decisions/seal remain immutable.
    stop=Path(a.request).parent/'checkpoint_stop.request'
    if a.phase=='selector' and stop.exists():
        print('CHECKPOINT_STOP: '+stop.read_text(encoding='utf8'),flush=True)
        raise SystemExit(7)

if __name__=='__main__':main()
