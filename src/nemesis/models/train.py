"""B03 fixed supervised method; no engine labels or confirmation holdout reads."""
from collections import defaultdict,Counter
import numpy as np
import json
import hashlib
from pathlib import Path
import torch
from nemesis.data.common import digest,write_json,append_json
from .adapters import ResidualAdapter,save_checkpoint,load_checkpoint,supervised_loss,legal_log_probs,state_hash

def target_index(row,vocabulary,white):
    move=row['target_move_uci']
    if not white: move=move[0]+str(9-int(move[1]))+move[2]+str(9-int(move[3]))+move[4:]
    return vocabulary.index(move)

def check_lineage(a,b):
    if {r['game_id'] for r in a}&{r['game_id'] for r in b}: raise ValueError('Fit-A/B game overlap')
    if any(r['split']!='fit_a' for r in a) or any(r['split']!='fit_b' for r in b):
        raise ValueError('Wrong training split')

def bootstrap_indices(rows,seed):
    games=defaultdict(list)
    for i,r in enumerate(rows): games[r['game_id']].append(i)
    keys=sorted(games); rng=np.random.default_rng(seed)
    return np.array([i for g in rng.choice(keys,len(keys),replace=True) for i in games[g]],dtype=np.int64)

def equal_weights(rows):
    counts=Counter((r['player_id'],r['game_id']) for r in rows)
    games=defaultdict(set)
    for r in rows: games[r['player_id']].add(r['game_id'])
    return np.array([1/(len(games)*len(games[r['player_id']])*counts[r['player_id'],r['game_id']]) for r in rows],dtype=np.float64)

def load_rows(root,cohort,splits,log):
    import pyarrow.parquet as pq
    if any(s in ('H1','H2') for s in splits) and cohort!='dev':
        raise PermissionError('Confirmation H1/H2 remains sealed')
    root=Path(root); manifest=json.loads((root/'artifact_manifest.json').read_text())
    for name in ('samples.parquet','split_manifest.json'):
        record=next(r for r in manifest['files'] if Path(r['path']).name==name)
        if digest(root/name)!=record['sha256']: raise ValueError('B01 hash mismatch '+name)
    split=json.loads((root/'split_manifest.json').read_text())
    players=group_players(split) if cohort=='all_targets' else split[{'dev':'dev_players','confirm':'confirm_players'}[cohort]]
    table=pq.read_table(root/'samples.parquet',filters=[('player_id','in',players),('split','in',list(splits)),('main_eligible','=',True)])
    rows=table.to_pylist()
    append_json(log,dict(action='READ',cohort=cohort,splits=splits,rows=len(rows),path=str(root/'samples.parquet'),sha256=digest(root/'samples.parquet')))
    return rows,split

def group_players(split):
    players=sorted(split['dev_players']+split['confirm_players'])
    if len(players)!=32 or len(set(players))!=32: raise ValueError('Pooled controls require exactly all32 frozen targets')
    return players

def rows_hash(rows):
    h=hashlib.sha256()
    for r in rows:
        h.update(json.dumps([r['sample_id'],r['game_id'],r['player_id'],r['split'],r['input_hash'],r['target_move_uci']],separators=(',',':')).encode())
    return h.hexdigest()

def cache_compatible(receipt,base_sha,family,row_sha):
    if receipt['base_sha']!=base_sha or receipt['family']!=family or receipt['rows_sha256']!=row_sha:
        raise ValueError('Frozen feature cache model/data lineage mismatch')

class FeatureCache:
    def __init__(self,path,rows):
        self.path=Path(path); self.rows=rows
        self.meta=json.loads((self.path/'receipt.json').read_text())
        if self.meta['rows_sha256']!=rows_hash(rows): raise ValueError('Cached feature lineage mismatch')
        self.arrays={k:np.load(self.path/(k+'.npy'),mmap_mode='r') for k in ('h','indices','mask','targets','original')}

    def batch(self,ids,device):
        a=self.arrays
        return tuple(torch.from_numpy(np.array(a[k][ids])).to(device) for k in ('h','indices','mask','targets','original'))

def build_cache(policy,predictor,rows,path,meter=None):
    from .common import context_from_row
    path=Path(path)
    if (path/'receipt.json').exists():
        receipt=json.loads((path/'receipt.json').read_text())
        cache_compatible(receipt,predictor.record['sha256'],policy.family,rows_hash(rows))
        if not receipt.get('scratch_tensors_released'): return FeatureCache(path,rows)
        (path/'receipt.json').rename(path/'released_receipt.json')
    if path.exists() and any(p.name!='released_receipt.json' for p in path.iterdir()): raise FileExistsError('Incomplete feature cache requires inspected recovery')
    path.mkdir(parents=True,exist_ok=True)
    n=len(rows); dim=(64,predictor.cfg.dim_vit) if policy.family=='maia3' else (predictor.cfg.dim_vit,)
    arrays={k:np.lib.format.open_memmap(path/(k+'.npy'),mode='w+',dtype=dtype,shape=shape) for k,dtype,shape in
        [('h','float32',(n,*dim)),('indices','int64',(n,256)),('mask','bool',(n,256)),('targets','int64',(n,)),('original','float32',(n,256))]}
    max_error=0.; before=state_hash(policy.base)
    for start in range(0,n,64):
        part=rows[start:start+64]; contexts=[context_from_row(r) for r in part]
        encoded=[predictor.encode(c) for c in contexts]
        tokens=torch.stack([e[0] for e in encoded]).to(predictor.device)
        se=torch.tensor([c.self_elo if policy.family=='maia3' else e[3] for c,e in zip(contexts,encoded)],device=predictor.device)
        oe=torch.tensor([c.oppo_elo if policy.family=='maia3' else e[4] for c,e in zip(contexts,encoded)],device=predictor.device)
        with torch.no_grad():
            h,logits=policy.extract(tokens,se,oe)
            if start==0:
                mask=torch.stack([e[1] for e in encoded]).to(predictor.device)
                max_error=(legal_log_probs(logits,mask).exp()-legal_log_probs(policy.policy(h,False),mask).exp()).abs().max().item()
                if max_error>1e-5: raise RuntimeError('Frozen feature policy violates upstream M5 parity')
        arrays['h'][start:start+len(part)]=h.cpu().numpy()
        for i,(r,e) in enumerate(zip(part,encoded)):
            ids=e[1].nonzero().flatten().tolist(); count=len(ids)
            if count>256: raise ValueError('Legal vocabulary cache overflow')
            index=target_index(r,predictor.vocab,e[2].turn)
            if index not in ids: raise ValueError('Human target outside legal vocabulary')
            arrays['indices'][start+i]=0; arrays['indices'][start+i,:count]=ids
            arrays['mask'][start+i]=False; arrays['mask'][start+i,:count]=True
            arrays['targets'][start+i]=ids.index(index)
            arrays['original'][start+i]=-np.inf
            arrays['original'][start+i,:count]=logits[i,ids].cpu().numpy()
        if start%4096==0:
            if meter: meter.checkpoint()
            print(f'cache {policy.family} {path.name}: {start+len(part)}/{n}',flush=True)
    for a in arrays.values(): a.flush()
    del arrays
    after=state_hash(policy.base)
    if before!=after: raise RuntimeError('Base changed during feature extraction')
    write_json(path/'receipt.json',dict(rows=n,rows_sha256=rows_hash(rows),family=policy.family,
        base_sha=predictor.record['sha256'],base_state_sha=before,base_state_after_sha=after,
        device=predictor.device,precision='FP32',max_upstream_probability_difference=max_error,
        files={p.name:digest(p) for p in path.glob('*.npy')}))
    return FeatureCache(path,rows)

class InputCache:
    """Full FP32 locked inputs, recomputing frozen backbone to bound disk usage."""
    def __init__(self,path,rows,policy):
        self.path=Path(path); self.rows=rows; self.policy=policy
        self.meta=json.loads((self.path/'receipt.json').read_text())
        if self.meta['rows_sha256']!=rows_hash(rows): raise ValueError('Input cache lineage')
        self.arrays={k:np.load(self.path/(k+'.npy'),mmap_mode='r') for k in ('tokens','se','oe','indices','mask','targets','original')}

    def batch(self,ids,device):
        a=self.arrays
        token,se,oe=(torch.from_numpy(np.array(a[k][ids])).to(device) for k in ('tokens','se','oe'))
        h,_=self.policy.extract(token,se,oe)
        return (h,*tuple(torch.from_numpy(np.array(a[k][ids])).to(device) for k in ('indices','mask','targets','original')))

class CacheView:
    """Read-only person/cohort selection from a hash-bound pooled input cache."""
    def __init__(self,cache,players):
        self.cache=cache; self.path=cache.path
        self.indices=np.array([i for i,r in enumerate(cache.rows) if r['player_id'] in players],dtype=np.int64)
        self.rows=[cache.rows[i] for i in self.indices]
        self.arrays={k:np.array(cache.arrays[k][self.indices]) for k in ('indices','mask','targets','original')}
        self.meta={**cache.meta,'view_rows_sha256':rows_hash(self.rows),'view_players':sorted(players)}

    def batch(self,ids,device): return self.cache.batch(self.indices[ids],device)

def build_input_cache(policy,predictor,rows,path,meter=None):
    from .common import context_from_row
    path=Path(path)
    if (path/'receipt.json').exists():
        receipt=json.loads((path/'receipt.json').read_text())
        cache_compatible(receipt,predictor.record['sha256'],policy.family,rows_hash(rows))
        if not receipt.get('scratch_tensors_released'): return InputCache(path,rows,policy)
        (path/'receipt.json').replace(path/'released_receipt.json')
    if path.exists() and any(p.name!='released_receipt.json' for p in path.iterdir()): raise FileExistsError('Incomplete input cache')
    path.mkdir(parents=True,exist_ok=True); n=len(rows)
    first=predictor.encode(context_from_row(rows[0]))
    shapes=[('tokens','float32',(n,*first[0].shape)),('se','int64',(n,)),('oe','int64',(n,)),
        ('indices','int64',(n,256)),('mask','bool',(n,256)),('targets','int64',(n,)),('original','float32',(n,256))]
    arrays={k:np.lib.format.open_memmap(path/(k+'.npy'),mode='w+',dtype=dtype,shape=shape) for k,dtype,shape in shapes}
    for start in range(0,n,64):
        part=rows[start:start+64]; contexts=[context_from_row(r) for r in part]
        encoded=[predictor.encode(c) for c in contexts]
        arrays['tokens'][start:start+len(part)]=torch.stack([e[0] for e in encoded]).numpy()
        arrays['se'][start:start+len(part)]=[c.self_elo if policy.family=='maia3' else e[3] for c,e in zip(contexts,encoded)]
        arrays['oe'][start:start+len(part)]=[c.oppo_elo if policy.family=='maia3' else e[4] for c,e in zip(contexts,encoded)]
        token,se,oe=(torch.from_numpy(np.array(arrays[k][start:start+len(part)])).to(predictor.device) for k in ('tokens','se','oe'))
        _,logits=policy.extract(token,se,oe)
        for i,(r,e) in enumerate(zip(part,encoded)):
            ids=e[1].nonzero().flatten().tolist(); count=len(ids)
            arrays['indices'][start+i]=0; arrays['indices'][start+i,:count]=ids
            arrays['mask'][start+i]=False; arrays['mask'][start+i,:count]=True
            arrays['targets'][start+i]=ids.index(target_index(r,predictor.vocab,e[2].turn))
            arrays['original'][start+i]=-np.inf; arrays['original'][start+i,:count]=logits[i,ids].cpu().numpy()
        if start%4096==0:
            if meter: meter.checkpoint()
            print(f'FP32 input cache {path.name}: {start+len(part)}/{n}',flush=True)
    for a in arrays.values(): a.flush()
    write_json(path/'receipt.json',dict(rows=n,rows_sha256=rows_hash(rows),base_sha=predictor.record['sha256'],family=policy.family,
        mode='on-demand frozen features from FP32 locked inputs',precision='FP32',device='cuda',
        files={p.name:digest(p) for p in path.glob('*.npy')}))
    return InputCache(path,rows,policy)

def predict_cached(policy,cache,ids=None,temperature=1.,meter=None):
    ids=list(range(len(cache.rows))) if ids is None else list(ids)
    output=np.empty((len(ids),256),dtype=np.float32)
    policy.eval()
    with torch.no_grad():
        for start in range(0,len(ids),64):
            h,indices,mask,_,_=cache.batch(ids[start:start+64],next(policy.personal.parameters()).device)
            output[start:start+len(h)]=legal_log_probs(policy.policy(h).gather(1,indices),mask,temperature).exp().cpu().numpy()
            if meter and start%4096==0: meter.checkpoint()
    return output

def fit_cached(policy,cache,ids,metadata,out,max_epochs=10,max_steps=2000,batch_size=256,
               patience=2,calibration=None,resume=False,fault_after=None,meter=None,initial_state=None):
    """Atomic exact optimizer-boundary resume, deterministic epoch permutations."""
    out=Path(out); out.mkdir(parents=True,exist_ok=True); device=next(policy.personal.parameters()).device
    ids=np.asarray(ids,dtype=np.int64); seed=metadata['seed']
    weights=equal_weights([cache.rows[i] for i in ids])
    # Bootstrap multiplicities must retain their weight: original game weighting
    # repeated once per sampled game occurrence, normalized across occurrences.
    if metadata.get('bootstrap'):
        game_sizes=Counter(r['game_id'] for r in cache.rows)
        weights=np.array([1/game_sizes[cache.rows[i]['game_id']] for i in ids]); weights/=weights.sum()
    torch.manual_seed(seed)
    if policy.family=='maia3': policy.personal=ResidualAdapter(policy.base.cfg.dim_vit).to(device)
    else: policy.personal.load_state_dict(policy.base.fc_1.state_dict())
    if initial_state is not None: policy.personal.load_state_dict(initial_state)
    opt=torch.optim.AdamW(policy.personal.parameters(),lr=.001,weight_decay=.0001)
    start_epoch=0; cursor=0; step=0; bad=0; best=float('inf'); losses=[]; sequence=[]; epochs=[]; gradient_max=0.
    before=state_hash(policy.base)
    snapshot=ids[:min(256,len(ids))]
    def snapshot_loss():
        h,ind,mask,target,original=cache.batch(snapshot,device)
        with torch.no_grad(): return supervised_loss(policy.policy(h).gather(1,ind),original,mask,target).item()
    initial=snapshot_loss()
    if resume:
        value=load_checkpoint(out/'resume.pt',policy.personal,metadata); state=value['training']
        opt.load_state_dict(state['optimizer']); start_epoch=state['epoch']; cursor=state['cursor']; step=state['step']
        bad=state['bad']; best=state['best']; losses=state['losses']; sequence=state['sequence']; epochs=state['epochs']
        gradient_max=state['gradient_max']; initial=state['initial_loss']
    elif (out/'resume.pt').exists(): raise FileExistsError('Run exists; explicit resume required')
    def checkpoint(epoch,cursor):
        save_checkpoint(out/'resume.pt',policy.personal,metadata,dict(optimizer=opt.state_dict(),epoch=epoch,cursor=cursor,
            step=step,bad=bad,best=best,losses=losses,sequence=sequence,epochs=epochs,gradient_max=gradient_max,initial_loss=initial))
    for epoch in range(start_epoch,max_epochs):
        order=np.random.default_rng(seed+epoch).permutation(len(ids))
        stop=len(order) if len(order)<batch_size else len(order)//batch_size*batch_size
        for offset in range(cursor,stop,batch_size):
            pos=order[offset:min(offset+batch_size,stop)]; batch_ids=ids[pos]
            policy.train(); opt.zero_grad(set_to_none=True)
            # microbatch 64, accumulating to effective 256; no AMP.
            total=0.
            for sub in range(0,len(batch_ids),64):
                h,ind,mask,target,original=cache.batch(batch_ids[sub:sub+64],device)
                w=torch.tensor(weights[pos[sub:sub+64]]*len(ids)/len(batch_ids),device=device,dtype=torch.float32)
                loss=supervised_loss(policy.policy(h).gather(1,ind),original,mask,target,w)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                loss.backward(); total+=loss.item()
            for p in policy.personal.parameters():
                if p.grad is None or not torch.isfinite(p.grad).all(): raise RuntimeError('Missing or nonfinite personal gradient')
                gradient_max=max(gradient_max,p.grad.abs().max().item())
            if any(p.grad is not None for p in policy.base.parameters()): raise RuntimeError('Backbone gradient')
            opt.step(); step+=1; losses.append(total)
            sequence.append(hashlib.sha256(batch_ids.tobytes()).hexdigest())
            if step%100==0:
                checkpoint(epoch,offset+len(pos))
                if meter: meter.checkpoint()
            if fault_after is not None and step==fault_after:
                checkpoint(epoch,offset+len(pos)); raise InterruptedError('Injected optimizer-boundary failure')
            if step>=max_steps: break
        cursor=0
        if calibration is None: cal_loss=snapshot_loss()
        else:
            from .calibrate import weighted_nll
            cal_cache,cal_ids=calibration; preds=predict_cached(policy,cal_cache,cal_ids,meter=meter)
            cal_loss=weighted_nll(preds,cal_cache.arrays['targets'][cal_ids],equal_weights([cal_cache.rows[i] for i in cal_ids]))
        improved=cal_loss<best
        epochs.append(dict(epoch=epoch+1,step=step,calibration_nll=cal_loss,improved=improved,
                           dropped_tail_decisions=len(order)-stop))
        if improved:
            best=cal_loss; bad=0; save_checkpoint(out/'selected.pt',policy.personal,metadata,dict(selected_epoch=epoch+1,calibration_nll=best))
        else: bad+=1
        checkpoint(epoch+1,0)
        print(f"{metadata['model_id']} epoch {epoch+1}: step {step}, Calibration NLL {cal_loss:.6f}",flush=True)
        if meter: meter.checkpoint()
        if bad>=patience or step>=max_steps: break
    load_checkpoint(out/'selected.pt',policy.personal,metadata)
    after=state_hash(policy.base)
    if before!=after: raise RuntimeError('Frozen backbone changed during training')
    final=snapshot_loss()
    result=dict(model_id=metadata['model_id'],steps=step,epochs=epochs,losses=losses,initial_loss=initial,final_loss=final,
        gradient_max=gradient_max,gradients_finite=True,base_state_before=before,base_state_after=after,
        sample_sequence_sha=hashlib.sha256(''.join(sequence).encode()).hexdigest(),
        checkpoint_sha=digest(out/'selected.pt'),selected_checkpoint=str(out/'selected.pt'),metadata=metadata,
        effective_batch=batch_size,microbatch=min(64,batch_size),stopping='patience' if bad>=patience else 'fixed_cap',
        trainable_parameters={n:{'shape':list(p.shape),'count':p.numel()} for n,p in policy.personal.named_parameters()})
    write_json(out/'training.json',result)
    return result

def fit_adapter(base_model,samples_path,config,out_dir):
    """I3 API: the same fixed Fit/Calibration method used by the CLI."""
    mid=config['metadata']['model_id']; split=config['split']; player=config.get('player_id')
    if mid=='G' or mid.startswith('T_'): required='fit_a'; family='maia3'
    elif mid.startswith('E3_'): required='fit_b'; family='maia3'
    elif mid.startswith('E2_'): required='fit_b'; family='maia2'
    else: raise ValueError('Undeclared training model role')
    if split!=required: raise ValueError('Training role requires '+required+'; no Calibration/H labels for fitting')
    group=mid in ('G','E3_group','E2_group')
    if group and (config['cohort']!='all_targets' or player is not None): raise ValueError('Pooled controls require all32 targets')
    if not group and (config['cohort'] not in ('dev','confirm') or player is None): raise ValueError('Personal role/cohort mismatch')
    if Path(samples_path).resolve()!=Path('artifacts/B01/samples.parquet').resolve(): raise ValueError('Approved B01 sample artifact required')
    predictor=config['predictor']
    if base_model.family!=family or base_model.base is not predictor.model or predictor.device!='cuda': raise ValueError('Locked CUDA predictor/base role mismatch')
    seed=config['metadata']['seed']
    expected_seed=1729 if mid=='G' else 4242 if mid.startswith('E3_') else 6161 if mid.startswith('E2_') else seed
    if seed!=expected_seed or (mid.startswith('T_') and seed not in (1729,2718,3141)): raise ValueError('Fixed seed required')
    from .budget import B03Meter
    own_meter=config.get('meter') is None
    meter=config.get('meter') or B03Meter(Path(config.get('stage_out','artifacts/B03')))
    try:
        root=Path(samples_path).parent; out=Path(out_dir)
        rows,_=load_rows(root,config['cohort'],(split,),out/'access.jsonl')
        cal_rows,_=load_rows(root,config['cohort'],('calibration',),out/'access.jsonl')
        if player is not None:
            rows=[r for r in rows if r['player_id']==player]; cal_rows=[r for r in cal_rows if r['player_id']==player]
        ids=bootstrap_indices(rows,seed) if mid.startswith('T_') else np.arange(len(rows))
        code_hashes={str(p):digest(p) for p in Path('src/nemesis').rglob('*.py')}
        actual=dict(base_sha=predictor.record['sha256'],split_sha=rows_hash([rows[i] for i in ids]),
            config_sha=digest(config.get('config_path','configs/experiment.proposed.yaml')),
            code_sha=hashlib.sha256(json.dumps(code_hashes,sort_keys=True).encode()).hexdigest(),seed=seed,model_id=mid,
            family=family,split=split,cohort=config['cohort'],player_id=player,bootstrap=mid.startswith('T_'),
            source_sha=digest(samples_path),calibration_sha=rows_hash(cal_rows))
        for key,value in config['metadata'].items():
            if key in actual and actual[key]!=value: raise ValueError('API metadata/input mismatch: '+key)
        fit=build_input_cache(base_model,predictor,rows,out/'fit_inputs',meter)
        cal=build_input_cache(base_model,predictor,cal_rows,out/'calibration_inputs',meter)
        fit_cached(base_model,fit,ids,actual,out,calibration=(cal,list(range(len(cal_rows)))),
                   resume=config.get('resume',False),meter=meter)
        return out/'selected.pt'
    finally:
        if own_meter: meter.checkpoint(enforce=False)
