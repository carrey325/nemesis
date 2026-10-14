"""B03 orchestration of fixed method, independent controls and full dev matrix."""
from collections import defaultdict
import json,hashlib,gc,time
from pathlib import Path
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from nemesis.data.common import write_json,digest,append_json
from nemesis.preflight import ResourceLimit
from .train import (load_rows,rows_hash,check_lineage,bootstrap_indices,equal_weights,build_cache,
                    fit_cached,predict_cached,FeatureCache,build_input_cache,CacheView)
from .adapters import FrozenPolicy,load_checkpoint,legal_log_probs
from .calibrate import probabilities,weighted_nll,choose_alpha,development_gate,require_calibration,ensemble_distribution,qualified_development_gate
from .budget import B03Meter

BASE='8bc03a8cbd9f3c858fdb5d49fe5597da66bb6e9a'
PLAN='sha256:99f22f7176b26acfe1613087e37bea5a64cd8d9c69140d547425a0077681dd4f'

def check_config(c):
    locked=yaml.safe_load(Path('nemesis-stage1-plan/configs/experiment.proposed.yaml').read_text(encoding='utf8'))
    for section in ('models','data','resources'):
        if c[section]!=locked[section]: raise ValueError('Approved method differs: '+section)

def verify_registry(registry):
    for r in registry['models']:
        if r['status']=='TRAINED' and digest(r['checkpoint'])!=r['checkpoint_sha256']:
            raise ValueError('Registry checkpoint hash mismatch: '+r.get('model_id','unknown'))

def summarize_predictions(probs,targets,rows,model_id,identity):
    p=probs[np.arange(len(targets)),targets]
    order=np.argsort(-probs,axis=1,kind='stable')[:,:3]
    metrics=dict(nll=-np.log(np.maximum(p,1e-30)),top1=(order[:,0]==targets).astype(float),
       top3=(order==targets[:,None]).any(1).astype(float),brier=(probs.astype(np.float64)**2).sum(1)-2*p+1)
    groups=defaultdict(list)
    for i,r in enumerate(rows): groups[r['player_id'],r['split']].append(i)
    records=[]
    for (player,window),ids in sorted(groups.items()):
        selected=[rows[i] for i in ids]; w=equal_weights(selected)
        record=dict(model_id=model_id,identity_id=identity,player_id=player,split=window,
          correct_identity=identity==player if identity else None,decisions=len(ids),games=len({r['game_id'] for r in selected}),
          **{k:float(np.dot(v[ids],w)) for k,v in metrics.items()})
        records.append(record)
    return records

def cache_logits(policy,cache,path,meter):
    path=Path(path)
    if path.exists(): return np.load(path,mmap_mode='r')
    n=len(cache.rows); a=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(n,256))
    with torch.no_grad():
        for start in range(0,n,64):
            h,ind,mask,_,_=cache.batch(list(range(start,min(start+64,n))), 'cuda')
            a[start:start+len(h)]=policy.policy(h).gather(1,ind).masked_fill(~mask,-torch.inf).cpu().numpy()
            if start%4096==0: meter.checkpoint()
    a.flush(); return a

def remove_features(cache,out):
    """Release only explicit reproducible scratch tensors inside B03/cache."""
    allowed=(Path(out)/'cache').resolve(); path=cache.path.resolve()
    if not path.is_relative_to(allowed): raise ValueError('Scratch path outside B03 cache')
    cache.arrays.clear(); gc.collect()
    receipt=json.loads((path/'receipt.json').read_text())
    receipt['scratch_tensors_released']=True
    for name in receipt['files']:
        target=path/name
        if target.exists(): target.unlink()
    write_json(path/'receipt.json',receipt)

def run_training(config_path,out,resume=False):
    from nemesis.v2.guards import reject_legacy_training
    reject_legacy_training()
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/'run_receipt.json').exists() and not resume: raise FileExistsError('B03 run exists; explicit inspected --resume required')
    c=yaml.safe_load(Path(config_path).read_text(encoding='utf8')); check_config(c)
    meter=B03Meter(out)
    config_sha=digest(config_path)
    source_hashes={str(p):digest(p) for p in Path('src/nemesis').rglob('*.py')}
    code_sha=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
    compatible_versions={code_sha}
    if resume:
        old=json.loads((out/'run_receipt.json').read_text())
        if old['config_sha']!=config_sha: raise ValueError('Run/config mismatch')
        if old['code_sha']!=code_sha:
            from .source_compatibility import prove_compatible_training
            compatible_versions=prove_compatible_training(out,old['code_sha'],code_sha)
            write_json(out/'run_receipt.json',{**old,'source_hashes':source_hashes,'code_sha':code_sha,
                'resumed_from_code_sha':old['code_sha'],'source_compatibility':'source_compatibility.json'})
    else:
        write_json(out/'run_receipt.json',dict(stage='B03',attempt_id='B03-N01',status='RUNNING',base_commit=BASE,
            plan_version=PLAN,config_sha=config_sha,code_sha=code_sha,source_hashes=source_hashes,device='cuda',precision='FP32',
            temperature_rule='one unified temperature from dev Calibration across bases/groups/correct personal models; tie lower T',
            alpha_rule='per model and person Calibration; tie lower alpha; retain alpha zero',
            personal_initialization='T/E3 independent zero-up adapter; E2 official base fc_1; no personal cross-initialization',
            group_training='all32 targets corresponding Fit split; equal games then players; seed 1729 for G, 4242/6161 for E group controls; all32 Calibration early stop',
            epoch_rule='shuffle full decision index; drop tail below effective batch256; max10epochs/2000steps; patience2; select minimum Calibration NLL',
            bootstrap_rule='sample same number of entire games with replacement per seed; variable decision count retained without cutting games',
            confirmation_H_opened=False))
    try:
        if not torch.cuda.is_available(): raise ResourceLimit('Canonical CUDA FP32 backend unavailable')
        registry={'stage':'B03','base_commit':BASE,'plan_version':PLAN,'config_sha':config_sha,'code_sha':code_sha,'models':[],
                  'confirmation_H_opened':False}
        runs=[]
        rows_a,split=load_rows(Path('artifacts/B01'),'dev',('fit_a',),out/'access_log.jsonl')
        rows_b,_=load_rows(Path('artifacts/B01'),'dev',('fit_b',),out/'access_log.jsonl')
        rows_cal,_=load_rows(Path('artifacts/B01'),'dev',('calibration',),out/'access_log.jsonl')
        rows_h,_=load_rows(Path('artifacts/B01'),'dev',('H1','H2'),out/'access_log.jsonl')
        check_lineage(rows_a,rows_b); require_calibration(rows_cal)
        pooled_games=pq.read_table('artifacts/B01/samples.parquet',columns=['game_id','player_id','split','main_eligible'],
            filters=[('split','in',['fit_a','fit_b']),('main_eligible','=',True)]).to_pylist()
        pooled_a={r['game_id'] for r in pooled_games if r['split']=='fit_a'}
        pooled_b={r['game_id'] for r in pooled_games if r['split']=='fit_b'}
        if pooled_a&pooled_b: raise ValueError('Pooled all32 Fit-A/B overlap')
        write_json(out/'pooled_lineage_audit.json',dict(status='PASS',players=sorted({r['player_id'] for r in pooled_games}),
            fit_a_games=len(pooled_a),fit_b_games=len(pooled_b),intersection=0,
            source_sha256=digest('artifacts/B01/samples.parquet'),confirmation_H_opened=False))
        del pooled_games,pooled_a,pooled_b
        write_json(out/'lineage_audit.json',dict(status='PASS',Fit_A_games=len({r['game_id'] for r in rows_a}),
            Fit_B_games=len({r['game_id'] for r in rows_b}),intersection=0,
            row_hashes={k:rows_hash(v) for k,v in [('fit_a',rows_a),('fit_b',rows_b),('calibration',rows_cal),('dev_H',rows_h)]},
            players=split['dev_players'],confirmation_labels_opened=False))
        for family in ('maia3','maia2'):
            if family=='maia3':
                from .maia3_bridge import Maia3Predictor
                predictor=Maia3Predictor(device='cuda')
            else:
                from .maia2_bridge import Maia2Predictor
                predictor=Maia2Predictor(device='cuda')
            policy=FrozenPolicy(predictor,family); torch.cuda.reset_peak_memory_stats()
            cal=build_cache(policy,predictor,rows_cal,out/'cache'/f'{family}_calibration',meter)
            future=build_cache(policy,predictor,rows_h,out/'cache'/f'{family}_dev_H',meter)
            pooled_cal_rows,_=load_rows(Path('artifacts/B01'),'all_targets',('calibration',),out/'access_log.jsonl')
            pooled_cal=build_input_cache(policy,predictor,pooled_cal_rows,out/'cache'/f'{family}_pooled_calibration',meter)
            (out/'logits').mkdir(exist_ok=True)
            for label,features in [('calibration',cal),('H',future)]:
                for name in ('targets','indices','mask'):
                    np.save(out/'logits'/f'{family}_{label}_{name}.npy',features.arrays[name])
            base_id='G0' if family=='maia3' else 'E2_base'
            for label,cache in [('calibration',cal),('H',future)]:
                np.save(out/'logits'/f'{base_id}_{label}.npy',cache.arrays['original'])
            registry['models'].append(dict(model_id=base_id,status='BASE',family=family,cohort='dev',identity=None,
                base_sha=predictor.record['sha256'],group=None,alpha=1.))
            tasks=[('fit_a','G',1729,'T') ,('fit_b','E3_group',4242,'E3')] if family=='maia3' else [('fit_b','E2_group',6161,'E2')]
            for partition,group_id,seed,role in tasks:
                fit_rows=rows_a if partition=='fit_a' else rows_b
                pooled_rows,_=load_rows(Path('artifacts/B01'),'all_targets',(partition,),out/'access_log.jsonl')
                pooled_cache=build_input_cache(policy,predictor,pooled_rows,out/'cache'/f'{family}_pooled_{partition}',meter)
                cache=CacheView(pooled_cache,split['dev_players'])
                if rows_hash(cache.rows)!=rows_hash(fit_rows): raise ValueError('Pooled/dev view ordering mismatch')
                jobs=[(group_id,None,seed,list(range(len(pooled_rows))),False)]
                for player in split['dev_players']:
                    player_ids=[i for i,r in enumerate(fit_rows) if r['player_id']==player]
                    for s in c['models']['train_seeds'] if role=='T' else [seed]:
                        ids=[player_ids[i] for i in bootstrap_indices([fit_rows[j] for j in player_ids],s)] if role=='T' else player_ids
                        jobs.append((f'{role}_{player}_{s}',player,s,ids,role=='T'))
                for model_id,player,s,ids,bootstrap in jobs:
                    active_rows=pooled_rows if player is None else fit_rows
                    active_cache=pooled_cache if player is None else cache
                    meta=dict(base_sha=predictor.record['sha256'],split_sha=rows_hash([active_rows[i] for i in ids]),
                        config_sha=config_sha,code_sha=code_sha,model_id=model_id,seed=s,family=family,split=partition,
                        cohort='all_targets' if player is None else 'dev',player_id=player,bootstrap=bootstrap,source_sha=digest('artifacts/B01/samples.parquet'),
                        calibration_sha=rows_hash(pooled_cal_rows if player is None else [r for r in rows_cal if r['player_id']==player]))
                    folder=out/'checkpoints'/model_id
                    if resume and (folder/'resume.pt').exists():
                        from .source_compatibility import preserve_metadata
                        previous=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)['metadata']
                        meta=preserve_metadata(previous,meta,compatible_versions)
                    cal_ids=[i for i,r in enumerate(rows_cal) if player is None or r['player_id']==player]
                    if (folder/'training.json').exists() and resume:
                        result=json.loads((folder/'training.json').read_text()); load_checkpoint(folder/'selected.pt',policy.personal,meta)
                    else:
                        result=fit_cached(policy,active_cache,ids,meta,folder,
                            calibration=(pooled_cal,list(range(len(pooled_cal_rows)))) if player is None else (cal,cal_ids),
                            resume=resume and (folder/'resume.pt').exists(),meter=meter)
                    runs.append(result)
                    for label,features in [('calibration',cal),('H',future)]:
                        cache_logits(policy,features,out/'logits'/f'{model_id}_{label}.npy',meter)
                    registry['models'].append(dict(model_id=model_id,status='TRAINED',family=family,cohort='dev',identity=player,
                        role=role if player else 'group',group=group_id if player else None,
                        checkpoint=str(folder/'selected.pt'),checkpoint_sha256=digest(folder/'selected.pt'),
                        metadata=meta,selected_epoch=min(result['epochs'],key=lambda r:r['calibration_nll'])['epoch'],
                        training_code_sha_initial=meta['code_sha'],current_runner_code_sha=code_sha,
                        train_decisions=len(ids),unique_games=len({active_rows[i]['game_id'] for i in ids}),
                        training_players=sorted({active_rows[i]['player_id'] for i in ids}),
                        sampled_games=len([active_rows[i]['game_id'] for j,i in enumerate(ids) if j==0 or active_rows[ids[j-1]]['game_id']!=active_rows[i]['game_id']]),
                        initialization='official_base_head' if family=='maia2' else 'independent_zero_up',
                        gradient_max=result['gradient_max'],base_unchanged=result['base_state_before']==result['base_state_after']))
                    write_json(out/'model_registry.json',registry)
                    write_json(out/'progress.json',dict(status='RUNNING',last_model=model_id,completed_models=len(runs)))
                    print(f'{model_id} complete ({len(runs)} trained)',flush=True)
                if role=='T': save_support(cache,cal,split['dev_players'],out,meter)
                cache.arrays.clear(); del cache
                remove_features(pooled_cache,out)
                del pooled_cache,pooled_rows; gc.collect()
            remove_features(cal,out); remove_features(future,out); remove_features(pooled_cal,out)
            del predictor,policy,cal,future,pooled_cal,pooled_cal_rows; gc.collect(); torch.cuda.empty_cache()
        parameter_records={r['model_id']:r['trainable_parameters'] for r in runs}
        write_json(out/'trainable_parameters.json',parameter_records)
        curves=[dict(model_id=r['model_id'],step=i+1,loss=v,seed=r['metadata']['seed'],split=r['metadata']['split'])
                for r in runs for i,v in enumerate(r['losses'])]
        pq.write_table(pa.Table.from_pylist(curves),out/'training_runs.parquet',compression='zstd')
        write_json(out/'training_summary.json',[{k:v for k,v in r.items() if k!='losses'} for r in runs])
        calibrate_registry(registry,rows_cal,out,c)
        matrix,gate=evaluate_registry(registry,rows_h,out/'validation',out,c)
        write_json(out/'dev_gate.json',gate)
        for player in split['confirm_players']:
            for role,seeds in [('T',c['models']['train_seeds']),('E3',[4242]),('E2',[6161])]:
                for seed in seeds:
                    registry['models'].append(dict(model_id=f'{role}_{player}_{seed}',cohort='confirm',identity=player,
                        status='NOT_TRAINED',reason='development gate failed' if not gate['passed'] else 'pending confirmation Fit training',
                        family='maia2' if role=='E2' else 'maia3'))
        if gate['passed']:
            # This conditional branch is implemented below; never open H labels.
            runs.extend(train_confirmation(registry,c,out,meter,config_sha,code_sha))
            write_json(out/'trainable_parameters.json',{r['model_id']:r['trainable_parameters'] for r in runs})
            curves=[dict(model_id=r['model_id'],step=i+1,loss=v,seed=r['metadata']['seed'],split=r['metadata']['split'])
                    for r in runs for i,v in enumerate(r['losses'])]
            pq.write_table(pa.Table.from_pylist(curves),out/'training_runs.parquet',compression='zstd')
            write_json(out/'training_summary.json',[{k:v for k,v in r.items() if k!='losses'} for r in runs])
        verify_registry(registry); write_json(out/'model_registry.json',registry)
        write_json(out/'run_receipt.json',{**json.loads((out/'run_receipt.json').read_text()),'status':'DONE',
            'research_status':gate['research_status'],'trained_models':len(runs),
            'peak_vram_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_vram_reserved_bytes':torch.cuda.max_memory_reserved()})
        return 0
    except ResourceLimit as exc:
        write_json(out/'resource_stop.json',dict(status='RESOURCE_LIMIT',reason=str(exc),statistics_complete=False))
        return 4
    finally: meter.checkpoint(enforce=False)

def save_support(fit,cal,players,out,meter):
    """Seed2026 reservoir of 1000 frozen mean-square features; Cal-only thresholds."""
    records={}
    for player in players:
        ids=[i for i,r in enumerate(fit.rows) if r['player_id']==player]
        rng=np.random.default_rng(2026); reservoir=ids[:1000]
        for n,index in enumerate(ids[1000:],1000):
            j=int(rng.integers(0,n+1))
            if j<1000: reservoir[j]=index
        def means(cache,indices):
            if 'h' in cache.arrays: return np.array(cache.arrays['h'][indices]).mean(1)
            return np.concatenate([cache.batch(indices[start:start+64],'cuda')[0].cpu().numpy().mean(1)
                                   for start in range(0,len(indices),64)])
        h=means(fit,reservoir)
        h/=np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-12)
        path=Path(out)/'support'/f'{player}.npy'; path.parent.mkdir(exist_ok=True); np.save(path,h)
        cal_ids=[i for i,r in enumerate(cal.rows) if r['player_id']==player]; distances=[]
        for start in range(0,len(cal_ids),256):
            q=means(cal,cal_ids[start:start+256])
            q/=np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12)
            distances.extend((1-(q@h.T).max(1)).tolist())
        records[player]=dict(path=str(path),sha256=digest(path),samples=len(h),seed=2026,
            source_sample_hashes=[hashlib.sha256(fit.rows[i]['sample_id'].encode()).hexdigest() for i in reservoir],
            calibration_quantiles={'q50':float(np.quantile(distances,.5)),'q90':float(np.quantile(distances,.9)),
                                   'q95':float(np.quantile(distances,.95))},metric='1 - nearest cosine similarity')
        meter.checkpoint()
    file=Path(out)/'support_index.json'
    previous=json.loads(file.read_text()) if file.exists() else {}
    write_json(file,{**previous,**records})

def calibrate_registry(registry,rows,out,c):
    require_calibration(rows); weights=equal_weights(rows)
    targets={family:np.load(Path(out)/'cache'/f'{family}_calibration'/'targets.npy') if (Path(out)/'cache'/f'{family}_calibration'/'targets.npy').exists() else None for family in ('maia3','maia2')}
    # Targets remain reproducible from vocabulary / mover; cache tensors may be released.
    targets={family:targets[family] if targets[family] is not None else calibration_targets(rows,family,out) for family in targets}
    scores=[]
    for t in c['models']['temperature_grid']:
        losses=[]
        for record in registry['models']:
            logits=np.load(Path(out)/'logits'/f"{record['model_id']}_calibration.npy",mmap_mode='r')
            ids=[i for i,r in enumerate(rows) if record['identity'] is None or r['player_id']==record['identity']]
            losses.append(weighted_nll(probabilities(logits[ids],t),targets[record['family']][ids],equal_weights([rows[i] for i in ids])))
        scores.append(dict(temperature=float(t),nll=float(np.mean(losses))))
    selected=min(scores,key=lambda r:(r['nll'],r['temperature']))['temperature']
    records={}
    for record in registry['models']:
        record['temperature']=selected; record['alpha']=1.
        if record['identity']:
            ids=[i for i,r in enumerate(rows) if r['player_id']==record['identity']]
            personal=probabilities(np.load(Path(out)/'logits'/f"{record['model_id']}_calibration.npy",mmap_mode='r')[ids],selected)
            group=probabilities(np.load(Path(out)/'logits'/f"{record['group']}_calibration.npy",mmap_mode='r')[ids],selected)
            result=choose_alpha(personal,group,targets[record['family']][ids],equal_weights([rows[i] for i in ids]),c['models']['alpha_grid'])
            record['alpha']=result['alpha']; records[record['model_id']]=result
    write_json(Path(out)/'calibration.json',dict(unified_temperature=selected,temperature_candidates=scores,models=records,
       source_rows_sha256=rows_hash(rows),selection_split='calibration',tie_rules='lower temperature / lower alpha',all_players_retained=True))

def calibration_targets(rows,family,out):
    # Saved compact labels/indices only for permitted dev Calibration/H.
    return np.load(Path(out)/'logits'/f'{family}_{rows[0]["split"]}_targets.npy')

def evaluate_registry(registry,rows,destination,out,c):
    destination=Path(destination); destination.mkdir(exist_ok=True)
    targets={family:np.load(Path(out)/'logits'/f'{family}_H_targets.npy') for family in ('maia3','maia2')}
    all_records=[]; ensembles=defaultdict(list); groups={}
    for record in registry['models']:
        if record.get('cohort')!='dev': continue
        mid=record['model_id']; family=record['family']; temp=record['temperature']
        p=probabilities(np.load(Path(out)/'logits'/f'{mid}_H.npy',mmap_mode='r'),temp)
        if record['identity']:
            g=probabilities(np.load(Path(out)/'logits'/f"{record['group']}_H.npy",mmap_mode='r'),temp)
            p=record['alpha']*p+(1-record['alpha'])*g
            if record['role']=='T': ensembles[record['identity']].append((p,record['alpha']))
        else: groups[mid]=summarize_predictions(p,targets[family],rows,mid,None)
        records=summarize_predictions(p,targets[family],rows,mid,record['identity'])
        all_records.extend(records)
        del p
    for player,values in ensembles.items():
        p=ensemble_distribution([v[0] for v in values],[v[1] for v in values])
        all_records.extend(summarize_predictions(p,targets['maia3'],rows,f'T_ensemble_{player}',player))
    pq.write_table(pa.Table.from_pylist(all_records),destination/'dev_prediction_matrix.parquet',compression='zstd')
    write_json(destination/'dev_prediction_matrix.json',all_records)
    def by_player(mid):
        result=defaultdict(list)
        for r in all_records:
            if r['model_id']==mid: result[r['player_id']].append((r['nll'],r['games']))
        return {p:float(sum(n*g for n,g in v)/sum(g for n,g in v)) for p,v in result.items()}
    players=sorted(ensembles); g=by_player('G'); e3g=by_player('E3_group'); e2g=by_player('E2_group')
    t=[g[p]-by_player(f'T_ensemble_{p}')[p] for p in players]
    e3=[e3g[p]-by_player(f'E3_{p}_4242')[p] for p in players]
    e2=[e2g[p]-by_player(f'E2_{p}_6161')[p] for p in players]
    e3base=[by_player('G0')[p]-by_player(f'E3_{p}_4242')[p] for p in players]
    e2base=[by_player('E2_base')[p]-by_player(f'E2_{p}_6161')[p] for p in players]
    gate=qualified_development_gate(t,float(np.mean(e3)),float(np.mean(e3base)),float(np.mean(e2)),float(np.mean(e2base)))
    gate.update(players=[dict(player_id=p,T_improvement=t[i],E3_improvement=e3[i],E2_improvement=e2[i]) for i,p in enumerate(players)],
        windows=['H1','H2'],window_weighting='pooled H1/H2 games equally; then people equally',
        decisions=len(rows),matrix_rows=len(all_records),all_eight_players_retained=True,
        confirmation_H_opened=False,method_correctness_evidence='overfit_resume_evidence.json, maia3_adapter_parity.json, maia2_adapter_parity.json')
    return all_records,gate

def validate_predictors(registry,samples_path,cohort,splits):
    if cohort!='dev' or set(splits)!={'H1','H2'}: raise PermissionError('B03 validation only full dev H1/H2')
    out=Path(registry).parent; value=json.loads(Path(registry).read_text()); verify_registry(value)
    rows,_=load_rows(Path(samples_path).parent,'dev',splits,out/'access_log.jsonl')
    gate=json.loads((out/'dev_gate.json').read_text())
    matrix=pq.read_table(out/'validation'/'dev_prediction_matrix.parquet').to_pylist()
    players={r['player_id'] for r in rows}
    for record in value['models']:
        if record['cohort']=='dev':
            present={(r['player_id'],r['split']) for r in matrix if r['model_id']==record['model_id']}
            if present!={(p,w) for p in players for w in splits}: raise ValueError('Incomplete development matrix')
    return dict(status='PASS',models_verified=True,players=len(players),decisions=len(rows),splits=list(splits),
        matrix_rows=len(matrix),research_status=gate['research_status'],confirmation_H_opened=False)

def run_validation(config_path,cohort,splits,out):
    if cohort!='dev': raise PermissionError('Confirmation H1/H2 remains sealed')
    from nemesis.v2.guards import v2_bound,require_new_output
    if v2_bound():
        require_new_output(out)
        if set(splits.split(','))!={'H1','H2'}: raise PermissionError('V2 validation only development H1/H2')
        c=yaml.safe_load(Path(config_path).read_text(encoding='utf8')); check_config(c)
        from nemesis.v2.assets import validate_readonly
        destination=Path(out)
        if destination.exists() and any(destination.iterdir()): raise FileExistsError('Validation audit exists')
        destination.mkdir(parents=True,exist_ok=True)
        write_json(destination/'validation_receipt.json',validate_readonly(Path('artifacts/B03'),destination))
        return 0
    c=yaml.safe_load(Path(config_path).read_text(encoding='utf8')); check_config(c)
    out=Path(out); root=out.parent
    if (out/'validation_receipt.json').exists(): raise FileExistsError('Validation audit exists')
    meter=B03Meter(root,gpu=False)
    try:
        result=validate_predictors(root/'model_registry.json',Path('artifacts/B01/samples.parquet'),cohort,tuple(splits.split(',')))
        write_json(out/'validation_receipt.json',result); return 0
    finally: meter.checkpoint(enforce=False)

def train_confirmation(registry,c,out,meter,config_sha,code_sha):
    """Same fixed method after positive gate; Fit/Calibration only, bounded disk."""
    from nemesis.v2.guards import reject_legacy_training
    reject_legacy_training()
    rows_a,split=load_rows(Path('artifacts/B01'),'confirm',('fit_a',),Path(out)/'access_log.jsonl')
    rows_b,_=load_rows(Path('artifacts/B01'),'confirm',('fit_b',),Path(out)/'access_log.jsonl')
    rows_cal,_=load_rows(Path('artifacts/B01'),'confirm',('calibration',),Path(out)/'access_log.jsonl')
    check_lineage(rows_a,rows_b); require_calibration(rows_cal)
    temperature=json.loads((Path(out)/'calibration.json').read_text())['unified_temperature']; runs=[]
    calibration_records={}
    for family in ('maia3','maia2'):
        from .maia3_bridge import Maia3Predictor
        from .maia2_bridge import Maia2Predictor
        predictor=(Maia3Predictor if family=='maia3' else Maia2Predictor)(device='cuda')
        policy=FrozenPolicy(predictor,family)
        cal=build_input_cache(policy,predictor,rows_cal,Path(out)/'cache'/f'{family}_confirm_calibration',meter)
        tasks=[('fit_a','G_confirm',1729,'T'),('fit_b','E3_group_confirm',4242,'E3')] if family=='maia3' else [('fit_b','E2_group_confirm',6161,'E2')]
        for partition,group_id,seed,role in tasks:
            fit_rows=rows_a if partition=='fit_a' else rows_b
            fit=build_input_cache(policy,predictor,fit_rows,Path(out)/'cache'/f'{family}_confirm_{partition}',meter)
            group_id={'T':'G','E3':'E3_group','E2':'E2_group'}[role]
            control=next(r for r in registry['models'] if r['model_id']==group_id)
            load_checkpoint(control['checkpoint'],policy.personal,control['metadata'])
            group_probs=predict_cached(policy,cal,None,temperature,meter)
            jobs=[]
            for player in split['confirm_players']:
                pid=[i for i,r in enumerate(fit_rows) if r['player_id']==player]
                for s in c['models']['train_seeds'] if role=='T' else [seed]:
                    ids=[pid[i] for i in bootstrap_indices([fit_rows[j] for j in pid],s)] if role=='T' else pid
                    jobs.append((f'{role}_{player}_{s}',player,s,ids,role=='T'))
            for mid,player,s,ids,bootstrap in jobs:
                cal_ids=[i for i,r in enumerate(rows_cal) if player is None or r['player_id']==player]
                meta=dict(base_sha=predictor.record['sha256'],split_sha=rows_hash([fit_rows[i] for i in ids]),config_sha=config_sha,
                    code_sha=code_sha,model_id=mid,seed=s,family=family,split=partition,cohort='confirm',player_id=player,
                    bootstrap=bootstrap,source_sha=digest('artifacts/B01/samples.parquet'),calibration_sha=rows_hash([rows_cal[i] for i in cal_ids]))
                folder=Path(out)/'checkpoints'/mid
                result=fit_cached(policy,fit,ids,meta,folder,calibration=(cal,cal_ids),meter=meter,
                    resume=(folder/'resume.pt').exists())
                runs.append(result)
                personal=predict_cached(policy,cal,cal_ids,temperature,meter)
                if player is None: group_probs=personal; alpha=1.
                else:
                    selection=choose_alpha(personal,group_probs[cal_ids],cal.arrays['targets'][cal_ids],
                        equal_weights([rows_cal[i] for i in cal_ids]),c['models']['alpha_grid'])
                    alpha=selection['alpha']; calibration_records[mid]=selection
                record=dict(model_id=mid,status='TRAINED',family=family,cohort='confirm',identity=player,role=role if player else 'group',
                    group=group_id if player else None,checkpoint=str(folder/'selected.pt'),checkpoint_sha256=digest(folder/'selected.pt'),
                    metadata=meta,temperature=temperature,alpha=alpha,train_decisions=len(ids),unique_games=len({fit_rows[i]['game_id'] for i in ids}),
                    selected_epoch=min(result['epochs'],key=lambda r:r['calibration_nll'])['epoch'],initialization='official_base_head' if family=='maia2' else 'independent_zero_up',
                    base_unchanged=result['base_state_before']==result['base_state_after'],gradient_max=result['gradient_max'])
                registry['models']=[r for r in registry['models'] if r['model_id']!=mid]+[record]
                write_json(Path(out)/'model_registry.json',registry); meter.checkpoint()
            if role=='T': save_support(fit,cal,split['confirm_players'],out,meter)
            remove_features(fit,out)
        remove_features(cal,out)
        del predictor,policy,fit,cal; gc.collect(); torch.cuda.empty_cache()
    write_json(Path(out)/'confirmation_training_receipt.json',dict(status='DONE',players=24,models=len(runs),
        temperature_from_dev=temperature,alpha_selections=calibration_records,confirmation_H_opened=False,
        row_hashes={k:rows_hash(v) for k,v in [('fit_a',rows_a),('fit_b',rows_b),('calibration',rows_cal)]}))
    return runs
