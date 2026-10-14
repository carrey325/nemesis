"""B01 outputs and independent integrity checks. No model calls."""
import json
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
import chess
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from .acquire import acquire_games,SourceBlocked
from .common import digest,stable_hash,write_json,append_json,Meter,ResourceLimit
from .prepare import load_games,coverage,decision_rows,write_samples,input_fingerprint,SAMPLE_SCHEMA
from .splits import exclude_cross_cohort,assert_split_integrity,split_for

def table(path,rows,schema):
    pq.write_table(pa.Table.from_pylist(rows,schema=pa.schema(schema)),path,compression='zstd')

def finish_dataset(config,out,meter=None):
    local=Path('data/B01'); state=json.loads((local/'acquisition_checkpoint.json').read_text()); discovery=json.loads((local/'candidates_private.json').read_text())
    names={f'C{i:03d}':name for i,(name,_) in enumerate(discovery['ranking'][:128],1)}
    outcomes=[]; selected=[]; retained={}; excluded=[]
    for cid,evidence in sorted(state.get('source_exclusions',{}).items()):
        outcomes.append({'candidate_id':cid,'eligible':False,'reasons':[evidence['reason']],'source_sha256':evidence['receipt_sha256'],'downloaded_games':None,'unique_games':None,'source_exclusion':evidence})
        excluded.append({'game_id':None,'candidate_id':cid,'player_id':None,'reason':evidence['reason'],'source_game_index':None,'source_sha256':evidence['receipt_sha256']})
    for r in state['completed']:
        cid=r['candidate_id']; name=names[cid]; games={}
        for index,g in enumerate(load_games([Path(r['path'])],meter,source_rated_blitz=True),1):
            if g['exclusion']:
                excluded.append({'game_id':g['game_id'],'candidate_id':cid,'player_id':None,'reason':g['exclusion'],'source_game_index':index,'source_sha256':r['sha256']})
            games[g['game_id']]=g
        c=coverage(list(games.values()),name)
        outcomes.append({'candidate_id':cid,**c,'source_sha256':r['sha256'],'downloaded_games':r['downloaded_games'],'unique_games':len(games)})
        if c['eligible']: selected.append(cid); retained[cid]=games
    outcomes.sort(key=lambda r:r['candidate_id'])
    outcome_index={r['candidate_id']:r for r in outcomes}
    # Repeatedly exclude candidates failing coverage after mutual games removed.
    removed=set()
    while True:
        chosen=[c for c in selected if c not in removed][:32]; cohort={names[c] for c in chosen}; failures=[]
        for cid in chosen:
            c=coverage(list(retained[cid].values()),names[cid],cohort)
            if not c['eligible']: failures.append(cid); outcome_index[cid]['post_cross_reasons']=c['reasons']
        if not failures: break
        removed.update(failures)
    sufficient=len(chosen)==32
    # No smaller exploratory cohort is declared a valid target cohort.
    if not sufficient: chosen=[]; cohort=set()
    order=chosen.copy(); random.Random(20261007).shuffle(order); dev=set(order[:8]); identities={cid:f'P{i:03d}' for i,cid in enumerate(chosen,1)}
    write_json(local/'private_identity_map.json',{identities[c]:names[c] for c in chosen})
    write_json(out/'candidate_outcomes.json',{'outcomes':outcomes,'selected':chosen,'insufficient':not sufficient,'required_players':32,'eligible_pre_cross':len(selected),'cross_failures':sorted(removed),'verified_candidates':len(outcomes),'max_candidates':128})
    player_rows=[]; game_rows=[]; rows=[]; seen=set(); duplicates=0
    for cid in chosen:
        pid=identities[cid]; name=names[cid]; c=coverage(list(retained[cid].values()),name,cohort)
        player_rows.append({'player_id':pid,'cohort':'dev' if cid in dev else 'confirm','candidate_id':cid,'fit_median_elo':float(c['fit_median_elo']),'coverage_json':json.dumps(c['partitions'],sort_keys=True)})
        usable={split:[] for split in ['fit_a','fit_b','calibration','H1','H2']}
        for g in retained[cid].values():
            reason=g['exclusion']
            if reason: continue  # already recorded for every candidate above
            if name not in (g['white'],g['black']): reason='source_subject_mismatch'
            if not reason and exclude_cross_cohort(g,cohort): reason='cross_cohort_game'
            if reason:
                excluded.append({'game_id':g['game_id'],'candidate_id':cid,'player_id':pid,'reason':reason,'source_game_index':None,'source_sha256':next(r['sha256'] for r in state['completed'] if r['candidate_id']==cid)}); continue
            usable[g['split']].append(g)
        for split,bucket in usable.items():
            bucket.sort(key=lambda g:stable_hash([g['game_id'],20261007]))
            for g in bucket[2000:]: excluded.append({'game_id':g['game_id'],'candidate_id':cid,'player_id':pid,'reason':'partition_hash_cap','source_game_index':None,'source_sha256':next(r['sha256'] for r in state['completed'] if r['candidate_id']==cid)})
            for g in bucket[:2000]:
                if g['game_id'] in seen: duplicates+=1; continue
                seen.add(g['game_id']); game_rows.append({'game_id':g['game_id'],'player_id':pid,'split':split,'utc':g['utc'],'source_candidate_id':cid,'source_sha256':next(r['sha256'] for r in state['completed'] if r['candidate_id']==cid),'plies':len(g['moves'])})
                samples=decision_rows(g,name,pid)
                for s in samples: s['cohort']='dev' if cid in dev else 'confirm'
                rows.extend(samples)
                if meter and len(seen)%500==0: meter.checkpoint()
    # D6: remove identical complete model inputs from primary confirmation sets.
    prior={s['input_hash'] for s in rows if s['split'] in {'fit_a','fit_b','calibration'}}
    prior_boards={s['board_hash'] for s in rows if s['split'] in {'fit_a','fit_b','calibration'}}
    overlap=0; board_overlap=0; sensitivity=[]; main=[]; sealed=[]
    for s in rows:
        if s['cohort']=='confirm' and s['split'] in {'H1','H2'}:
            if s['input_hash'] in prior: s['main_eligible']=False; overlap+=1
            if s['board_hash'] in prior_boards: board_overlap+=1
            sealed.append(s)
        elif s['main_eligible']: main.append(s)
        else: sensitivity.append(s)
    table(out/'players.parquet',player_rows,[('player_id',pa.string()),('cohort',pa.string()),('candidate_id',pa.string()),('fit_median_elo',pa.float64()),('coverage_json',pa.string())])
    table(out/'games_manifest.parquet',game_rows,[('game_id',pa.string()),('player_id',pa.string()),('split',pa.string()),('utc',pa.string()),('source_candidate_id',pa.string()),('source_sha256',pa.string()),('plies',pa.int32())])
    table(out/'exclusions.parquet',excluded,[('game_id',pa.string()),('candidate_id',pa.string()),('player_id',pa.string()),('reason',pa.string()),('source_game_index',pa.int32()),('source_sha256',pa.string())])
    write_samples(out/'samples.parquet',main); write_samples(out/'sensitivity.parquet',sensitivity)
    for split in ['H1','H2']: write_samples(out/('sealed_'+split+'.parquet'),[s for s in sealed if s['split']==split])
    split_m={'dev_players':[p['player_id'] for p in player_rows if p['cohort']=='dev'],'confirm_players':[p['player_id'] for p in player_rows if p['cohort']=='confirm'],'game_counts':dict(Counter(g['split'] for g in game_rows)),'primary_development_samples':len(main),'sensitivity_development_samples':len(sensitivity),'sealed_samples':len(sealed),'confirmation_labels_exposed':False,'source_checkpoint_sha256':digest(local/'acquisition_checkpoint.json'),'cohort_seed':20261007,'status':'READY' if sufficient else 'DATA_INSUFFICIENT'}
    write_json(out/'split_manifest.json',split_m)
    write_json(out/'leakage_audit.json',{**assert_split_integrity(out/'samples.parquet',out/'split_manifest.json'),'global_duplicate_games_removed':duplicates,'complete_input_overlap_removed_from_confirmation_main':overlap,'board_only_overlap_stratified':board_overlap,'cross_cohort_exclusions':sum(e['reason']=='cross_cohort_game' for e in excluded),'coverage_by_phase_color_elo_clock_window':dict(Counter(str((s['split'],s['phase'],s['mover_color'],s['self_elo']//100,s['clock_before'] is None)) for s in rows)),'risk_subset_and_engine_probe_selection':'NOT_RUN: B05/B06; raw legal decision histories retained; no engine or model used'})
    append_json(out/'holdout_access_log.jsonl',{'stage':'B01','action':'PREPROCESS_AND_SEAL','labels_printed':False,'model_metrics_computed':False,'H1_rows':sum(s['split']=='H1' for s in sealed),'H2_rows':sum(s['split']=='H2' for s in sealed)})
    (out/'data_report.md').write_text('# B01 data report\n\nStatus: '+split_m['status']+'\n\n'+f"Verified {len(outcomes)} frozen candidates; {len(selected)} passed preliminary coverage; {len(player_rows)} selected under exact 32-player contract. See candidate_outcomes.json for every gap.\n\nNo training, model performance or confirmation label output. Public accounts exclude known BOT tags; human identity is unverified. Maia2 PRETRAIN_OVERLAP_UNKNOWN persists.\n",encoding='utf-8')
    return split_m

def verify(out,meter=None):
    out=Path(out); m=json.loads((out/'artifact_manifest.json').read_text()); checked=0
    for item in m['files']:
        p=Path(item['path'])
        if not p.exists() or digest(p)!=item['sha256'] or p.stat().st_size!=item['bytes']: raise ValueError('artifact mismatch: '+str(p))
        if p.suffix=='.parquet' and pq.read_metadata(p).num_rows!=item['rows']: raise ValueError('row mismatch')
        checked+=1
    results=assert_split_integrity(out/'samples.parquet',out/'split_manifest.json')
    # Independent replay includes sealed shards, reads only for integrity, logs metadata.
    total=0; game_splits={}; sample_ids=set()
    gm=pq.read_table(out/'games_manifest.parquet').to_pylist()
    if len({g['game_id'] for g in gm})!=len(gm): raise ValueError('duplicate global game manifest')
    game_index={g['game_id']:g for g in gm}; source_games={}
    for cid in sorted({g['source_candidate_id'] for g in gm}):
        source=Path('data/B01')/(cid+'.pgn')
        expected=next(g['source_sha256'] for g in gm if g['source_candidate_id']==cid)
        if digest(source)!=expected: raise ValueError('source PGN hash mismatch')
        source_games[cid]={g['game_id']:g for g in load_games([source],meter,source_rated_blitz=True)}
    identities=json.loads(Path('data/B01/private_identity_map.json').read_text())
    @lru_cache(maxsize=128)
    def contexts(gid,pid):
        import math
        meta=game_index[gid]; raw=source_games[meta['source_candidate_id']][gid]
        who=identities.get(pid)
        if who not in (raw['white'],raw['black']): raise ValueError('raw game does not contain mapped target identity')
        if raw['white'] in identities.values() and raw['black'] in identities.values(): raise ValueError('cross cohort game in sample queue')
        color=raw['white']==who; board=chess.Board(raw['initial_fen']); start_color=board.turn
        eligible=[ply for ply in range(20,min(100,len(raw['moves']))+1) if (start_color if ply%2 else not start_color)==color]
        if len(eligible)>20: eligible=[eligible[round(i*(len(eligible)-1)/19)] for i in range(20)]
        selected=set(eligible); result={}; boards=[board.copy()]; clocks={True:180.0,False:180.0}
        self_elo=raw['elos'][0 if color else 1]; opp=raw['elos'][1 if color else 0]
        for ply,u in enumerate(raw['moves'],1):
            if ply>100: break
            before=clocks[board.turn]; post=raw['clocks'][ply-1]
            if ply in selected:
                result[ply]={'fen6':board.fen(),'input_hash':input_fingerprint(board,boards,self_elo,opp,0.0),'clock_before':before,'self_elo':self_elo,'oppo_elo':opp,'mover_color':'white' if color else 'black'}
            valid=post is not None and math.isfinite(post) and post>=0 and before is not None and post<=before
            clocks[board.turn]=post if valid else None
            move=chess.Move.from_uci(u)
            if move not in board.legal_moves: raise ValueError('illegal raw source history')
            board.push(move); boards.append(board.copy())
            if len(boards)>8: boards.pop(0)
        return result
    for p in [out/'samples.parquet',out/'sensitivity.parquet',out/'sealed_H1.parquet',out/'sealed_H2.parquet']:
        if not pq.read_schema(p).equals(SAMPLE_SCHEMA): raise ValueError('sample schema mismatch')
        for batch in pq.ParquetFile(p).iter_batches(batch_size=4096):
            for s in batch.to_pylist():
                if s['sample_id'] in sample_ids: raise ValueError('duplicate sample across shards')
                sample_ids.add(s['sample_id'])
                if s['game_id'] in game_splits and game_splits[s['game_id']]!=s['split']: raise ValueError('game overlap across sealed/development shards')
                game_splits[s['game_id']]=s['split']
                meta=game_index.get(s['game_id'])
                if not meta or meta['player_id']!=s['player_id'] or meta['utc']!=s['utc']: raise ValueError('sample game lineage mismatch')
                raw=source_games[meta['source_candidate_id']][s['game_id']]
                if raw['exclusion'] or raw['utc']!=s['utc'] or raw['moves'][:s['ply']-1]!=s['history_uci'] or raw['moves'][s['ply']-1]!=s['target_move_uci']: raise ValueError('sample differs from raw game history/target')
                expected=contexts(s['game_id'],s['player_id']).get(s['ply'])
                if expected is None: raise ValueError('sample not in fixed evenly spaced decisions')
                for key,value in expected.items():
                    if s[key]!=value: raise ValueError('raw context mismatch: '+key)
                if s['initial_fen']!=raw['initial_fen'] or s['clock_history']!=raw['clocks'][:s['ply']-1]: raise ValueError('source initial FEN or clock history mismatch')
                if s['sample_id']!=stable_hash([s['game_id'],s['player_id'],s['ply']]): raise ValueError('sample ID mismatch')
                if s['main_eligible'] and (s['clock_before'] is None or s['clock_before']<30): raise ValueError('ineligible clock in primary queue')
                if split_for(s['utc'])!=s['split']: raise ValueError('wrong sealed split')
                total+=1
                if meter and total%5000==0: meter.checkpoint()
    append_json(out/'holdout_access_log.jsonl',{'stage':'B01','action':'INTEGRITY_REPLAY_ONLY','labels_printed':False,'rows':total})
    return {'status':'VALID','manifest_files_checked':checked,'legally_replayed_samples':total,'raw_source_games_linked':len(gm),'all_shard_game_overlap':0,'all_shard_duplicate_samples':0,**results}

def manifest(out):
    files=[]
    for root in [out,Path('data/B01')]:
        for p in sorted(root.glob('*')):
            if not p.is_file() or p.name=='artifact_manifest.json': continue
            item={'path':str(p),'sha256':digest(p),'bytes':p.stat().st_size,'format':p.suffix.lstrip('.'),'local_only':root!=out or p.suffix=='.parquet','source':'official archive/export or B01 deterministic processing','command':'python -m nemesis data prepare --config nemesis-stage1-plan/configs/experiment.proposed.yaml --out artifacts/B01'}
            if p.suffix=='.parquet': item['rows']=pq.read_metadata(p).num_rows
            files.append(item)
    write_json(out/'artifact_manifest.json',{'stage':'B01','attempt_id':'B01-N01','files':files})

def run_data(config_path,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); config=yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    try:
        acquire_games(config,out)
        from .prepare import build_samples
        state=json.loads(Path('data/B01/acquisition_checkpoint.json').read_text())
        build_samples([Path(r['path']) for r in state['completed']],config,out)
        result=json.loads((out/'split_manifest.json').read_text())
        write_json(out/'stage_disposition.json',{'stage':'B01','branch':'main','execution_status':'DONE','artifact_status':'VALID','research_status':'NOT_TESTED' if result['status']=='READY' else 'INSUFFICIENT_EVIDENCE','data_status':result['status'],'permitted_downstream_uses':['B02 input verification'] if result['status']=='READY' else ['independent review of data insufficiency'],'exit_code':0})
        manifest(out); vmeter=Meter(out)
        try: validation=verify(out,vmeter)
        finally: vmeter.checkpoint()
        write_json(out/'data_verification.json',validation); manifest(out)
        return 0
    except (SourceBlocked,ResourceLimit) as exc:
        write_json(out/'stage_disposition.json',{'stage':'B01','branch':'main','execution_status':'BLOCKED','artifact_status':'INCOMPLETE','research_status':'NOT_TESTED','reason':str(exc),'permitted_downstream_uses':['inspect checkpoint; resume only'],'exit_code':4})
        manifest(out); return 4
