from collections import defaultdict
import json
import math
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
from nemesis.data.common import append_json, digest

METRICS=('nll','top1','top3','brier')


def score(distribution,target):
    if target not in distribution.moves_uci: raise ValueError('Target absent from legal policy')
    pairs=sorted(zip(distribution.moves_uci,distribution.probabilities),key=lambda x:(-x[1],x[0]))
    p=distribution.probabilities[distribution.moves_uci.index(target)]
    return dict(nll=-math.log(max(p,1e-30)),top1=int(pairs[0][0]==target),
                top3=int(target in [x[0] for x in pairs[:3]]),
                brier=sum((v-int(m==target))**2 for m,v in pairs),
                confidence=pairs[0][1],predicted_move=pairs[0][0])


def aggregate(rows):
    grouped=defaultdict(list)
    for r in rows: grouped[(r['player_id'],r['game_id'])].append(r)
    games=[]; people=defaultdict(list)
    for (player,game),values in sorted(grouped.items()):
        out=dict(player_id=player,game_id=game,decisions=len(values))
        out.update({k:sum(r[k] for r in values)/len(values) for k in METRICS})
        games.append(out); people[player].append(out)
    players=[]
    for player,values in sorted(people.items()):
        players.append(dict(player_id=player,games=len(values),
                            **{k:sum(r[k] for r in values)/len(values) for k in METRICS}))
    summary={k:sum(r[k] for r in players)/len(players) if players else None for k in METRICS}
    summary.update(players=len(players),games=len(games),decisions=len(rows))
    return games,players,summary


def load_development(root,cohort,log):
    if cohort!='dev':
        append_json(log,dict(action='REJECTED',cohort=cohort,reason='B02 development only'))
        raise PermissionError('B02 never opens confirmation H1/H2')
    root=Path(root)
    path=root/'samples.parquet'
    manifest=json.loads((root/'artifact_manifest.json').read_text())
    split_path=root/'split_manifest.json'
    split_record=next(r for r in manifest['files'] if Path(r['path']).name=='split_manifest.json')
    if digest(split_path)!=split_record['sha256']: raise ValueError('B01 split manifest hash mismatch')
    split=json.loads(split_path.read_text())
    record=next(r for r in manifest['files'] if Path(r['path']).name=='samples.parquet')
    if digest(path)!=record['sha256']: raise ValueError('B01 samples hash mismatch')
    # B01 samples contains permitted Fit/Calibration and dev H only; sealed files
    # are never named, discovered, opened or counted by the B02 evaluator.
    table=pq.read_table(path)
    keep=pc.and_(pc.is_in(table['player_id'],value_set=pa.array(split['dev_players'])),
                 pc.is_in(table['split'],value_set=pa.array(['H1','H2'])))
    selected=table.filter(keep)
    append_json(log,dict(action='READ',cohort='dev',splits=['H1','H2'],path=str(path),
                         sha256=record['sha256'],rows=selected.num_rows))
    return selected


def run_baseline(config_path,cohort,out):
    """Evaluate every primary eligible dev H row; no subsampling or tuning."""
    if cohort!='dev': raise PermissionError('B02 development only')
    out=Path(out)
    if any((out/name).exists() for name in ('run_receipt.json','baseline_dev.parquet','inference_benchmark.json')):
        raise FileExistsError('B02 output already contains a run; explicit inspected checkpoint required')
    out.mkdir(parents=True,exist_ok=True)
    from .budget import B02Meter
    from nemesis.data.common import write_json
    from nemesis.preflight import ResourceLimit
    import yaml
    import time
    import torch
    import gc
    from dataclasses import asdict
    from nemesis.models.common import context_from_row
    from nemesis.models.maia3_bridge import Maia3Predictor
    from nemesis.models.maia2_bridge import Maia2Predictor
    meter=B02Meter(out)
    config=yaml.safe_load(Path(config_path).read_text(encoding='utf8'))
    write_json(out/'run_receipt.json',dict(stage='B02',attempt_id='B02-N01',status='RUNNING',
        source_hashes={str(p):digest(p) for p in sorted(Path('src/nemesis').rglob('*.py'))
                       if 'models' in p.parts or 'evaluation' in p.parts or p.name in ('__main__.py','types.py','preflight.py')},
        config_path=str(config_path),config_sha256=digest(config_path),cohort='dev',
        temperature=1,precision='FP32',primary_model='Maia3-23M',history='real length 8',
        base_commit='ee6af2e9d374403e2a57fa09a944692bb01ccd1a',
        plan_version='sha256:99f22f7176b26acfe1613087e37bea5a64cd8d9c69140d547425a0077681dd4f'))
    writer=None
    try:
        if config['resources']['stage_caps']['B02']!={'gpu_hours':.5,'cpu_core_hours':1}:
            raise ValueError('B02 cap differs from approved contract')
        table=load_development(Path('artifacts/B01'),cohort,out/'holdout_access_log.jsonl')
        if 'main_eligible' in table.column_names: table=table.filter(table['main_eligible'])
        rows=table.to_pylist(); del table
        if not rows: raise ValueError('No development H rows')
        print(f'Development H baseline: {len(rows)} decisions, full frozen primary set',flush=True)
        contexts=[context_from_row(r) for r in rows[:64]]
        benchmark=[]
        # CPU and GPU, including batch/microbatch. FP32 only; warm up first.
        for cls in (Maia3Predictor,Maia2Predictor):
            for device in ('cpu','cuda'):
                if device=='cuda' and not torch.cuda.is_available():
                    benchmark.append(dict(model=cls.__name__,device=device,status='UNAVAILABLE')); continue
                if device=='cuda': meter.gpu_on(); torch.cuda.reset_peak_memory_stats()
                predictor=cls(device=device)
                predictor.predict(contexts[0])
                for batch_size in ((1,8) if device=='cpu' else (1,8,32,64)):
                    tick=time.perf_counter()
                    for offset in range(0,64,batch_size): predictor.predict_many(contexts[offset:offset+batch_size])
                    if device=='cuda': torch.cuda.synchronize()
                    elapsed=time.perf_counter()-tick
                    benchmark.append(dict(model=cls.__name__,model_hash=predictor.record['sha256'],
                        device=device,precision='FP32',batch=batch_size,microbatch=batch_size,
                        effective_batch=64,positions=64,seconds=elapsed,positions_per_second=64/elapsed,
                        peak_vram_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                        peak_vram_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else None,
                        includes_encoding=True,status='PASS'))
                    meter.checkpoint()
                del predictor; gc.collect()
                if device=='cuda': torch.cuda.empty_cache(); meter.gpu_off()
        device='cuda' if torch.cuda.is_available() else 'cpu'
        options=[r for r in benchmark if r['model']=='Maia3Predictor' and r['device']==device and r['status']=='PASS']
        best=max(options,key=lambda r:r['positions_per_second'])
        batch=best['batch']
        # Initial benchmark includes short real histories; don't treat it as a
        # budget guarantee. Both caps are polled throughout the actual run.
        write_json(out/'inference_benchmark.json',dict(status='PASS',measurements=benchmark,
                    tf32_matmul=False,tf32_cudnn=False,
                    locked_mode=dict(device=device,precision='FP32',microbatch=batch,temperature=1,
                                     reason='highest measured full-pipeline warm throughput'),
                    mixed_precision=dict(status='NOT_RUN',error=None)))
        print(f'Locked {device} FP32 microbatch {batch}',flush=True)
        if device=='cuda': meter.gpu_on()
        predictor=Maia3Predictor(device=device)
        trace_games=sorted({r['game_id'] for r in rows})[:10]
        results=[]
        with (out/'trace_samples.jsonl').open('w',encoding='utf8') as traces:
            for start in range(0,len(rows),batch):
                part=rows[start:start+batch]
                contexts=[context_from_row(r) for r in part]
                distributions=predictor.predict_many(contexts)
                output=[]
                for row,ctx,d in zip(part,contexts,distributions):
                    scored=score(d,row['target_move_uci'])
                    record={k:row[k] for k in ('sample_id','game_id','player_id','split','ply','mover_color','phase')}
                    record.update(scored,model_id=d.model_id,model_hash=d.model_hash,model_input_hash=d.input_hash,
                                  b01_input_hash=row['input_hash'])
                    output.append(record); results.append(record)
                    if row['game_id'] in trace_games:
                        # Original locators replaced by one-way anonymous hashes.
                        traces.write(json.dumps(dict(game_alias=__import__('hashlib').sha256(row['game_id'].encode()).hexdigest(),
                            sample_alias=__import__('hashlib').sha256(row['sample_id'].encode()).hexdigest(),
                            player_id=row['player_id'],split=row['split'],context=asdict(ctx),
                            target_move_uci=row['target_move_uci'],distribution=asdict(d),metrics=scored))+'\n')
                chunk=pa.Table.from_pylist(output)
                if writer is None: writer=pq.ParquetWriter(out/'baseline_dev.parquet',chunk.schema,compression='zstd')
                writer.write_table(chunk)
                if start% (batch*16)==0:
                    meter.checkpoint()
                    write_json(out/'progress.json',dict(decisions=len(results),total=len(rows),status='RUNNING'))
                if start% (batch*256)==0: print(f'{len(results)}/{len(rows)} evaluated',flush=True)
        writer.close(); writer=None
        del predictor; gc.collect()
        if device=='cuda': torch.cuda.empty_cache(); meter.gpu_off()
        games,players,summary=aggregate(results)
        strata={}
        for field in ('split','mover_color','phase'):
            strata[field]={v:aggregate([r for r in results if r[field]==v])[2] for v in sorted({r[field] for r in results})}
        # Reliability diagram data uses per-game then per-player weights too.
        calibration=[]
        counts=defaultdict(int); game_counts=defaultdict(set)
        for r in results: counts[(r['player_id'],r['game_id'])]+=1; game_counts[r['player_id']].add(r['game_id'])
        for bin_id in range(10):
            selected=[r for r in results if min(9,int(r['confidence']*10))==bin_id]
            weights=[1/(len(game_counts)*len(game_counts[r['player_id']])*counts[(r['player_id'],r['game_id'])]) for r in selected]
            mass=sum(weights)
            calibration.append(dict(bin=bin_id,weight=mass,decisions=len(selected),
                confidence=sum(w*r['confidence'] for r,w in zip(selected,weights))/mass if mass else None,
                accuracy=sum(w*r['top1'] for r,w in zip(selected,weights))/mass if mass else None))
        write_json(out/'metrics_summary.json',dict(overall=summary,players=players,strata=strata,
                      reliability=calibration,aggregation='mean decisions within game; mean games within player; mean players',
                      cohort='dev',all_primary_eligible_rows_evaluated=True,trace_games=len(trace_games)))
        pq.write_table(pa.Table.from_pylist(games),out/'baseline_by_game.parquet',compression='zstd')
        pq.write_table(pa.Table.from_pylist(players),out/'baseline_by_player.parquet',compression='zstd')
        (out/'baseline_summary.md').write_text('# B02 development G0 baseline\n\n'+
            f"Full frozen development H1/H2: {summary['players']} players, {summary['games']} games, {summary['decisions']} decisions.\n\n"+
            f"Equal game then player weights: NLL {summary['nll']:.6f}; Top-1 {summary['top1']:.6f}; Top-3 {summary['top3']:.6f}; Brier {summary['brier']:.6f}.\n\n"+
            'These are development results at T=1, with true historical context and per-decision Elo. No confirmation results or scientific effectiveness claim. All split, color and phase results and reliability bins are in metrics_summary.json.\n',encoding='utf8')
        write_json(out/'progress.json',dict(decisions=len(results),total=len(rows),status='DONE'))
        receipt=json.loads((out/'run_receipt.json').read_text()); receipt.update(status='DONE',decisions=len(results))
        write_json(out/'run_receipt.json',receipt)
        meter.checkpoint()
        return 0
    except ResourceLimit as exc:
        write_json(out/'resource_stop.json',dict(status='RESOURCE_LIMIT',reason=str(exc),statistics_complete=False))
        return 4
    finally:
        if writer is not None: writer.close()
        meter.gpu_off(); meter.checkpoint(enforce=False)
