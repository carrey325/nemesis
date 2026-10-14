"""Read frozen choices and independent E values; never infer or choose a model."""
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from nemesis.data.common import write_json,digest
from .panels import DEV
from .engine import hash_object
from .statistics import frequency,matrix_summary,bootstrap_endpoints
from .diagnostics import headroom,pilot_decision

def table(path,rows):pq.write_table(pa.Table.from_pylist(rows),path,compression='zstd')

def build_outputs(out,panels,bundles,selected,audits,evaluation,risk_rows,risk_values,risk_predictions,meter):
    out=Path(out);rmap={r['root_id']:r for r in panels['S_dev']+panels['TUNE']}
    selections=selected['selections'];choice={(s['root_id'],s['method_id'],s['identity']):s['move_uci'] for s in selections}
    q={(r['root_id'],r['evaluator_family'],r['target_id'],r['move_uci']):r for r in evaluation['values']}
    generic=lambda rid:choice[rid,'GENERIC','GENERIC']
    moves=lambda rid,method:[choice[rid,method,p] for p in DEV]
    records=[];root_boot=[];head=[];safety=[];freqrows=[];lambda_scores=[];perwindow=[];matrix_rows=[]
    all_methods=['PERSONAL_PLAIN','PERSONAL_ROBUST','TEMP_09','TEMP_11']
    sums={}
    for panel in ('S_dev','TUNE'):
        for row in panels[panel]:
            rid=row['root_id'];bundle=bundles[rid];cp=audits[rid]['cp'];finite=[cp[m] for m in bundle.moves if cp[m] is not None]
            stronger=[v for v in cp.values() if v is not None]
            losses={m:None if cp[m] is None or not finite else max(0,max(finite)-cp[m]) for m in cp}
            for s in [s for s in selections if s['root_id']==rid]:
                m=s['move_uci'];safety.append(dict(root_id=rid,panel=panel,method=s['method_id'],identity=s['identity'],loss_cp=losses[m],
                    stronger_loss_cp=None if cp[m] is None or not stronger else max(0,max(stronger)-cp[m]),
                    safe_low_budget=m in bundle.safe_moves if s['method_id']!='SF_COMPUTE_MATCHED' else None,
                    outside_nomination=m not in bundle.moves,cp_mate=cp[m] is None,
                    loss_reason='MATE_CP_UNDEFINED' if losses[m] is None else None))
            families={}
            for family in ('E3','E2'):
                safe_q=np.array([[q[rid,family,p,m]['q'] for p in DEV] for m in bundle.safe_moves]);h=headroom(safe_q,bundle.safe_moves.index(generic(rid)))
                for j,p in enumerate(DEV):head.append(dict(root_id=rid,panel=panel,family=family,target_id=p,H=h['H'][j],H_ID=h['H_ID']))
                families[family]={}
                for method in all_methods+(['LAMBDA_05','LAMBDA_2'] if panel=='TUNE' else []):
                    msm=moves(rid,method);values=np.array([[q[rid,family,p,m]['q'] for p in DEV] for m in msm]);base=np.array([q[rid,family,p,generic(rid)]['q'] for p in DEV]);delta=values-base
                    summary=matrix_summary(delta);families[family][method]=summary
                    sums[panel,family,method,rid]=delta
                    for i,identity in enumerate(DEV):
                        for j,p in enumerate(DEV):
                            m=msm[i];value=q[rid,family,p,m]
                            record=dict(root_id=rid,source_game_id=row['source_game_id'],stratum=row['sampling_stratum'],panel=panel,
                              source_player=row['source_player_id'],window=row['source_window'],root_color=row['root_color'],family=family,
                              method=method,strategy_identity=identity,target_id=p,move=m,delta=float(delta[i,j]),q=float(values[i,j]),
                              generic_q=float(base[j]),loss_cp=losses[m],H=h['H'][j],H_ID=h['H_ID'],
                              loss_reason='MATE_CP_UNDEFINED' if losses[m] is None else None,
                              q_temp09=value['q_temp09'],q_temp11=value['q_temp11'],q_sensitivity=value['q_sensitivity'])
                            for baseline in ('GENERIC_G0','MIX8','SF_BEST','SAFE_RANDOM','SF_COMPUTE_MATCHED'):
                                bid='GENERIC' if baseline in ('SF_COMPUTE_MATCHED','SF_BEST','SAFE_RANDOM') else baseline
                                bm=choice[rid,baseline,bid];record['delta_vs_'+baseline]=float(values[i,j]-q[rid,family,p,bm]['q'])
                            records.append(record)
            change=np.array([m!=generic(rid) for m in moves(rid,'PERSONAL_PLAIN')]);safe=np.array([losses[m] is not None and losses[m]<=27 for m in moves(rid,'PERSONAL_PLAIN')])
            g3=np.array(families['E3']['PERSONAL_PLAIN']['per_target_G']);s3=np.array(families['E3']['PERSONAL_PLAIN']['per_target_S'])
            g2=np.array(families['E2']['PERSONAL_PLAIN']['per_target_G']);s2=np.array(families['E2']['PERSONAL_PLAIN']['per_target_S'])
            useful=change & (g3>=.005)&(g2>=.005)&safe;specific=useful&(s3>=.005)&(s2>=.005)
            h3=[h['H'] for h in head[-16:-8]];h2=[h['H'] for h in head[-8:]]
            if panel=='S_dev':root_boot.append(dict(root_id=rid,source_game_id=row['source_game_id'],stratum=row['sampling_stratum'],
                vectors=np.stack([g3,s3,g2,s2,h3,h2,useful.astype(float),specific.astype(float),
                  [h['H_ID'] for h in head[-16:-8]],[h['H_ID'] for h in head[-8:]]]).tolist(),
                safe_q_E3=[[q[rid,'E3',p,m]['q'] for p in DEV] for m in bundle.safe_moves],
                safe_q_E2=[[q[rid,'E2',p,m]['q'] for p in DEV] for m in bundle.safe_moves],
                changed=change.tolist(),useful=useful.tolist(),specific=specific.tolist()))
    for family in ('E3','E2'):
        for method in all_methods:
            roots=[sums['S_dev',family,method,r['root_id']] for r in panels['S_dev']]
            m=np.mean(roots,axis=0);summ=matrix_summary(m)
            for i,p in enumerate(DEV):
                for j,target in enumerate(DEV):matrix_rows.append(dict(method=method,family=family,strategy_identity=p,target_identity=target,delta=float(m[i,j]),n_roots=256))
            write_json(out/(family+'_'+method+'_summary.json'),summ)
        table(out/(family+'_matrix.parquet'),[r for r in matrix_rows if r['family']==family])
        for w in ('H1','H2'):
            ids=[r['root_id'] for r in panels['S_dev'] if r['source_window']==w]
            perwindow.append(dict(family=family,window=w,**matrix_summary(np.mean([sums['S_dev',family,'PERSONAL_PLAIN',rid] for rid in ids],axis=0))))
    table(out/'root_effects.parquet',records);table(out/'audit_values.parquet',evaluation['values']);table(out/'safety.parquet',safety)
    table(out/'support.parquet',selected['support'])
    table(out/'disagreement.parquet',selected['disagreement'])
    table(out/'selections.parquet',selections)
    fby={};yield_values={'E3':0.,'E2':0.};useful_total=specific_total=union_total=sunion_total=0
    for h in sorted({r['sampling_stratum'] for r in panels['N_dev']}):
        n=[r for r in panels['N_dev'] if r['sampling_stratum']==h];f=frequency([r['status'] for r in n]);fby[h]=f
        group=[r for r in root_boot if r['stratum']==h];arr=np.array([r['vectors'] for r in group]);means=arr.mean(axis=0)
        useful_total+=f['eligible_rate']*means[6].mean()/32;specific_total+=f['eligible_rate']*means[7].mean()/32
        union_total+=f['eligible_rate']*np.array([r['useful'] for r in group]).any(axis=1).mean()/32
        sunion_total+=f['eligible_rate']*np.array([r['specific'] for r in group]).any(axis=1).mean()/32
        for family,index in [('E3',0),('E2',2)]:yield_values[family]+=f['eligible_rate']*means[index].mean()/32
        for j,p in enumerate(DEV):freqrows.append(dict(panel='N_dev_times_S_dev',stratum=h,target_id=p,**f,
             changed=float(np.array([r['changed'] for r in group])[:,j].mean()),useful=float(means[6,j]),specific_useful=float(means[7,j])))
    table(out/'frequency_by_stratum.parquet',freqrows)
    table(out/'natural_status.parquet',panels['N_dev'])
    write_json(out/'population_yield.json',dict(Y=yield_values,F_useful_person_macro=useful_total,F_specific_person_macro=specific_total,
        F_useful_any_identity=union_total,F_specific_any_identity=sunion_total,frequency=fby,
        scope='Game-first source/window/color equal strata; two independent panels; not practical game reachability',missing_total=sum(v['missing'] for v in fby.values())))
    bootstrap=bootstrap_endpoints(root_boot,panels['N_dev'],dict(identities=list(DEV),replicates=5000));write_json(out/'bootstrap.json',bootstrap)
    upper=[bootstrap['results']['same_ID_block'][f+'_H']['upper_975'] for f in ('E3','E2')]
    write_json(out/'headroom.json',dict(per_root=head,H_upper_975_simultaneous_two_families=upper,
       H_ID_CI95={f:bootstrap['results']['same_ID_block'][f+'_H_ID']['ci95'] for f in ('E3','E2')},
       meaning='Optimistic fixed candidate/evaluator geometry, not a human upper bound',mean_H_ID={f:float(np.mean([r['H_ID'] for r in head if r['panel']=='S_dev' and r['family']==f])) for f in ('E3','E2')}))
    for lam,method in [(0,'PERSONAL_PLAIN'),(.5,'LAMBDA_05'),(1,'PERSONAL_ROBUST'),(2,'LAMBDA_2')]:
        scores={f:matrix_summary(np.mean([sums['TUNE',f,method,r['root_id']] for r in panels['TUNE']],axis=0))['S'] for f in ('E3','E2')}
        lambda_scores.append(dict(lambda_=lam,scores=scores,objective=min(scores.values())))
    winner=min(lambda_scores,key=lambda r:(-r['objective'],r['lambda_']))
    write_json(out/'lambda_selection.json',dict(panel='TUNE64',scores=lambda_scores,selected_lambda=winner['lambda_'],rule='max min(E3,E2) S; tie smaller lambda',pilot_primary_remains_lambda0=True,S_compare_outcome_reads=0))
    # Each risk decision is game unique; personal ensemble risk is mean of full probabilities' event mass.
    pred={(r['sample_id'],r['model_id']):r['risk'] for r in risk_predictions['predictions']};risk_table=[]
    for row in risk_rows:
        sid=row['sample_id'];vals=risk_values[sid]['values'];actual=row['target_move_uci'];event=max(vals.values())-vals[actual]>=.05
        personal=np.mean([pred[sid,f'T_{row["player_id"]}_{seed}'] for seed in (1729,2718,3141)])
        g=pred[sid,'G'];risk_table.append(dict(sample_id=sid,game_id=row['game_id'],player_id=row['player_id'],window=row['split'],
          event=bool(event),actual_loss_U=max(vals.values())-vals[actual],risk_G=g,risk_Tmean=float(personal),
          brier_G=(g-event)**2,brier_Tmean=float((personal-event)**2),source_input_hash=row['input_hash']))
    table(out/'human_risk.parquet',risk_table)
    risk_summary=[]
    risk_rng=np.random.default_rng(8675309)
    for w in ('H1','H2'):
        subset=[r for r in risk_table if r['window']==w];events=sum(r['event'] for r in subset)
        per_person=[np.array([r['brier_G']-r['brier_Tmean'] for r in subset if r['player_id']==p]) for p in DEV]
        draws=[]
        for _ in range(5000):
            identity_draw=risk_rng.integers(0,8,8)
            draws.append(np.mean([per_person[j][risk_rng.integers(0,len(per_person[j]),len(per_person[j]))].mean() for j in identity_draw]))
        risk_summary.append(dict(window=w,n=len(subset),events=events,evidence_status='SUFFICIENT_EVENTS' if events>=50 else 'INSUFFICIENT_EVENTS',
             brier_G=float(np.mean([r['brier_G'] for r in subset])),brier_Tmean=float(np.mean([r['brier_Tmean'] for r in subset])),
             brier_improvement_ci95=np.quantile(draws,[.025,.975]).tolist(),bootstrap='5000 paired identity then within-person/window whole games; seed8675309'))
    qualification=json.loads(Path('artifacts/V2/B03R/dev_qualification_by_window.json').read_text())
    historical=json.loads(Path('artifacts/V2/B03R/dev_metrics_replay.json').read_text())
    historical_path=Path('artifacts/B03/validation/dev_prediction_matrix.parquet')
    if digest(historical_path)!=historical['input_sha256']:raise ValueError('Historical prediction matrix changed')
    historic_matrix=pq.read_table(historical_path).to_pylist()
    if len(historic_matrix)!=848:raise ValueError('Historical full prediction matrix incomplete')
    write_json(out/'historical_prediction_matrix.json',dict(source_sha256=digest(historical_path),rows=historic_matrix,
        scope='Approved cached full-development queue aggregates: NLL,Top1,Top3,multiclass Brier, correct/wrong identities, all8 players and both windows; not recomputed inference'))
    risk_person=[]
    for p in DEV:
        for w in ('H1','H2'):
            group=[r for r in risk_table if r['player_id']==p and r['window']==w]
            risk_person.append(dict(player_id=p,window=w,n=len(group),events=sum(r['event'] for r in group),
               brier_G=float(np.mean([r['brier_G'] for r in group])),brier_Tmean=float(np.mean([r['brier_Tmean'] for r in group]))))
    write_json(out/'per_player_windows.json',dict(utility_windows=perwindow,risk_windows=risk_summary,risk_player_windows=risk_person,qualification=qualification,historical_prediction_metrics=historical))
    # Identity/source deletion keeps the wrong-policy contrast at the original7.
    loo=[]
    for dropped in DEV:
        group=[r for r in root_boot if r['stratum'].split('|')[0]!=dropped];keep=[i for i,p in enumerate(DEV) if p!=dropped]
        arr=np.array([r['vectors'] for r in group]);loo.append(dict(dropped_identity_and_source=dropped,G_S_H=arr[:,:6,keep].mean(axis=(0,2)).tolist()))
    write_json(out/'leave_one_out.json',dict(same_identity_source_target=loo,
       source_only=[dict(dropped_source=p,G_S_H=np.array([r['vectors'] for r in root_boot if r['stratum'].split('|')[0]!=p])[:,:6].mean(axis=(0,2)).tolist()) for p in DEV]))
    # Post-seal sensitivity and baseline comparisons retain adverse cases.
    sensitivities=[]
    for family in ('E3','E2'):
        for field in ('q_sensitivity','q_temp09','q_temp11'):
            matrices=[]
            for row in panels['S_dev']:
                rid=row['root_id'];ms=moves(rid,'PERSONAL_PLAIN')
                if q[rid,family,DEV[0],generic(rid)][field] is None:continue
                matrix=np.array([[q[rid,family,p,m][field]-q[rid,family,p,generic(rid)][field] for p in DEV] for m in ms]);matrices.append(matrix)
            if matrices:sensitivities.append(dict(family=family,variant=field,n_roots=len(matrices),**matrix_summary(np.mean(matrices,axis=0))))
        for label,field in [('09','q_temp09'),('11','q_temp11')]:
            if (panels['S_dev'][0]['root_id'],'TEMP_GENERIC_'+label,'GENERIC') not in choice:continue
            matrices=[]
            for row in panels['S_dev']:
                rid=row['root_id'];bm=choice[rid,'TEMP_GENERIC_'+label,'GENERIC'];ms=moves(rid,'TEMP_'+label)
                matrices.append(np.array([[q[rid,family,p,m][field]-q[rid,family,p,bm][field] for p in DEV] for m in ms]))
            sensitivities.append(dict(family=family,variant='Joint_T_'+label+'_selector_generic_evaluator',n_roots=256,**matrix_summary(np.mean(matrices,axis=0))))
    write_json(out/'sensitivity.json',dict(temperature='Original calibrated T x0.9/x1.1 before each raw policy softmax and alpha mixture; selector and E variants separately recorded',
       evaluator_and_budget_summaries=sensitivities,
       high_budget_roots=len({r['root_id'] for r in safety if r['root_id'] in audits and 'sensitivity' in audits[r['root_id']]}),
       fixed_hash_rule="int(hash_object(['sensitivity',root_id])[:16],16)%10==0",selection_frozen=True))
    plain=[r for r in safety if r['panel']=='S_dev' and r['method']=='PERSONAL_PLAIN']
    violation=sum(r['loss_cp'] is None or r['loss_cp']>27 for r in plain)/len(plain)
    safety_ok=violation<=.05
    summaries={f:matrix_summary(np.mean([sums['S_dev',f,'PERSONAL_PLAIN',r['root_id']] for r in panels['S_dev']],axis=0)) for f in ('E3','E2')}
    # B03R pooled qualification is reused with per-column/window caveats.
    qualified=historical['E3_vs_group']>0 and historical['E2_vs_group']>0 and historical['E2_vs_base']>0
    decision=pilot_decision(upper,256,qualified)
    gates=dict(gain_specificity=all(summaries[f][k]>=.005 for f in summaries for k in ('G','S')),
      positive_targets=all(sum(x>0 for x in summaries[f]['per_target_S'])>=4 for f in summaries),
      windows_nonnegative=all(r[k]>=0 for r in perwindow for k in ('G','S')),frequency=useful_total>=.05,
      safety=safety_ok,qualification=qualified,unique_opportunity_games=sum(any(r['useful']) for r in root_boot)>=20,
      budget=meter.actual_cpu()/3600<11 and meter.gpu/3600<1)
    gates={k:bool(v) for k,v in gates.items()}
    decision.update(reference_gates=gates,reference_gates_all=all(gates.values()),summaries=summaries,
       safety_violation_fraction=violation,stage='B04B',expansion_authorized=False,
       interpretation='Developer investment reference only; valid negative effects retain all8 players and all roots; downstream requires independent review')
    write_json(out/'pilot_decision.json',decision)
    # Calibration-only disagreement median; no effect-dependent cutpoints.
    tune_ids={r['root_id'] for r in panels['TUNE']};support={(r['root_id'],r['target_id']):r for r in selected['support']}
    disagreement={(r['root_id'],r['target_id'],r['move']):r for r in selected['disagreement']}
    threshold=float(np.median([r['std'] for r in selected['disagreement'] if r['root_id'] in tune_ids]))
    strata=defaultdict(list)
    import chess
    for r in records:
        if r['panel']!='S_dev' or r['method']!='PERSONAL_PLAIN' or r['strategy_identity']!=r['target_id']:continue
        root=rmap[r['root_id']];rid=r['root_id'];target=r['target_id'];board=chess.Board(root['fen6'])
        phase='opening_late' if root['ply']<=30 else 'endgame' if len(board.piece_map())<=12 else 'middlegame'
        split=[('window',root['source_window']),('root_color',root['root_color']),('phase',phase),
            ('support','high' if support[rid,target]['high_support'] else 'low'),
            ('T_disagreement','high' if disagreement[rid,target,r['move']]['std']>threshold else 'low')]
        for dimension,value in split:strata[r['family'],dimension,value].append(r['delta'])
    write_json(out/'stratified_diagnostics.json',dict(disagreement_threshold_from_TUNE64=threshold,
        phase_rule='Original B01: ply<=30 opening late; then<=12 pieces endgame; otherwise middlegame',
        rows=[dict(family=k[0],dimension=k[1],value=k[2],n=len(v),mean_gain=float(np.mean(v))) for k,v in strata.items()]))
    # NLL gain versus utility is descriptive across8 people, with original roles/windows retained.
    write_json(out/'prediction_to_decision.json',dict(players=[dict(player_id=p,recorded_NLL_gain=next(r['T'] for r in historical['players'] if r['player_id']==p),
        E3_G=summaries['E3']['per_target_G'][i],E3_S=summaries['E3']['per_target_S'][i],
        E2_G=summaries['E2']['per_target_G'][i],E2_S=summaries['E2']['per_target_S'][i]) for i,p in enumerate(DEV)],
        claim='Descriptive eight-player comparison; no causal prediction-to-utility conversion claim'))
    group_evaluators=[]
    for family,target in [('E3','E3_group'),('E2','E2_group'),('E2','E2_base')]:
        if (panels['S_dev'][0]['root_id'],family,target,generic(panels['S_dev'][0]['root_id'])) not in q:continue
        for method in ('PERSONAL_PLAIN','PERSONAL_ROBUST'):
            gains=[]
            for row in panels['S_dev']:
                rid=row['root_id'];base=q[rid,family,target,generic(rid)]['q']
                gains.extend(q[rid,family,target,m]['q']-base for m in moves(rid,method))
            group_evaluators.append(dict(evaluator=target,method=method,mean_gain=float(np.mean(gains)),
                interpretation='No target identity axis; group/base evaluator diagnostic, not specificity evidence'))
    write_json(out/'group_base_evaluator_diagnostics.json',dict(rows=group_evaluators))
    role_rows=[]
    for phase,body in [('selector',selected),('evaluator',evaluation),('risk_predict',risk_predictions)]:
        for c in body['role_costs']:role_rows.append(dict(method='A00_ASSET',role=phase,run_id='B04B-N01',**c,
             historical_training_cpu=None,cold_equivalent=None,reuse_compute_match=False))
    for u in meter.units:
        if u['work_unit_id'].startswith(('decision-','audit-','risk-')):role_rows.append(dict(method='SHARED_ENGINE',role=u['work_unit_id'].split('-')[0],run_id='B04B-N01',
          model_id=None,cpu_seconds=u['cpu_seconds'],wall_seconds=u['end_unix']-u['start_unix'],nodes=u['nodes'],historical_training_cpu=None,cold_equivalent=None,reuse_compute_match=False))
    table(out/'role_costs.parquet',role_rows)
    # Cold-equivalent screening cost is diagnostic only: historical CPU is
    # already in inherited_actual and must not be charged again.
    screen_path=Path('artifacts/V2/B04A/local/engine_calls.json')
    old_manifest=json.loads(Path('artifacts/V2/B04A/artifact_manifest.json').read_text())
    assert digest(screen_path)==next(r['sha256'] for r in old_manifest['files'] if r['path']=='local/engine_calls.json')
    screen_calls={c['cache_key']:c for c in json.loads(screen_path.read_text())}
    from .engine import rebuild_root
    from .panels import root_context
    cold_equivalents=[]
    for row in panels['S_dev']+panels['TUNE']:
        decision_path=out/'local/decision'/(hash_object(row['root_id'])+'.json')
        if not decision_path.exists():
            if (out/'run_binding.json').exists():raise FileNotFoundError('Missing production decision cost checkpoint')
            continue # explicit SYNTHETIC report fixture has no engine payload
        rid=row['root_id'];bundle=bundles[rid];board=rebuild_root(root_context(row));cpu=0.
        for nodes,forced,k in [(100000,None,min(5,board.legal_moves.count()))]+[(50000,[m],1) for m in bundle.moves]:
            key=hash_object(['decision',board.root().fen(),[m.uci() for m in board.move_stack],board.turn,nodes,forced,k,bundle.engine_hash])
            cpu+=screen_calls[key]['cpu_seconds']
        if decision_path.exists():
            body=json.loads(decision_path.read_text());matched_cpu=body['measure']['matched_call']['cpu_seconds']
            cold_equivalents.append(dict(root_id=rid,screen_reused_cpu_seconds=cpu,
              current_personal_engine_cpu_seconds=body['cpu_seconds']-matched_cpu,
              cold_personal_engine_equivalent_cpu_seconds=body['cpu_seconds']-matched_cpu+cpu,
              separate_SF_COMPUTE_MATCHED_cpu_seconds=matched_cpu,historical_screen_recharged=False))
    selector_costs={r['model_id']:r for r in selected['role_costs']}
    target_costs=[]
    for p in DEV:
        mids=[f'T_{p}_{seed}' for seed in (1729,2718,3141)]
        target_costs.append(dict(target_id=p,exclusive_model_CPU_seconds=sum(selector_costs[mid]['cpu_seconds'] for mid in mids) if mids[0] in selector_costs else None,
          exclusive_model_resident_wall_seconds=sum(selector_costs[mid]['wall_seconds'] for mid in mids) if mids[0] in selector_costs else None))
    write_json(out/'cost_by_role.json',dict(rows=role_rows,shared_engine_charged_once=True,historical_per_role_cost='UNKNOWN',reuse_compute_match=False,
       cold_vs_screen_reuse_engine=cold_equivalents,single_target_exclusive_inference=target_costs,
       eight_target_selector_CPU_seconds=sum(r['cpu_seconds'] for r in selected['role_costs']),
       inference_scope='Measured model calls include fixed temperature diagnostics; shared G/G0 and support cost reported in role rows, not charged8 times. CUDA actor full wall is authoritative residency.',
       cold_online_node_equivalent=[dict(root_id=row['root_id'],reused_screen_nodes=row['cold_screen_actual_nodes'],
         actual_matched_nodes=next(s for s in selected['traces'] if s['root_id']==row['root_id'] and s['method_id']=='SF_COMPUTE_MATCHED')['matched_nodes']) for row in panels['S_dev']+panels['TUNE']],
       p50_p95_CPU_seconds={phase:np.quantile([u['cpu_seconds'] for u in meter.units if u['work_unit_id'].startswith(phase+'-')],[.5,.95]).tolist() for phase in ('decision','audit','risk')},
       single_target_vs_all8='Engine shared once; inference costs per actual model; single-target attribution is diagnostic and historical training remains unknown'))
    # Anonymous rows are locators and scores only; no source game IDs, FEN, move or policy payload.
    safe_extract=[{k:r[k] for k in ('root_id','family','method','strategy_identity','target_id','delta','loss_cp','loss_reason','H','H_ID')} for r in records if r['panel']=='S_dev' and r['method']=='PERSONAL_PLAIN']
    write_json(out/'anonymous_effects_extract.json',dict(rows=safe_extract,source_parquet_sha256=digest(out/'root_effects.parquet'),rows_total=len(safe_extract)))
    # Plotting reads saved matrix tables only.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for f in ('E3','E2'):
        data=pq.read_table(out/(f+'_matrix.parquet')).to_pylist();m=np.array([r['delta'] for r in data if r['method']=='PERSONAL_PLAIN']).reshape(8,8)
        fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(m,cmap='RdBu',vmin=-max(abs(m).max(),.005),vmax=max(abs(m).max(),.005));ax.set_xticks(range(8),DEV,rotation=45);ax.set_yticks(range(8),DEV);ax.set_xlabel('Independent evaluator target');ax.set_ylabel('Selection identity');ax.set_title(f+' delta U vs GENERIC_A00');fig.colorbar(im,ax=ax);fig.tight_layout()
        fig.savefig(out/(f+'_matrix.png'));fig.savefig(out/(f+'_matrix.svg'));plt.close(fig)
    write_json(out/'report_source_binding.json',dict(source_hashes={str(p):digest(p) for p in Path('src/nemesis/v2').glob('*.py')},
        selector_seal_sha256=selected['selection_seal_sha256'],evaluator_seal_sha256=evaluation['selection_seal_sha256']))
