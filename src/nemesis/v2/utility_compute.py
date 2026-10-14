"""Two independent persistent SF sessions; one writer, stable root boundaries."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict,replace
import json
from pathlib import Path
import time
import chess
from nemesis.data.common import write_json,digest
from .engine import EngineSession,rebuild_root,reply_leaves,hash_object,audit_bundle
from .panels import root_context
from .types import RootContext,CandidateBundle,Selection
from .evaluate import verify_selection_seal

class ResourceLimit(RuntimeError):pass

def sensitivity_root(rid):return int(hash_object(['sensitivity',rid])[:16],16)%10==0

def root_nodes(row,bundle):
    board=rebuild_root(root_context(row));replies=[];extra=[]
    for move in board.legal_moves:
        branch=board.copy();branch.push(move)
        n=0 if branch.is_game_over(claim_draw=True) else branch.legal_moves.count()
        (replies if move.uci() in bundle.safe_moves else extra).append(n)
    screen=row['cold_screen_actual_nodes'];r=sum(replies);x=max(extra,default=0)
    decision=r*20000+screen+r*20000
    audit=(len(bundle.moves)+1+r+x)*200000
    sensitivity=(len(bundle.moves)+1+r+x)*400000 if sensitivity_root(row['root_id']) else 0
    return dict(decision=decision,audit=audit,sensitivity=sensitivity,replies=r,max_extra_replies=x)

class ComputeMeter:
    def __init__(self,out,inherited,allowance=1800):
        self.out=Path(out);self.inherited=inherited;self.cpu_start=time.process_time();self.start=time.time()
        self.allowance=allowance;self.sessions=[];self.closed_engine_cpu=0.;self.child_cpu=0.;self.prior_cpu=0.;self.gpu=0.;self.units=[]
    def actual_cpu(self):
        return self.prior_cpu+time.process_time()-self.cpu_start+self.closed_engine_cpu+self.child_cpu+sum(s.cpu_seconds() for s in self.sessions)
    def gate(self,cpu=0,gpu=0):
        used=(self.actual_cpu()+self.allowance)/3600
        if used+1.25*cpu/3600>10.45 or self.inherited['project_actual']['cpu_core_hours']+used+1.25*cpu/3600>52:
            raise ResourceLimit('Next complete work unit exceeds CPU stage/project gate with 5% reserve')
        if (self.gpu+1.25*gpu)/3600>.95 or self.inherited['project_actual']['gpu_hours']+(self.gpu+1.25*gpu)/3600>20:
            raise ResourceLimit('Next complete work unit exceeds GPU stage/project gate')
    def save(self):
        components=dict(prior_checkpoint_cpu_seconds=self.prior_cpu,parent_all_thread_cpu_seconds=time.process_time()-self.cpu_start,
          closed_engine_cpu_seconds=self.closed_engine_cpu,active_engine_cpu_seconds=sum(s.cpu_seconds() for s in self.sessions),
          new_CUDA_actor_cpu_seconds=self.child_cpu)
        cpu=(sum(components.values())+self.allowance)/3600;gpu=self.gpu/3600;inherited=self.inherited['project_actual']
        caps=dict(stage_cpu_core_hours=11,stage_gpu_hours=1,project_cpu_core_hours=52,project_gpu_hours=20,disk_gib=80,external_spend_usd=0)
        project={r:inherited[r]+v for r,v in [('cpu_core_hours',cpu),('gpu_hours',gpu)]}
        ledger=dict(stage='B04B',attempt_id='B04B-N01',inherited_actual=inherited,stage_actual=dict(cpu_core_hours=cpu,gpu_hours=gpu),
          project_actual=project,caps=caps,
          remaining=dict(stage_cpu_core_hours=11-cpu,stage_gpu_hours=1-gpu,project_cpu_core_hours=52-project['cpu_core_hours'],project_gpu_hours=20-project['gpu_hours']),
          inherited_binding=self.inherited,work_units=self.units,
          measurement_components=components,
          allowances=[dict(work_unit_id='B04B-N01-implementation-tests-router-A03-coordination-git-report',owner_stage='B04B',cpu_seconds=self.allowance,gpu_seconds=0,
            reason='Conservative inclusive allowance for implementation, shell/tests, Router A03 adoption/coordination, plotting/report, commit/push; measured runner CPU and engine/GPU children charged separately')],
          measurement_method='Parent process_time all threads once; each of two SF psutil user+system lifetime CPU once; GPU child all-thread CPU plus startup allowance; full GPU child wall residency, release at exit; no division by concurrency',
          measurement_start_unix=self.start,measurement_end_unix=time.time(),
          disk=dict(inherited_peak_gib=46.48238283511251,new_stage_bytes=sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file()),peak_upper_bound_gib=54.48238283511251),
          per_role_cost=dict(historical_training=None,reason='UNATTRIBUTED; reuse_compute_match=false'),external_spend_usd=0)
        write_json(self.out/'resource_ledger.json',ledger)

def compute_phase(rows,bundles,config,meter,phase,*,seal=None,selections=None,traces=None):
    """Two roots max in flight; main thread alone commits complete checkpoints."""
    sessions=[EngineSession(config),EngineSession(config)];meter.sessions=sessions
    if any(b.engine_hash!=sessions[0].engine_hash or b.budget_hash!=hash_object(config['engine']) for b in bundles.values()):
        for session in sessions:
            total=session.cpu_seconds();session.close();meter.closed_engine_cpu+=total
        meter.sessions=[];meter.save()
        raise ValueError('Cached bundle engine/budget differs from the locked live sessions')
    def unit(row,session):
        before=session.cpu_seconds();tick=time.time();calls_before=len(session.calls)
        if phase=='risk':
            root=RootContext(row['sample_id'],row['game_id'],(row['player_id'],row['split'],row['mover_color']),row['initial_fen'],row['fen6'],tuple(row['history_uci']),row['mover_color']=='white',row['input_hash'])
            board=rebuild_root(root);values={}
            for move in sorted(board.legal_moves,key=lambda m:m.uci()):
                branch=board.copy();branch.push(move)
                from .engine import terminal_value
                u=terminal_value(branch,board.turn)
                if u is None:u=session.analyse(board,board.turn,200000,[move],namespace='real_risk')[0]['u']
                values[move.uci()]=u
            body=dict(sample_id=row['sample_id'],values=values)
        else:
            root=root_context(row);bundle=bundles[root.root_id]
            if phase=='decision':
                start_nodes=session.nodes
                leaves=reply_leaves(root,bundle.safe_moves,session,20000,'decision')
                online_nodes=row['cold_screen_actual_nodes']+session.nodes-start_nodes
                matched=session.analyse(rebuild_root(root),root.root_color,online_nodes,namespace='sf_compute_matched')[0]
                call=session.calls[-1]
                body=dict(bundle=asdict(replace(bundle,reply_values=leaves)),measure=dict(root_id=root.root_id,online_nodes=online_nodes,
                    sf_compute_matched_move=matched['move'],matched_call=call),cold_screen_actual_nodes=row['cold_screen_actual_nodes'])
            else:
                cp,leaves=audit_bundle(root,bundle,session,nodes=200000,selection_seal=seal,selections=selections,traces=traces)
                body=dict(cp=cp,values=leaves)
                if sensitivity_root(root.root_id):
                    scp,svalues=audit_bundle(root,bundle,session,nodes=400000,selection_seal=seal,selections=selections,traces=traces)
                    body['sensitivity']=dict(cp=scp,values=svalues)
        body['calls']=session.calls[calls_before:]
        body['cpu_seconds']=session.cpu_seconds()-before;body['start_unix']=tick;body['end_unix']=time.time()
        return body
    results={};ordered=sorted(rows,key=lambda r:r.get('root_id',r.get('sample_id')))
    def requested(row):
        if phase=='risk':return chess.Board(row['fen6']).legal_moves.count()*200000
        ns=root_nodes(row,bundles[row['root_id']]);return ns[phase]+(ns['sensitivity'] if phase=='audit' else 0)
    planned_nodes=[requested(row) for row in ordered]
    measured_cpu=measured_nodes=0.;rate=config['_seconds_per_node']
    folder=meter.out/'local'/phase;folder.mkdir(exist_ok=True)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for start in range(0,len(ordered),2):
                chunk=ordered[start:start+2];forecast=0
                for row in chunk:
                    forecast+=requested(row)*rate
                meter.gate(cpu=forecast)
                # Retain a refreshed full-remainder forecast, while the hard start
                # gate checks the largest complete next pair and the lifecycle.
                write_json(meter.out/'rolling_forecast.json',dict(phase=phase,completed=start,total=len(ordered),
                    remaining_phase_nodes=sum(planned_nodes[start:]),measured_cpu_seconds=measured_cpu,
                    measured_actual_nodes=measured_nodes,seconds_per_node=rate,multiplier=1.25,
                    next_pair_cpu_seconds_with_margin=1.25*forecast,
                    remaining_phase_cpu_hours_with_margin=1.25*sum(planned_nodes[start:])*rate/3600,
                    current_stage_cpu_hours=(meter.actual_cpu()+meter.allowance)/3600,
                    current_project_cpu_hours=meter.inherited['project_actual']['cpu_core_hours']+(meter.actual_cpu()+meter.allowance)/3600,
                    stage_gate=10.45,project_gate=52))
                futures=[pool.submit(unit,row,sessions[i]) for i,row in enumerate(chunk)]
                for row,future in zip(chunk,futures):
                    body=future.result();rid=row.get('root_id',row.get('sample_id'))
                    path=folder/(hash_object(rid)+'.json')
                    if path.exists():raise FileExistsError('Work unit already exists')
                    write_json(path,body);results[rid]=body
                    measured_cpu+=body['cpu_seconds'];measured_nodes+=sum(c['actual_nodes'] for c in body['calls'])
                    rate=max(config['_seconds_per_node'],measured_cpu/max(1,measured_nodes))
                    meter.units.append(dict(work_unit_id=phase+'-'+rid,owner_stage='B04B',input_hash=hash_object(row),
                        start_unix=body['start_unix'],end_unix=body['end_unix'],cpu_seconds=body['cpu_seconds'],gpu_seconds=0,
                        output_sha256=digest(path),nodes=sum(c['actual_nodes'] for c in body['calls'])))
                meter.save()
                print(phase,start+len(chunk),'/',len(ordered),'CPUh',round((meter.actual_cpu()+meter.allowance)/3600,4),flush=True)
                # Long-lived engine sessions retain no cross-root hash/value cache.
                for s in sessions:s.cache.clear();s.calls.clear()
    finally:
        for session in sessions:
            # Include interpreter startup/configuration since process birth, not per-call sum.
            total=session.cpu_seconds();session.close();meter.closed_engine_cpu+=total
        meter.sessions=[];meter.save()
    return results
