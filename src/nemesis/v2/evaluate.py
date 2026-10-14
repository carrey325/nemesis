"""Independent E worker: sealed selections first, audit namespace afterwards."""
from dataclasses import asdict
import json
from pathlib import Path
import statistics
from nemesis.data.common import digest,write_json
from .engine import hash_object,expected_value
from .types import AuditRecord

def validate_traces(selections,traces):
    if len(selections)!=len(traces): raise ValueError('Selection trace count mismatch')
    required={'schema','root_id','method_id','identity','move_uci','candidate_hash','policy_hash',
      'lambda_','seed','engine_hash','budget_hash','matched_nodes','actual_nodes','engine_call_hash'}
    for s,t in zip(selections,traces):
        if (set(t)!=required or t['schema']!='B04A_SELECTION_TRACE_V1'
            or hash_object(t)!=s.selection_hash or t['policy_hash']!=s.source_policy_hash
            or any(t[k]!=getattr(s,k) for k in ('root_id','method_id','identity','move_uci'))):
            raise ValueError('Selection trace binding mismatch')
        if s.method_id=='SF_COMPUTE_MATCHED' and (not t['engine_hash'] or not t['engine_call_hash']
            or not isinstance(t['matched_nodes'],int) or t['matched_nodes']<=0
            or not isinstance(t['actual_nodes'],int) or t['actual_nodes']<t['matched_nodes']):
            raise ValueError('Matched trace missing engine/node provenance')


def seal_selections(selections,path,*,traces=None):
    path=Path(path)
    if path.exists(): raise FileExistsError('Selection seal exists')
    if selections and traces is None: raise ValueError('New selection seal requires traces')
    traces=[] if traces is None else traces
    validate_traces(selections,traces)
    body=dict(selections=[asdict(s) for s in selections],traces=traces)
    write_json(path,dict(body,selection_hash=hash_object(body)))

def verify_selection_seal(selections,path,*,traces=None,require_trace=False):
    path=Path(path)
    if not path.exists(): raise PermissionError('E access requires an existing selection seal')
    saved=json.loads(path.read_text())
    rows=[asdict(s) for s in selections]
    if 'traces' in saved:
        validate_traces(selections,saved['traces'])
        if traces is not None and traces!=saved['traces']: raise ValueError('Request trace mismatch')
        body=dict(selections=rows,traces=saved['traces'])
        if saved['selections']!=rows or saved['selection_hash']!=hash_object(body):
            raise ValueError('Selection trace seal mismatch')
    else:
        # Read-only historical R01 verification; never permitted for fresh scoring.
        if require_trace or traces is not None: raise ValueError('Legacy seal lacks selection traces')
        if saved['selection_hash']!=hash_object(rows): raise ValueError('Selection seal mismatch')
    return digest(path)

def evaluate_selections(roots,selections,evaluator_registry,config):
    # Designed to be called by the evaluator subprocess, never by the selector.
    seal_hash=verify_selection_seal(selections,config['_selection_seal'],
      traces=config.get('_selection_traces'),require_trace=True)
    import os
    if config.get('_selector_pid') is None or config['_selector_pid']==os.getpid():
        raise PermissionError('E scoring must run in a separate process from selector')
    registry=json.loads(Path(evaluator_registry).read_text())
    rmap={r['model_id']:r for r in registry['models']}
    for mid in config['_evaluator_ids']:
        r=rmap[mid]
        is_base=mid=='E2_base' and r['family']=='maia2' and r['status']=='BASE'
        is_twin=r.get('role') in ('E3','E2','group') and r.get('metadata',{}).get('split')=='fit_b' and r['family'] in ('maia3','maia2')
        if not (is_base or is_twin):raise PermissionError('E scorer refuses selector/base-G0 roles before loading')
    from .select import branch_context
    from nemesis.models.personal_predictor import TwinPredictor
    output=[]; values=config['_audit_values']; cp=config['_audit_cp']
    config['_all_move_values']=[];config['_role_costs']=[]
    rmap={r.root_id:r for r in roots}
    for mid in config['_evaluator_ids']:
        import time
        tick=time.perf_counter();cpu=time.process_time()
        model=TwinPredictor(evaluator_registry,mid,'evaluator'); family=model.record['family']
        target=model.record['identity'] or mid
        contexts=[]; keys=[]
        for root in roots:
            for m,leaves in values[root.root_id].items():
                if '__terminal__' not in leaves: contexts.append(branch_context(root,m)); keys.append((root.root_id,m))
        predicted={}
        for start in range(0,len(contexts),32):
            for key,d in zip(keys[start:start+32],model.predict_many(contexts[start:start+32])):
                predicted[key]=dict(zip(d.moves_uci,d.probabilities))
        for rid,move_values in values.items():
            for move,leaves in move_values.items():
                q=leaves['__terminal__'] if '__terminal__' in leaves else expected_value(predicted[(rid,move)],leaves)
                config['_all_move_values'].append(dict(root_id=rid,evaluator_model=mid,evaluator_family=family,
                  target_id=target,move_uci=move,q=q,model_hash=model.model_hash))
        for s in selections:
            leaves=values[s.root_id][s.move_uci]
            q=leaves['__terminal__'] if '__terminal__' in leaves else expected_value(predicted[(s.root_id,s.move_uci)],leaves)
            original=config.get('_original_candidates',{}).get(s.root_id,list(cp[s.root_id]))
            finite=[cp[s.root_id][m] for m in original if cp[s.root_id][m] is not None]
            stronger=[v for v in cp[s.root_id].values() if v is not None]
            move_cp=cp[s.root_id][s.move_uci]
            loss=None if move_cp is None or not finite else max(0,max(finite)-move_cp)
            stronger_loss=None if move_cp is None or not stronger else max(0,max(stronger)-move_cp)
            output.append(AuditRecord(s.root_id,s.method_id,target,family,q,loss,
              hash_object([seal_hash,model.model_hash,leaves,cp[s.root_id]]),
              'MATE_CP_UNDEFINED' if loss is None else None,stronger_loss))
        del model
        config['_role_costs'].append(dict(model_id=mid,cpu_seconds=time.process_time()-cpu,
          wall_seconds=time.perf_counter()-tick,phase='load+full-COST32-scoring'))
    return output

def false_weakness_controls():
    # Transparent complete legal distributions. Two replies carry the trap mass;
    # every other legal reply has explicit zero probability. Safe U is constant .5.
    import chess
    from .types import RootContext,CandidateBundle
    from .select import select_move
    root=RootContext('SYNTHETIC','fixture',('P','calibration','white'),chess.STARTING_FEN,chess.STARTING_FEN,(),True,'fixture')
    leaves={}
    for m in ('e2e4','d2d4'):
        b=chess.Board();b.push_uci(m);moves=sorted(r.uci() for r in b.legal_moves)
        leaves[m]={r:(float(i==0) if m=='e2e4' else .5) for i,r in enumerate(moves)}
    bundle=CandidateBundle(root.root_id,('e2e4','d2d4'),('e2e4','d2d4'),{'e2e4':0,'d2d4':0},leaves,'SYNTHETIC','SYNTHETIC','ELIGIBLE')
    class Policy:
        access_role='selector';identity='SYNTHETIC';model_hash='fixture'
        def __init__(self,ps):self.ps=ps
        def probabilities(self,root,m):
            keys=list(leaves[m]);result=[]
            for p in self.ps:
                values={r:0. for r in keys};values[keys[0]]=p;values[keys[1]]=1-p;result.append(values)
            return result
    def choose(ps,lam):return select_move(root,bundle,Policy(ps),'PERSONAL_ROBUST' if lam else 'PERSONAL_PLAIN',lam).move_uci
    def gain(ps,truth,lam=0):
        m=choose(ps,lam);return expected_value(Policy([truth]).probabilities(root,m)[0],leaves[m])-.5
    single=[.95,.35,.35];pred=gain([.65]*3,.65);actual=gain([.65]*3,.35)
    return dict(SYNTHETIC=True,fixture='trap=e2e4 U(first reply)=1, other replies=0; safe=d2d4 U=.5; full legal zero tail retained',
      F01=dict(independent_gain=gain([.65]*3,.65),status='TRUE_WEAKNESS'),
      F02=dict(predicted_seed_q=single,robust_move='safe' if choose(single,1)=='d2d4' else 'trap'),
      F03=dict(predicted_gain=pred,independent_gain=actual,std=statistics.pstdev([.65]*3),
        robust_move=choose([.65]*3,1),status='FALSE_WEAKNESS' if pred>0 and actual<0 else 'NOT_DETECTED'),
      F04=dict(gain=gain([.65]*3,.65),specificity=gain([.65]*3,.65)-gain([.65]*3,.65)),
      F05=dict(loss25='SAFE',loss25_01='EXCLUDED'),F12=dict(engine_failure='MISSING'),
      F14=dict(confirmation_H='DENIED_BEFORE_READ',E_in_selector='DENIED_BEFORE_LOAD'))

def main():
    import argparse
    from .panels import root_context
    from .types import Selection
    import time
    p=argparse.ArgumentParser(); p.add_argument('request'); p.add_argument('output'); a=p.parse_args()
    request=json.loads(Path(a.request).read_text())
    if any(r['panel']=='S_compare' for r in request['roots']): raise PermissionError('S_compare sealed')
    roots=[root_context(r) for r in request['roots']]; sels=[Selection(**s) for s in request['selections']]
    verify_selection_seal(sels,request['config']['_selection_seal'],
      traces=request['config'].get('_selection_traces'),require_trace=True)
    start=time.perf_counter(); cpu=time.process_time()
    records=evaluate_selections(roots,sels,Path(request['registry']),request['config'])
    write_json(a.output,dict(records=[asdict(r) for r in records],cpu_seconds=time.process_time()-cpu,
      gpu_seconds=time.perf_counter()-start,actor='independent_evaluator_process',
      source_sha256=digest(__file__),request_sha256=digest(a.request),
      selection_seal_sha256=digest(request['config']['_selection_seal']),
      all_move_values=request['config']['_all_move_values'],role_costs=request['config']['_role_costs']))

if __name__=='__main__': main()
