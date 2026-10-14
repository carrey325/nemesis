"""Selector contains T/G only; no evaluator path is accepted."""
from dataclasses import asdict
import math
from pathlib import Path
import random
import statistics
import chess
from nemesis.types import PositionContext
from .engine import hash_object,rebuild_root,expected_value
from .types import Selection

def branch_context(root,move):
    b=rebuild_root(root); b.push_uci(move)
    return PositionContext(b.fen(),root.history_uci+(move,),root.initial_fen,b.turn,1900,1900,180,0,None,
      hash_object([root.history_uci,move]))

def choose_scores(scores,cp):
    if not scores or any(not math.isfinite(v) for v in scores.values()): raise ValueError('Invalid selection scores')
    return min(scores,key=lambda m:(-scores[m],-cp[m],m))

def robust_score(values,lambda_):
    return statistics.mean(values)-lambda_*statistics.pstdev(values)

def mix_probabilities(distributions):
    if not distributions or any(set(p)!=set(distributions[0]) for p in distributions):
        raise ValueError('MIX8 requires identical full legal vocabularies')
    return {r:statistics.mean(p[r] for p in distributions) for r in distributions[0]}

class PolicyBank:
    access_role='selector'
    def __init__(self,registry_path,model_ids,identity='GENERIC'):
        import json
        from nemesis.models.personal_predictor import registry_record
        registry=json.loads(Path(registry_path).read_text())
        # Check every role before the first weight is opened.
        for mid in model_ids: registry_record(registry,mid,'selector')
        from nemesis.models.personal_predictor import TwinPredictor
        self.models=[TwinPredictor(registry_path,mid,'selector') for mid in model_ids]
        self.identity=identity; self.model_hash=hash_object([m.model_hash for m in self.models])
    def probabilities(self,root,move):
        context=branch_context(root,move)
        return [dict(zip(d.moves_uci,d.probabilities)) for d in (m.predict(context) for m in self.models)]

def bind_selection(root_id,method_id,identity,move,bundle,policy_hash,lambda_,seed,traces,
                   *,matched_nodes=None,actual_nodes=None,engine_call_hash=None):
    # Versioned sidecar is part of the selection commitment, not post-seal metadata.
    trace=dict(schema='B04A_SELECTION_TRACE_V1',root_id=root_id,method_id=method_id,
      identity=identity,move_uci=move,candidate_hash=hash_object(asdict(bundle)),
      policy_hash=policy_hash,lambda_=lambda_,seed=seed,engine_hash=bundle.engine_hash,
      budget_hash=bundle.budget_hash,matched_nodes=matched_nodes,actual_nodes=actual_nodes,
      engine_call_hash=engine_call_hash)
    selection=Selection(root_id,method_id,identity,move,hash_object(trace),policy_hash)
    if traces is not None: traces.append(trace)
    return selection


def matched_selection(bundle,measure,traces):
    call=measure['matched_call']; nodes=measure['online_nodes']
    if (measure['root_id']!=bundle.root_id or call['namespace']!='sf_compute_matched'
        or call['requested_nodes']!=nodes or nodes<=0 or call['actual_nodes']<nodes):
        raise ValueError('Matched search provenance mismatch')
    policy_hash=hash_object([bundle.engine_hash,nodes,call])
    return bind_selection(bundle.root_id,'SF_COMPUTE_MATCHED','GENERIC',
      measure['sf_compute_matched_move'],bundle,policy_hash,0,None,traces,
      matched_nodes=nodes,actual_nodes=call['actual_nodes'],engine_call_hash=hash_object(call))


def select_move(root,bundle,policy,method_id,lambda_,*,traces=None):
    if getattr(policy,'access_role',None)!='selector': raise PermissionError('Selector policy role required')
    if bundle.root_id!=root.root_id: raise ValueError('Candidate root mismatch')
    if bundle.status not in ('ELIGIBLE','NO_BRANCH','OUT_OF_RANGE','MATE'): raise RuntimeError('Missing or terminal root')
    seed=20261010+int(hash_object(root.root_id)[:12],16) if method_id=='SAFE_RANDOM' else None
    cp={m:v for m,v in bundle.cp_root.items() if v is not None}
    if not cp or bundle.status=='MATE':
        move=bundle.moves[0]
        return bind_selection(root.root_id,method_id,policy.identity,move,bundle,policy.model_hash,lambda_,seed,traces)
    safe=bundle.safe_moves
    if bundle.status!='ELIGIBLE' or method_id=='SF_BEST':
        move=min(cp,key=lambda m:(-cp[m],m))
    elif method_id=='SAFE_RANDOM':
        move=random.Random(seed).choice(sorted(safe))
    else:
        scores={}
        for m in safe:
            leaves=bundle.reply_values[m]
            if '__terminal__' in leaves: scores[m]=leaves['__terminal__']; continue
            q=[expected_value(p,leaves) for p in policy.probabilities(root,m)]
            scores[m]=robust_score(q,lambda_ if method_id=='PERSONAL_ROBUST' else 0)
        move=choose_scores(scores,cp)
    return bind_selection(root.root_id,method_id,policy.identity,move,bundle,policy.model_hash,lambda_,seed,traces)
