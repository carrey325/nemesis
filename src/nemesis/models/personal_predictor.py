"""Loadable B03 policy with registry hashes, fixed calibration and role isolation."""
import hashlib,json
from pathlib import Path
import torch
from nemesis.data.common import digest
from .adapters import FrozenPolicy,load_checkpoint,legal_log_probs
from .common import tensor_hash,distribution

def registry_record(registry,model_id,access_role):
    if access_role not in ('selector','evaluator'): raise ValueError('Explicit selector/evaluator access role required')
    record=next(r for r in registry['models'] if r['model_id']==model_id)
    if record['status'] not in ('TRAINED','BASE'): raise ValueError('Model is not trained: '+model_id)
    if access_role=='selector' and (model_id.startswith('E') or record['family']=='maia2'
        or record.get('role') in ('E3','E2') or record.get('metadata',{}).get('split')=='fit_b'):
        raise PermissionError('Selector cannot read an evaluation model')
    return record

class TwinPredictor:
    def __init__(self,registry_path,model_id,access_role,device='cuda'):
        if device!='cuda': raise ValueError('B03 canonical reproduction requires CUDA FP32')
        registry=json.loads(Path(registry_path).read_text()); self.record=registry_record(registry,model_id,access_role)
        self.device=device; self.model_id=model_id
        from .maia3_bridge import Maia3Predictor
        from .maia2_bridge import Maia2Predictor
        self.predictor=(Maia3Predictor if self.record['family']=='maia3' else Maia2Predictor)(device=device)
        self.policy=FrozenPolicy(self.predictor,self.record['family']).eval()
        self.group=None
        if self.record['status']=='TRAINED':
            self._load(self.record,self.policy)
        if self.record.get('group'):
            g=registry_record(registry,self.record['group'],access_role)
            self.group=FrozenPolicy(self.predictor,self.record['family']).eval(); self._load(g,self.group)
        self.model_hash=hashlib.sha256(json.dumps(dict(model=self.record,group=g if self.group else None),sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _load(record,policy):
        if digest(record['checkpoint'])!=record['checkpoint_sha256']: raise ValueError('Registry checkpoint hash mismatch')
        load_checkpoint(record['checkpoint'],policy.personal,record['metadata'])
        policy.eval()

    @torch.no_grad()
    def predict_many(self,contexts):
        encoded=[self.predictor.encode(c) for c in contexts]; family=self.record['family']
        token=torch.stack([e[0] for e in encoded]).to(self.device)
        se=torch.tensor([c.self_elo if family=='maia3' else e[3] for c,e in zip(contexts,encoded)],device=self.device)
        oe=torch.tensor([c.oppo_elo if family=='maia3' else e[4] for c,e in zip(contexts,encoded)],device=self.device)
        mask=torch.stack([e[1] for e in encoded]).to(self.device)
        h,_=self.policy.extract(token,se,oe)
        temp=self.record['temperature']; alpha=self.record.get('alpha',1.)
        probs=legal_log_probs(self.policy.policy(h),mask,temp).exp()
        if self.group: probs=alpha*probs+(1-alpha)*legal_log_probs(self.group.policy(h),mask,temp).exp()
        from .maia3_bridge import mirror_move
        result=[]
        for i,(context,e) in enumerate(zip(contexts,encoded)):
            ids=e[1].nonzero().flatten().tolist(); board=e[2]
            moves=[self.predictor.vocab[j] if board.turn else mirror_move(self.predictor.vocab[j]) for j in ids]
            result.append(distribution(board,moves,probs[i,ids].cpu().tolist(),self.model_id,self.model_hash,
                tensor_hash(e[0],e[1],context.self_elo if family=='maia3' else e[3],context.oppo_elo if family=='maia3' else e[4])))
        return result

    def predict(self,context): return self.predict_many([context])[0]
