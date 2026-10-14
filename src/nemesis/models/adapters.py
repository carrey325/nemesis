"""Frozen upstream models with the M2 residual / M3 official policy head."""
import math
from pathlib import Path
import torch
from torch import nn

class ResidualAdapter(nn.Module):
    def __init__(self,dim,rank=16):
        super().__init__()
        self.norm=nn.LayerNorm(dim)
        self.down=nn.Linear(dim,rank)
        self.up=nn.Linear(rank,dim)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self,h):
        return h+self.up(torch.nn.functional.gelu(self.down(self.norm(h))))

def save_checkpoint(path,module,metadata,training=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    torch.save(dict(parameters=module.state_dict(),metadata=metadata,training=training),temp)
    temp.replace(path)

def load_checkpoint(path,module,expected):
    value=torch.load(path,map_location='cpu',weights_only=False)
    for key,v in expected.items():
        if value['metadata'].get(key)!=v: raise ValueError('Checkpoint lineage mismatch: '+key)
    module.load_state_dict(value['parameters'],strict=True)
    return value

def state_hash(model,exclude=()):
    import hashlib
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        if any(name.startswith(p) for p in exclude): continue
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class FrozenPolicy(nn.Module):
    def __init__(self,predictor,family):
        super().__init__(); self.base=predictor.model; self.family=family
        self.base.requires_grad_(False); self.base.eval()
        if family=='maia3':
            self.personal=ResidualAdapter(predictor.cfg.dim_vit).to(predictor.device)
        elif family=='maia2':
            import copy
            self.personal=copy.deepcopy(self.base.fc_1)
            self.personal.requires_grad_(True)
        else: raise ValueError('Unknown family')

    def train(self,mode=True):
        super().train(mode); self.base.eval(); self.personal.train(mode)
        return self

    def extract(self,tokens,se,oe):
        result=[]
        if self.family=='maia3':
            hook=self.base.transformer.register_forward_hook(lambda m,i,o: result.append(o.detach()))
        else:
            hook=self.base.fc_1.register_forward_pre_hook(lambda m,i: result.append(i[0].detach()))
        try:
            with torch.no_grad(): original=self.base(tokens,se,oe)[0]
        finally: hook.remove()
        return result[0],original

    def policy(self,h,personal=True):
        if self.family=='maia2': return (self.personal if personal else self.base.fc_1)(h)
        if personal: h=self.personal(h)
        frm=self.base.proj_sq_from(h[:,:64]); to=self.base.proj_sq_to(h[:,:64])
        scores=torch.einsum('bid,bjd->bij',frm,to)/math.sqrt(self.base.cfg.head_hid_dim)
        bias=self.base.promo_bias_proj(to[:,56:64])*math.sqrt(self.base.cfg.head_hid_dim)
        promo=scores[:,48:56,56:64].unsqueeze(-1)+bias.unsqueeze(1)
        return torch.cat((scores.flatten(1),promo.flatten(1)),1)

    def forward(self,tokens,se,oe):
        h,_=self.extract(tokens,se,oe)
        return self.policy(h)

def legal_log_probs(logits,mask,temperature=1.):
    if not mask.any(-1).all(): raise ValueError('Empty legal policy')
    return (logits.float()/temperature).masked_fill(~mask.bool(),-torch.inf).log_softmax(-1)

def supervised_loss(logits,base_logits,mask,targets,weights=None):
    logp=legal_log_probs(logits,mask); logq=legal_log_probs(base_logits,mask)
    if not mask[torch.arange(len(targets),device=targets.device),targets].all():
        raise ValueError('Human target is illegal')
    nll=-logp.gather(1,targets[:,None]).squeeze(1)
    # Avoid 0*(-inf - -inf) on illegal vocabulary entries.
    difference=logp.masked_fill(~mask,0)-logq.masked_fill(~mask,0)
    kl=(logp.exp()*difference).sum(-1)
    loss=nll+.1*kl
    return loss.mean() if weights is None else (loss*weights).sum()
