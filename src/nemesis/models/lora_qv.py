"""C01: only packed Q/V rank-four increments in locked blocks six and seven."""
import math
import torch
from torch import nn
from torch.nn.utils import parametrize


class PackedQV(nn.Module):
    def __init__(self, dim, rank=4):
        super().__init__()
        if dim != 512 or rank != 4:
            raise ValueError('C01 requires dim512/rank4/scale1')
        self.aq = nn.Parameter(torch.empty(rank, dim))
        self.av = nn.Parameter(torch.empty(rank, dim))
        self.bq = nn.Parameter(torch.zeros(dim, rank))
        self.bv = nn.Parameter(torch.zeros(dim, rank))
        nn.init.kaiming_uniform_(self.aq, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.av, a=math.sqrt(5))

    def forward(self, original):
        q = self.bq @ self.aq
        v = self.bv @ self.av
        return original + torch.cat((q, torch.zeros_like(q), v))


class QVPolicy(nn.Module):
    """Preserve upstream full forward and GAB. Frozen prefix creates no graph.

    No A00 residual is installed. Parametrization is registered before optimizer.
    Both eval fastpath and grad forward read the effective packed tensor.
    """
    family = 'maia3'
    def __init__(self, predictor):
        super().__init__()
        self.base = predictor.model
        if predictor.cfg.dim_vit != 512 or len(self.base.transformer.layers) != 8:
            raise ValueError('METHOD_CONFLICT: locked backbone is not eight512 blocks')
        self.base.requires_grad_(False).eval()
        self.personal = nn.ModuleList()
        for index in (6, 7):
            mha = self.base.transformer.layers[index].self_attn.mha
            if parametrize.is_parametrized(mha) or tuple(mha.in_proj_weight.shape) != (1536,512):
                raise ValueError('METHOD_CONFLICT: unexpected packed QKV')
            update = PackedQV(512).to(predictor.device)
            parametrize.register_parametrization(mha, 'in_proj_weight', update)
            mha.parametrizations.in_proj_weight.original.requires_grad_(False)
            self.personal.append(update)
        self.base.eval()
        if sum(p.numel() for p in self.base.parameters() if p.requires_grad) != 16384:
            raise ValueError('C01 trainable set mismatch')

    def train(self, mode=True):
        # Dropout remains off; autograd is independently enabled on Q/V.
        self.base.eval()
        return self

    def forward(self, tokens, se, oe):
        return self.base(tokens,se,oe)[0]

    def frozen_hash(self):
        from .adapters import state_hash
        # Original packed matrices are included; only four low-rank tensors excluded.
        return state_hash(self.base, exclude=tuple(
            f'transformer.layers.{i}.self_attn.mha.parametrizations.in_proj_weight.0.'
            for i in (6,7)))
