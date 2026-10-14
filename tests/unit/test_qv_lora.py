import torch
from torch import nn


def test_qv_lora_updates_only_declared_weights():
    from nemesis.models.lora_qv import PackedQV
    torch.manual_seed(1729)
    layer = PackedQV(512)
    base = torch.randn(1536, 512)
    before = base.clone()
    assert sum(p.numel() for p in layer.parameters()) * 2 == 16384
    assert torch.equal(layer(base), base)
    x = torch.randn(8, 512)
    opt = torch.optim.AdamW(layer.parameters(), lr=.001)
    for step in range(2):
        opt.zero_grad()
        y = nn.functional.linear(x, layer(base))
        y.square().mean().backward()
        assert layer.bq.grad.abs().sum() > 0
        if step == 1:
            assert layer.aq.grad.abs().sum() > 0
            assert layer.av.grad.abs().sum() > 0
        opt.step()
    updated = layer(base)
    assert torch.equal(updated[512:1024], base[512:1024])
    assert torch.equal(base, before)
    assert not torch.equal(updated, base)


def test_lora_rejects_other_structure():
    import pytest
    from nemesis.models.lora_qv import PackedQV
    with pytest.raises(ValueError):
        PackedQV(512, rank=8)


def test_qv_declares_official_policy_head_family():
    from nemesis.models.lora_qv import QVPolicy
    assert QVPolicy.family == 'maia3'
