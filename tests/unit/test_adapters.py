"""Catch missing residual, accidental backbone unfreeze and broken save/load."""
import torch
import pytest

def api():
    from nemesis.models import adapters
    return adapters

def test_zero_adapter_restores_original_logits():
    a=api().ResidualAdapter(512)
    h=torch.randn(3,64,512)
    assert torch.equal(a(h),h)

def test_only_declared_parameters_update():
    a=api().ResidualAdapter(32)
    x=torch.randn(4,64,32)
    a(x).square().mean().backward()
    assert a.up.weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in a.parameters())

def test_checkpoint_resume_preserves_metadata(tmp_path):
    a=api().ResidualAdapter(32)
    path=tmp_path/'a.pt'; meta={'base_sha':'abc','split_sha':'def','seed':1729,'config_sha':'ghi','model_id':'T_P006_1729'}
    api().save_checkpoint(path,a,meta)
    b=api().ResidualAdapter(32)
    api().load_checkpoint(path,b,meta)
    x=torch.randn(1,64,32)
    assert torch.equal(a(x),b(x))
    for key in ('base_sha','split_sha','model_id','config_sha'):
        with pytest.raises(ValueError): api().load_checkpoint(path,b,{**meta,key:'wrong'})
