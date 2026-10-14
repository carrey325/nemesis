"""Real fixed-sample training cost measurements; no outcome-based selection."""
import copy
import json
from pathlib import Path
import time
import torch
from nemesis.data.common import digest, write_json
from nemesis.models.adapters import supervised_loss


def fit_fixture(policy, batch, optimizer, steps=10, *, start_step=0, fault_after=None,
                checkpoint_path=None, metadata=None):
    tokens,se,oe,mask,target,original=batch
    losses=[];b_first=0.;a_after=0.
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits=policy(tokens,se,oe)
        loss=supervised_loss(logits,original,mask,target)
        loss.backward()
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite LoRA fixture loss')
        for module in policy.personal:
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in module.parameters()):
                raise RuntimeError('Missing/nonfinite packed LoRA gradients')
            if step==0: b_first=max(b_first,module.bq.grad.abs().max().item(),module.bv.grad.abs().max().item())
            if step>=1: a_after=max(a_after,module.aq.grad.abs().max().item(),module.av.grad.abs().max().item())
        optimizer.step();losses.append(loss.item())
        if fault_after is not None and step+1==fault_after:
            from nemesis.models.adapters import save_checkpoint
            if checkpoint_path is None or metadata is None:
                raise ValueError('Fault boundary requires hash-bound checkpoint metadata')
            save_checkpoint(checkpoint_path,policy.personal,metadata,dict(optimizer=optimizer.state_dict(),
                step=start_step+step+1,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
            raise InterruptedError('Injected optimizer-boundary fixture interruption')
    with torch.no_grad(): final=supervised_loss(policy(tokens,se,oe),original,mask,target).item()
    return dict(initial_loss=losses[0],final_loss=final,losses=losses,
        b_first_gradient=b_first,a_after_b_gradient=a_after)


def timed(operation):
    torch.cuda.synchronize(); cpu=time.process_time(); wall=time.perf_counter()
    result=operation()
    torch.cuda.synchronize()
    return result,dict(cpu_seconds=time.process_time()-cpu,gpu_seconds=time.perf_counter()-wall)


def encode_batch(predictor,rows):
    from nemesis.models.common import context_from_row
    from nemesis.models.train import target_index
    contexts=[context_from_row(r) for r in rows]
    encoded=[predictor.encode(c) for c in contexts]
    return (torch.stack([e[0] for e in encoded]).to('cuda'),
        torch.tensor([c.self_elo for c in contexts],device='cuda'),
        torch.tensor([c.oppo_elo for c in contexts],device='cuda'),
        torch.stack([e[1] for e in encoded]).to('cuda'),
        torch.tensor([target_index(r,predictor.vocab,e[2].turn) for r,e in zip(rows,encoded)],device='cuda'))


def measured_sample(rows, out):
    """Cold encode, real forward/backward, validation, switch. Batch256/micro64.

    The baseline extraction is shared; no neural output is retained across
    personalized blocks. Report cached-prefix best-case and streamed costs.
    """
    from nemesis.models.maia3_bridge import Maia3Predictor
    from nemesis.models.adapters import FrozenPolicy
    from nemesis.models.lora_qv import QVPolicy
    from nemesis.models.train import rows_hash
    predictor,switch=timed(lambda:Maia3Predictor(device='cuda'))
    encoded,encoding=timed(lambda:encode_batch(predictor,rows))
    tokens,se,oe,mask,target=encoded
    reference=[]
    with torch.no_grad():
        for i in range(0,256,64): reference.append(predictor.model(tokens[i:i+64],se[i:i+64],oe[i:i+64])[0])
    original=torch.cat(reference).detach()
    results={}
    for method in ('A00_MATCHED','C01'):
        p=copy.deepcopy(predictor)
        policy=FrozenPolicy(p,'maia3') if method=='A00_MATCHED' else QVPolicy(p)
        optimizer=torch.optim.AdamW(policy.personal.parameters(),lr=.001,weight_decay=.0001)
        def update():
            optimizer.zero_grad(set_to_none=True)
            for i in range(0,256,64):
                if method=='A00_MATCHED': logits=policy(tokens[i:i+64],se[i:i+64],oe[i:i+64])
                else: logits=policy(tokens[i:i+64],se[i:i+64],oe[i:i+64])
                loss=supervised_loss(logits,original[i:i+64],mask[i:i+64],target[i:i+64])/4
                loss.backward()
            optimizer.step()
        update() # one charged warmup, no recipe tuning
        samples=[]
        for _ in range(5): _,cost=timed(update);samples.append(cost)
        def validate():
            with torch.no_grad():
                for i in range(0,256,64): policy(tokens[i:i+64],se[i:i+64],oe[i:i+64])
        _,validation=timed(validate)
        # Best-case prefix cache: hook frozen block5 output, then replay only
        # last two blocks, final norm, official from/to/promotion head.
        if method=='C01':
            prefix=[]
            hook=policy.base.transformer.layers[5].register_forward_hook(lambda m,ins,o:prefix.append(o.detach()))
            validate();hook.remove()
            def prefix_update():
                optimizer.zero_grad(set_to_none=True)
                for i,h in enumerate(prefix):
                    x=h
                    for block in policy.base.transformer.layers[6:]:x=block(x)
                    x=policy.base.transformer.norm(x)
                    # Reuse A00 official policy-head implementation with no residual.
                    head=FrozenPolicy.policy(policy,x,personal=False)
                    loss=supervised_loss(head,original[64*i:64*i+64],mask[64*i:64*i+64],target[64*i:64*i+64])/4
                    loss.backward()
                optimizer.step()
            costs=[]
            prefix_update()
            for _ in range(5): _,c=timed(prefix_update);costs.append(c)
            best_case=dict(cpu_seconds=max(c['cpu_seconds'] for c in costs),
                           gpu_seconds=max(c['gpu_seconds'] for c in costs))
        else:
            # Real final64-square A00 cached work, identical to existing B03 cache.
            h=[]
            for i in range(0,256,64): h.append(policy.extract(tokens[i:i+64],se[i:i+64],oe[i:i+64])[0])
            def cached_update():
                optimizer.zero_grad(set_to_none=True)
                for i,x in enumerate(h):
                    loss=supervised_loss(policy.policy(x),original[64*i:64*i+64],mask[64*i:64*i+64],target[64*i:64*i+64])/4
                    loss.backward()
                optimizer.step()
            costs=[];cached_update()
            for _ in range(5):_,c=timed(cached_update);costs.append(c)
            best_case=dict(cpu_seconds=max(c['cpu_seconds'] for c in costs),gpu_seconds=max(c['gpu_seconds'] for c in costs))
        results[method]=dict(stream_update_samples=samples,best_case_cached_update=best_case,
                            validation_batch256=validation,model_switch=switch)
        del policy,p,optimizer
    result=dict(sample_rows=256,rows_sha256=rows_hash(rows),encoding_batch256=encoding,
        methods=results,device='CUDA FP32',hardware=torch.cuda.get_device_name(),effective_batch=256,
        microbatch=64,training_recipe='AdamW .001 wd.0001 CE + .1KL',
        uses_H=False,uses_S_compare=False,base_sha256=predictor.record['sha256'])
    write_json(out,result)
    return result


def run_benchmark(out,lock,folder):
    from .matched_training import minimum_feasibility
    from .challenge_resources import build_ledger
    from .types import StageResult
    from .challenge import candidate_decision
    evidence_root=Path('artifacts/V2/B05A')
    shared=json.loads((evidence_root/'shared_forecast_probe.json').read_text())
    samples={scope:json.loads((evidence_root/f'engineering02/{scope}_engineering_benchmark.json').read_text())
             for scope in ('group','personal')}
    audit=[]
    for role in lock['role_budgets']['roles']:
        sample=samples['group' if role['player'] is None else 'personal']
        measured=sample['methods'][role['method']]
        ncal=role['calibration_rows']/256
        rate=dict(update_cpu=max(x['cpu_seconds'] for x in measured['stream_update_samples'])+sample['encoding_batch256']['cpu_seconds'],
            update_gpu=max(x['gpu_seconds'] for x in measured['stream_update_samples'])+sample['encoding_batch256']['gpu_seconds'],
            validation_cpu=ncal*(measured['validation_batch256']['cpu_seconds']+sample['encoding_batch256']['cpu_seconds']),
            validation_gpu=ncal*(measured['validation_batch256']['gpu_seconds']+sample['encoding_batch256']['gpu_seconds']),
            switch_cpu=measured['model_switch']['cpu_seconds'],switch_gpu=measured['model_switch']['gpu_seconds'])
        admission=minimum_feasibility([role],{role['method']:rate})['roles'][0]
        admission['measured_rates']=rate
        admission['cached_update_best_case']=measured['best_case_cached_update']
        audit.append(admission)
    ledger=build_ledger(out)
    engineering_cpu=ledger['stage_actual']['cpu_core_hours']
    engine=shared['engine_only_cpu_hours_1_25']
    shared_gate=engineering_cpu+engine<=.95*lock['role_budgets']['shared_cpu_seconds']/3600
    minimum=dict(role_minimum_admission=all(r['feasible'] for r in audit),roles=audit,
        shared_cpu_gate=shared_gate,shared_gpu_gate=None,
        required_new_engine_cpu_hours_pessimistic=engine,shared_cpu_cap_hours=1.,shared_cpu_start_gate_hours=.95,
        already_charged_shared_cpu_hours=engineering_cpu,
        shared_cpu_minimum_forecast_hours=engineering_cpu+engine,
        omitted_other_mandatory_costs=['neural predictions','TUNE C01 selection/evaluation','risk predictions on frozenD6',
                                      'calibration selection','paired bootstrap','report closure'],
        reason='Necessary engine comparison work alone fails fixed shared CPU admission; total full contract cannot be safely financed',
        feasible=all(r['feasible'] for r in audit) and shared_gate,
        forecast_multiplier=1.25,reserve_fraction=.05,no_borrowing=True,
        source_forecast_sha256=digest(evidence_root/'shared_forecast_probe.json'),
        benchmark_source_snapshots={str(p):digest(p) for p in sorted((evidence_root/'engineering02/source').glob('*.py'))},
        models_trained=0,engineering_fixture_not_training_result=True)
    write_json(folder/'minimum_forecast.json',minimum)
    if minimum['feasible']:
        # This run is a bounded admission closure. Never silently start an
        # unaudited training implementation or return a fake negative result.
        raise PermissionError('Budget closure condition absent; this candidate does not authorize unimplemented full challenge execution')
    reason='METHOD_BUDGET_INCONCLUSIVE: necessary S_compare engine work exceeds R4 shared CPU quota'
    for method,name in (('C01','candidate_registry.json'),('A00_MATCHED','matched_reference_registry.json')):
        models=[dict(method=method,model_id=r['role'],player=r['player'],seed=r['seed'],
                     checkpoint=None,model_hash=None,T=None,alpha=None,status='NOT_STARTED_BUDGET',
                     reason=reason,fit_draws_sha256=r['draws_sha256'])
                for r in lock['role_budgets']['roles'] if r['method']==method]
        write_json(out/name,dict(run_id=lock['run_id'],method=method,lock_sha256=digest(out/'challenge.lock.json'),
            registry_complete=False,models=models,models_trained=0,reason=reason))
    write_json(out/'fair_compute_audit.json',dict(status='METHOD_BUDGET_INCONCLUSIVE',fair_reference_complete=False,
        inherited_costs_not_reset=True,historical_match=False,reuse_compute_match=False,old_A00_read_only=True,
        data_game_draws_identical=True,per_arm_roles=25,three_seeds=True,recipe_count_each=1,
        planned_role_budgets_sha256=digest(out/'role_budgets.json'),
        minimum_forecast_sha256=digest(folder/'minimum_forecast.json'),
        shared_actual_once=ledger['stage_actual'],
        per_arm_shared_allocation={r:v*.5 for r,v in ledger['stage_actual'].items()},
        per_role_actual=None,reason=reason,no_effect_estimates=True))
    write_json(out/'candidate_decision.json',dict(decision=candidate_decision(fair=False),reason=reason,
        expansion_authorized=False,fair_comparison_complete=False,pooled_nll_delta=None,
        direct_D_E3=None,direct_D_E2=None,specificity_delta_E3=None,specificity_delta_E2=None,
        frequency_delta=None,not_a_method_effect_result=True,
        lock_sha256=digest(out/'challenge.lock.json'),forecast_sha256=digest(folder/'minimum_forecast.json')))
    outputs={str(p.relative_to(out)):digest(p) for p in (folder/'minimum_forecast.json',out/'fair_compute_audit.json',out/'candidate_decision.json')}
    write_json(folder/'phase_seal.json',dict(lock_sha256=digest(out/'challenge.lock.json'),run_id=lock['run_id'],
        research_status='METHOD_BUDGET_INCONCLUSIVE',registries_complete=False,
        registries={n:digest(out/n) for n in ('candidate_registry.json','matched_reference_registry.json')},outputs=outputs))
    return StageResult('B05A','DONE','METHOD_BUDGET_INCONCLUSIVE',out,0,reason)
