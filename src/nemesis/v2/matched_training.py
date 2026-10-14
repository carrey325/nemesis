"""Fixed role allocation, source draws, and measured minimum-work admission."""
import hashlib
import math
import numpy as np
from nemesis.models.train import bootstrap_indices, rows_hash


def reference_reuse_audit(evidence):
    # B03 has no attributable model costs or equivalent stopping protocol.
    return dict(historical_match=False, reuse_compute_match=False,
                cold_equivalent=None, incremental_training_cost=0,
                reason='Historical per-role resources/stopping proof unavailable; original A00 read-only asset only',
                evidence=evidence, matched_reference_required=True)


def role_protocol(rows, player, seed):
    if any(r['split'] != 'fit_a' for r in rows):
        raise PermissionError('Challenge training accepts Fit-A only; no Fit-B/H')
    selected = [r for r in rows if player is None or r['player_id'] == player]
    if not selected:
        raise ValueError('Missing Fit-A role')
    ids = np.arange(len(selected)) if player is None else bootstrap_indices(selected, seed)
    draws = [selected[int(i)] for i in ids]
    # Trace every sampled game occurrence via existing B03 whole-game bootstrap.
    return dict(player=player, seed=seed, rows=len(ids), input_rows_sha256=rows_hash(selected),
                draws_sha256=rows_hash(draws), indices_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                game_draw_count=len({r['game_id'] for r in selected}),
                updates_per_epoch=max(1,len(ids)//256), dropped_tail=len(ids)%256,
                bootstrap='none_all32_equal_game_then_player' if player is None else 'whole_game_with_replacement')


def minimum_feasibility(roles, rates):
    results=[]
    for role in roles:
        rate=rates[role['method']]
        required=('update_cpu','update_gpu','validation_cpu','validation_gpu','switch_cpu','switch_gpu')
        if any(k not in rate or not math.isfinite(rate[k]) or rate[k] < 0 for k in required):
            raise ValueError('Invalid or missing measured training rate')
        minimum=min(100,role['updates_per_epoch'])
        cpu=1.25*(minimum*rate['update_cpu']+rate['validation_cpu']+rate['switch_cpu'])
        gpu=1.25*(minimum*rate['update_gpu']+rate['validation_gpu']+rate['switch_gpu'])
        results.append(dict(**role,minimum_updates=minimum,cpu_forecast_seconds=cpu,gpu_forecast_seconds=gpu,
            feasible=cpu <= .95*role['cpu_quota_seconds'] and gpu <= .95*role['gpu_quota_seconds']))
    return dict(feasible=all(r['feasible'] for r in results),roles=results,
                multiplier=1.25,reserve_fraction=.05,minimum_rule='100 updates or one full drop_last epoch, first reached')


def role_budgets(protocols):
    roles=[]
    for method in ('A00_MATCHED','C01'):
        for protocol in protocols:
            group=protocol['player'] is None
            roles.append(dict(**protocol,method=method,
                role='G' if group else f"T_{protocol['player']}_{protocol['seed']}",
                cpu_quota_seconds=1440. if group else 240.,
                gpu_quota_seconds=1260. if group else 210.))
    # CPU2h*.8/24 =240 seconds (not192); GPU1.75h*.8/24 =210.
    rng=np.random.default_rng(609)
    players=sorted({r['player'] for r in roles if r['player'] is not None},
                   key=lambda s:hashlib.sha256(s.encode()).hexdigest())
    order=['G']+[f'T_{p}_{int(seed)}' for seed in rng.permutation([1729,2718,3141]) for p in players]
    return dict(roles=roles,rotation_order=order,rotation_seed=609,
                shared_cpu_seconds=3600,shared_gpu_seconds=3600,shared_each_arm_fraction=.5,
                arm_exclusive_cpu_seconds=7200,arm_exclusive_gpu_seconds=6300,
                arm_inclusive_cpu_seconds=9000,arm_inclusive_gpu_seconds=8100)
