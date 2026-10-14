"""Append-only B05A measured receipts plus one conservative outside allowance."""
import json
from pathlib import Path
import time
from nemesis.data.common import digest, write_json
from .resources import reconcile_balances


def build_ledger(out):
    out=Path(out)
    root=Path('artifacts/V2/B05A')
    previous=Path('artifacts/V2/B04B/resource_ledger.json')
    prior=json.loads(previous.read_text())
    units=[]
    receipt_paths=[root/'test_resource_receipts.jsonl',root/'phase_resource_receipts.jsonl']
    if out.resolve()!=root.resolve():receipt_paths.append(out/'phase_resource_receipts.jsonl')
    for path in receipt_paths:
        if path.exists(): units.extend(json.loads(line) for line in path.read_text().splitlines())
    for path in (root/'engineering_benchmark.cost.json',root/'engineering02/engineering_benchmark.cost.json'):
        if path.exists():
            unit=json.loads(path.read_text())
            if path.parent.name=='engineering02':unit['work_unit_id']='B05A-engineering-benchmark-repaired'
            units.append(unit)
    ids=[u['work_unit_id'] for u in units]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate B05A cost work unit')
    cpu=sum(u['cpu_seconds'] for u in units)
    gpu=sum(u['gpu_seconds'] for u in units)
    allowance=dict(work_unit_id='B05A-N01-outside-implementation-shell-red-tests-git-report',owner_stage='B05A',
        cpu_seconds=1200.,gpu_seconds=0.,reason='Conservative allowance for shell, imports before receipt hooks, initial unmetered unit red/green tests, state-only probe, hashing/snapshot/report/commit/push and coordination; measured phase/test CPU charged additionally')
    stage=dict(cpu_core_hours=(cpu+allowance['cpu_seconds'])/3600,gpu_hours=gpu/3600)
    inherited=prior['project_actual']
    caps=dict(stage_cpu_core_hours=5,stage_gpu_hours=4.5,project_cpu_core_hours=52,project_gpu_hours=20,
              disk_gib=80,external_spend_usd=0)
    ledger=dict(stage='B05A',attempt_id='B05A-N01',inherited_actual=inherited,
        inherited_ledger_sha256=digest(previous),inherited_ledger_path=str(previous),
        inherited_work_unit_count=len(prior['work_units']),prior_allowances_preserved_in_hashed_parent=True,
        A03_audit_overlay_recharged=False,stage_actual=stage,project_actual={},caps=caps,remaining={},
        work_units=units,allowances=[allowance],measurement_timestamps=dict(closed_unix=time.time()),
        measurement_method='All-thread process_time per test/phase; each process charged once; full CUDA test process or engineering residency; no engine subprocess launched; parent CPU phases never assigned GPU residency',
        per_role_cost=dict(historical_training=None,new_training=None,reason='No fair model training started; only engineering fixtures and method benchmark'),
        cold_warm=dict(historical_cold_equivalent=None,reason='Historical role compute remains unattributed',
            incremental_cpu_seconds=cpu,incremental_gpu_seconds=gpu),
        disk=dict(inherited_upper_bound_gib=prior['disk']['peak_upper_bound_gib'],
                  new_stage_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
                  peak_upper_bound_gib=prior['disk']['peak_upper_bound_gib']+1.,
                  reason='No full input/prefix or engine cache built; <=1GiB conservative stage evidence/checkpoint/temporary allowance; prior bound retained'),
        external_spend_usd=0)
    return reconcile_balances(ledger)


def save_ledger(out):
    ledger=build_ledger(out)
    write_json(Path(out)/'resource_ledger.json',ledger)
    return ledger
