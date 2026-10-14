"""B04A cumulative ledger generation and fail-closed downstream consumption."""
import copy
import json
import math
from pathlib import Path

from nemesis.data.common import digest

REFERENCE_NAME = 'authoritative_resource_ledger.json'
RESOURCES = ('cpu_core_hours', 'gpu_hours')


def reconcile_balances(ledger):
    """Derive totals and balances after all charges, never from older balances."""
    for resource in RESOURCES:
        ledger['project_actual'][resource] = (ledger['inherited_actual'][resource]
                                               + ledger['stage_actual'][resource])
        for scope in ('stage', 'project'):
            key = f'{scope}_{resource}'
            ledger['remaining'][key] = ledger['caps'][key] - ledger[f'{scope}_actual'][resource]
    validate_ledger(ledger)
    return ledger


def load_a03_inherited(root=Path('artifacts/V2/B04A'), *, overlay_path=Path('artifacts/V2/A03/resource_overlay.json')):
    """Validate historical authority first, then consume the authorized overlay once."""
    prior=load_authoritative_ledger(root)
    receipt=json.loads(Path('artifacts/V2/A03/adoption_receipt.json').read_text(encoding='utf8'))
    if (digest(overlay_path)!=receipt['resource_overlay_sha256']
        or receipt['resource_overlay_sha256']!='a26fdcbe099534708472d850e4d384e152b9af1e55823d9dbe131f449e10b62b'
        or receipt['old_plan_version']!='sha256:592b95df1e6414dbd3fbbe40fb280ccf02ecd2b4c10912d80c06dc016e9118f4'
        or receipt['new_plan_version']!='sha256:f2ffc68dffe1a238cd284a58b96801eaaf89332afea54fa3bdda98bf4d188a7d'):
        raise ValueError('A03 overlay hash mismatch')
    overlay=json.loads(Path(overlay_path).read_text(encoding='utf8'))
    if (overlay['base_ledger_sha256']!=digest(Path(root)/'attempt03/resource_ledger.json')
        or overlay['base_accepted_candidate']!='a09984926f28bde5f8e0758d87db56f58550f02b'
        or overlay['action_id']!=receipt['action_id']):
        raise ValueError('A03 prior ledger/authorization mismatch')
    ids=[c['work_unit_id'] for c in overlay['charges']]
    historical={c['work_unit_id'] for c in prior['work_units']+prior['allowances']}
    if ids!=['b04b-budget-audit-20261010'] or historical.intersection(ids):
        raise ValueError('Duplicate/unknown A03 charge')
    actual={r:prior['project_actual'][r]+sum(c[r] for c in overlay['charges']) for r in RESOURCES}
    if actual!=overlay['project_actual'] or overlay['charges'][0]['cpu_core_hours']!=.05 or overlay['charges'][0]['gpu_hours']!=0:
        raise ValueError('A03 overlay arithmetic differs')
    return dict(project_actual=actual,charge_ids=ids,base_ledger_sha256=overlay['base_ledger_sha256'],
        overlay_sha256=digest(overlay_path),caps=overlay['caps'])


def validate_ledger(ledger):
    for resource in RESOURCES:
        inherited = ledger['inherited_actual'][resource]
        stage = ledger['stage_actual'][resource]
        project = ledger['project_actual'][resource]
        if not all(math.isfinite(x) and x >= 0 for x in (inherited, stage, project)):
            raise ValueError('Invalid resource actual')
        if project != inherited + stage:
            raise ValueError('Project actual must equal inherited plus stage actual')
        for scope in ('stage', 'project'):
            key = f'{scope}_{resource}'
            cap = ledger['caps'][key]
            actual = ledger[f'{scope}_actual'][resource]
            if not math.isfinite(cap) or cap < actual:
                raise ValueError('Resource cap exceeded or invalid')
            if ledger['remaining'][key] != cap - actual:
                raise ValueError('Stale remaining resource balance')


def append_correction(prior_path, charge):
    """Retain every historical charge; append one uniquely identified allowance."""
    prior_path = Path(prior_path)
    prior = json.loads(prior_path.read_text())
    ledger = copy.deepcopy(prior)
    ids = {c['work_unit_id'] for c in prior['work_units'] + prior['allowances']}
    if charge['work_unit_id'] in ids:
        raise ValueError('Duplicate resource charge')
    for seconds in ('cpu_seconds', 'gpu_seconds'):
        if not math.isfinite(charge[seconds]) or charge[seconds] < 0:
            raise ValueError('Invalid resource charge')
    ledger['allowances'].append(copy.deepcopy(charge))
    increment = dict(cpu_core_hours=charge['cpu_seconds']/3600, gpu_hours=charge['gpu_seconds']/3600)
    for resource in RESOURCES:
        ledger['stage_actual'][resource] += increment[resource]
    ledger['attempt_id'] = 'B04A-N03'
    ledger['prior_ledger_sha256'] = digest(prior_path)
    ledger['correction_increment'] = increment
    return reconcile_balances(ledger)


def load_authoritative_ledger(root):
    """Require the bound reference; missing/tampered/stale evidence has no fallback."""
    root = Path(root).resolve()
    reference = json.loads((root/REFERENCE_NAME).read_text())
    paths = {}
    for key in ('ledger', 'prior_ledger'):
        record = reference[key]
        path = (root/record['path']).resolve()
        if not path.is_relative_to(root) or digest(path) != record['sha256']:
            raise ValueError('Resource ledger reference hash/path mismatch')
        paths[key] = path
    ledger = json.loads(paths['ledger'].read_text())
    prior = json.loads(paths['prior_ledger'].read_text())
    validate_ledger(ledger)
    if (ledger['attempt_id'] != reference['attempt_id']
            or ledger['stage'] != 'B04A'
            or ledger['prior_ledger_sha256'] != reference['prior_ledger']['sha256']
            or ledger['caps'] != prior['caps']
            or ledger['inherited_actual'] != prior['inherited_actual']
            or ledger['inherited_ledger_sha256'] != prior['inherited_ledger_sha256']
            or ledger['work_units'] != prior['work_units']
            or ledger['allowances'][:-1] != prior['allowances']):
        raise ValueError('Resource ledger history/binding mismatch')
    charge = ledger['allowances'][-1]
    for resource, seconds in zip(RESOURCES, ('cpu_seconds', 'gpu_seconds')):
        delta = charge[seconds]/3600
        if delta < 0 or ledger['stage_actual'][resource] != prior['stage_actual'][resource]+delta:
            raise ValueError('Resource ledger charge mismatch')
        if ledger['correction_increment'][resource] != delta:
            raise ValueError('Resource ledger increment mismatch')
    return ledger
