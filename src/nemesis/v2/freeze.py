"""Lightweight B05B: consume accepted metadata, never load models or H outcomes.

The development route decision and resource admission are separate findings.
Missing forecasts fail closed; an extrapolated scenario is not a measurement.
"""
import hashlib
import json
import math
from pathlib import Path
import time
import yaml

from .types import StageResult
from .guards import require_new_output
from .resources import validate_ledger, reconcile_balances

PLAN = 'sha256:f2ffc68dffe1a238cd284a58b96801eaaf89332afea54fa3bdda98bf4d188a7d'
BASE = '112a5defe33be6bd1da42a22cc1ee3651c6b0528'
B04B = 'ebde118057f8760f7b9de4583b90e469818170af'
MANIFESTS = {
    'B04B': '9d4bdee46ace35f92d8715f1c84e10cc847f97ffa4aab0cef16ae7b323aa3449',
    'B05A/attempt02': 'bf702e83336282819979fd2aa749047976b437afaf72005a8a4a4ab5845f1ca1',
}
INPUTS = {
    'pilot': 'B04B/pilot_decision.json',
    'bootstrap': 'B04B/bootstrap.json',
    'population': 'B04B/population_yield.json',
    'risk': 'B04B/risk_exact_cost.json',
    'cost': 'B04B/cost_forecast.json',
    'lambda': 'B04B/lambda_selection.json',
    'group_base_diagnostics': 'B04B/group_base_evaluator_diagnostics.json',
    'windows': 'B04B/per_player_windows.json',
    'run_binding': 'B04B/run_binding.json',
    'challenger': 'B05A/attempt02/candidate_decision.json',
    'ledger': 'B05A/attempt02/resource_ledger.json',
    'challenge_minimum': 'B05A/attempt02/train/minimum_forecast.json',
    'candidate_registry': 'B05A/attempt02/candidate_registry.json',
    'reference_registry': 'B05A/attempt02/matched_reference_registry.json',
}
NON_EXPANSION = {'STOP_CURRENT_ROUTE', 'GENERIC_ONLY_NO_EXPANSION',
    'INCONCLUSIVE_NO_AUTO_EXPANSION', 'BUDGET_BLOCKED', 'INSUFFICIENT_EVIDENCE'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


def read_metadata(root, relative, receipt):
    """Allowlist before resolve/open; no user-provided outcome path is accepted."""
    allowed = set(INPUTS.values()) | {p+'/artifact_manifest.json' for p in MANIFESTS} | {'B05A/canonical_attempt.json', 'B04A/cost_forecast.json'}
    if relative not in allowed:
        raise PermissionError('B05B metadata allowlist rejects: '+relative)
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root):
        raise PermissionError('Evidence path escapes stage root')
    raw = path.read_bytes()
    if len(raw) > 2*1024*1024:
        raise ValueError('B05B metadata size bound exceeded')
    receipt.append(dict(path=relative, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
        actor='B05B-freeze', purpose='accepted development metadata/cost only', time_unix=time.time()))
    return json.loads(raw.decode('utf8'))


def load_accepted_evidence(root):
    receipt = []
    indexes = {}
    for stage, expected in MANIFESTS.items():
        manifest = read_metadata(root, stage+'/artifact_manifest.json', receipt)
        if receipt[-1]['sha256'] != expected or manifest['plan_version'] != PLAN:
            raise ValueError('Accepted manifest hash/plan differs: '+stage)
        # B05A paths are stage-root-relative, including attempt02, unlike B04B.
        indexes[stage] = {r['path'].replace('\\', '/'): r['sha256'] for r in manifest['files']}
    canonical = read_metadata(root, 'B05A/canonical_attempt.json', receipt)
    if (canonical['canonical_output'] != 'attempt02' or canonical['plan_version'] != PLAN
            or canonical['lock_sha256'] != '2285308da6e8a4f2d4cd34f49fdae112d34ecea3b39670b136d24e2312fe78b4'):
        raise ValueError('B05A canonical attempt differs')
    data = {}
    for name, relative in INPUTS.items():
        stage = 'B04B' if relative.startswith('B04B/') else 'B05A/attempt02'
        item = read_metadata(root, relative, receipt)
        key = relative.split('/', 1)[1]
        if stage == 'B04B':
            key = relative[len('B04B/'):]
        if indexes[stage].get(key) != receipt[-1]['sha256']:
            raise ValueError('Accepted input hash differs: '+relative)
        data[name] = item
    if sha(Path(root)/'B05A/attempt02/candidate_decision.json') != canonical['decision_sha256']:
        raise ValueError('Canonical decision hash mismatch')
    data['COST32'] = read_metadata(root, 'B04A/cost_forecast.json', receipt)
    if (receipt[-1]['sha256'] != data['cost']['source_cost_sha256']
            or data['COST32']['cost_roots_measured'] != 32
            or data['COST32']['complete_COST32'] is not True):
        raise ValueError('COST32 source lineage differs')
    # Fixed read-only registry metadata, not checkpoints or sample labels.
    # The accepted B04B run binds B03R's asset map, which binds legacy registry.
    asset_path = Path(root)/'B03R/approved_assets.json'
    if sha(asset_path) != data['run_binding']['assets_sha256']:
        raise ValueError('Accepted asset map hash mismatch')
    assets = json.loads(asset_path.read_text(encoding='utf8'))
    registry_path = Path(root).parent/'B03/model_registry.json'
    if sha(registry_path) != assets['registry_hash']:
        raise ValueError('Accepted registry hash mismatch')
    registry = json.loads(registry_path.read_text(encoding='utf8'))
    if registry.get('confirmation_H_opened') is not False:
        raise ValueError('Registry holdout seal missing')
    confirmation = [r for r in registry['models'] if r['cohort'] == 'confirm']
    if len(confirmation) != 120 or any(r['status'] != 'NOT_TRAINED' for r in confirmation):
        raise ValueError('Confirmation models must remain NOT_TRAINED')
    data['registry'] = registry
    for path in (asset_path, registry_path):
        receipt.append(dict(path=path.as_posix(), sha256=sha(path), bytes=path.stat().st_size,
            actor='B05B-freeze', purpose='read-only accepted registry metadata; no model loads', time_unix=time.time()))
    validate_ledger(data['ledger'])
    if data['ledger']['project_actual'] != dict(cpu_core_hours=15.553885904153187, gpu_hours=4.979192232880852):
        raise ValueError('Nonmonotonic/unexpected inherited ledger')
    if (data['challenger']['decision'] != 'METHOD_BUDGET_INCONCLUSIVE'
            or data['challenger']['fair_comparison_complete'] is not False
            or any(data['challenger'][k] is not None for k in ('pooled_nll_delta', 'direct_D_E3', 'direct_D_E2'))):
        raise ValueError('This B05B closure requires accepted inconclusive challenger')
    return data, receipt


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def decide_route(pilot, intervals, *, budget):
    gates = pilot.get('reference_gates', {})
    required = ('gain_specificity', 'positive_targets', 'windows_nonnegative', 'frequency',
        'safety', 'qualification', 'unique_opportunity_games', 'budget')
    if any(type(gates.get(k)) is not bool for k in required):
        return 'INSUFFICIENT_EVIDENCE'
    if not gates['qualification'] or not gates['safety']:
        return 'INSUFFICIENT_EVIDENCE'
    try:
        values = [pilot['summaries'][f][m] for f in ('E3', 'E2') for m in ('G', 'S')]
        if not all(finite(v) for v in values):
            return 'INSUFFICIENT_EVIDENCE'
        positive = all(sum(finite(v) and v > 0 for v in pilot['summaries'][f]['per_target_S']) >= 4
            and len(pilot['summaries'][f]['per_target_S']) == 8 for f in ('E3', 'E2'))
    except (KeyError, TypeError):
        return 'INSUFFICIENT_EVIDENCE'
    if all(gates.values()) and min(values) >= .005 and positive:
        return 'EXPAND_CONFIRMATION' if budget is True else 'BUDGET_BLOCKED'
    bounds = [intervals.get(f+'_'+m, {}).get('ci95') for f in ('E3', 'E2') for m in ('G', 'S')]
    valid = all(isinstance(b, list) and len(b) == 2 and all(finite(x) for x in b)
        and b[0] <= b[1] for b in bounds)
    if valid and all(b[1] < .005 for b in bounds):
        return 'STOP_CURRENT_ROUTE'
    return 'INCONCLUSIVE_NO_AUTO_EXPANSION'


def choose_sample_size(half480, half960, *, feasible_960, holdout_opened=False,
                       development_in_confirmation=False):
    if holdout_opened:
        raise PermissionError('Sample choice must precede confirmation holdout access')
    if development_in_confirmation:
        raise ValueError('Development identities cannot enter confirmation statistics')
    if not all(finite(v) and v >= 0 for v in (half480, half960)):
        raise ValueError('Missing development precision forecast')
    selected = 960 if half480 > .01 and half960 < half480 and feasible_960 is True else 480
    return dict(S_confirm=selected, N_confirm=960,
        precision='PRECISION_LIMITED' if (half960 if selected == 960 else half480) > .01 else 'DESCRIPTIVE_PROJECTION_ONLY',
        projected_half480=half480, projected_half960=half960, confirmation_H_opened=False,
        development_in_confirmation=False, guarantee_80_percent_power=False)


def budget_admits(cost, stage_cap, project_remaining):
    return finite(cost) and cost >= 0 and cost <= .95*stage_cap and cost <= .95*project_remaining


def forecast_confirmation(cost, risk, inherited, *, s_confirm=480):
    if s_confirm not in (480, 960) or len(cost['per_root_nodes']) not in (32, 320) or len(risk['legal_moves']) != 1600:
        raise ValueError('Forecast requires complete development profiles and frozen1600 risk counts')
    rate = cost['seconds_per_node']
    if not finite(rate) or rate <= 0:
        raise ValueError('Invalid measured CPU rate')
    roots = cost['per_root_nodes']
    mean_decision = sum(r['decision'] for r in roots)/len(roots)
    mean_audit = sum(r['audit'] for r in roots)/len(roots)
    max_decision = max(r['decision'] for r in roots)
    # 10% sensitivity at double audit nodes; no neural/statistical costs assumed zero.
    matrix_nodes = s_confirm*(mean_decision+mean_audit*1.2)
    risk_nodes = risk['nodes']*4800/1600
    screen_nodes = 960*(100000+5*50000)
    arena_nodes = 128*200*max_decision
    to_hours = lambda nodes: nodes*rate*1.25/3600
    missing = ['B06A: full72T/24E3/24E2 encoding/training/Calibration/model switching; E2 timing absent',
        'B06B: confirmation-specific branching/screening acceptance, neural H/risk/matrix inference,20000 bootstrap,report',
        'B07: guarded full-game pilot throughput, neural replies, robustness/audits and censored-game reporting',
        'Guarded CUDA resident-wall time and resident-memory/peak-disk forecasts for all downstream units',
        'B08 final reporting cost remains separately capped atCPU.5/GPU.1 with5%reserve']
    b06b = to_hours(matrix_nodes+risk_nodes+screen_nodes)
    b07 = to_hours(arena_nodes)
    return dict(status='BUDGET_BLOCKED_NOT_ADMITTED', admitted=False, multiplier=1.25, reserve_fraction=.05,
        counts=dict(models=dict(T=72, E3=24, E2=24, total=120), risk_positions=4800,
            N_confirm=960, S_confirm=s_confirm, matrix_cells_per_family=24*24,
            evaluator_families=2, arena_games=128, arena_max_ply=200, arena_max_plies=25600),
        stages={
            'B06A': dict(cpu_core_hours=None, gpu_hours=None, cap_cpu=9, cap_gpu=6,
                status='UNMEASURED_FULL_CONTRACT', reason=missing[0]),
            'B06B': dict(cpu_core_hours=None, gpu_hours=None, partial_cpu_core_hours=b06b,
                stage_start_gate_cpu=4.75, cap_cpu=5, cap_gpu=1,
                partial_within_stage_gate=budget_admits(b06b, 5, 52-inherited['cpu_core_hours'])),
            'B07': dict(cpu_core_hours=None, gpu_hours=None, partial_cpu_core_hours=b07,
                stage_start_gate_cpu=9.5, cap_cpu=10, cap_gpu=1.5,
                partial_within_stage_gate=budget_admits(b07, 10, 52-inherited['cpu_core_hours'])),
            'B08': dict(cpu_core_hours=None, gpu_hours=None, cap_cpu=.5, cap_gpu=.1,
                stage_start_gate_cpu=.475, stage_start_gate_gpu=.095, status='SEPARATE_REVIEW_AND_ADMISSION_REQUIRED')},
        components=dict(risk_nodes=risk_nodes, risk_cpu_core_hours=to_hours(risk_nodes),
            matrix_nodes=matrix_nodes, matrix_cpu_core_hours=to_hours(matrix_nodes),
            minimum_natural_screen_nodes=screen_nodes, natural_screen_cpu_core_hours=to_hours(screen_nodes),
            arena_max_length_scenario_nodes=arena_nodes, arena_cpu_core_hours=to_hours(arena_nodes)),
        measured_sources=dict(COST_roots=32, development_profile_roots=len(roots), requested_dev_risk_nodes=risk['nodes'],
            measured_risk_positions=1600, seconds_per_node=rate,
            mean_development_decision_nodes=mean_decision, mean_development_audit_nodes=mean_audit,
            max_development_decision_nodes=max_decision, mean_risk_legal_moves=risk['total']/1600),
        partial_project_cpu_core_hours=inherited['cpu_core_hours']+b06b+b07,
        total_cpu_core_hours=None, total_gpu_hours=None, peak_disk_gib=None,
        system_resident_peak_gib=None, external_spend_usd=0, missing_costs=missing,
        assumptions=['Forecast scenarios extrapolate measured development branching; not confirmation measurements or physical lower bounds',
            'CPU rate derived from32 COST roots plus A03 measured two-process factor;320 development S_dev/TUNE branching profiles are distinct',
            'All legal replies retained; candidate-outside-set SF comparison and10%400k audits inherited in development profile',
            'Screening is a minimum960-root scenario; conditional panel rejection costs unknown, up to50000 candidates per panel',
            '128 games at200ply is a max-length scenario, not observed expected game length',
            'No CPU-core-hour discount from throttling or two processes; unmeasured GPU resident-wall may increase',
            '1.25 applied exactly once to raw node scenarios;5% stage reserve is separate; no quota borrowing'],
        inherited_A01_warning_preserved=True, OFFICE_PRIORITY='COMPUTE_GUARD_PENDING')


def make_method_lock(decision, choices):
    if decision in NON_EXPANSION:
        return None
    if decision != 'EXPAND_CONFIRMATION':
        raise ValueError('Unknown route decision')
    return dict(kind='METHOD_ONLY_REQUIRES_INDEPENDENT_REVIEW', choices=choices,
        confirmation_models=dict(status='NOT_TRAINED', count=120), artifact_lock=None,
        confirmation_H_access='DENIED_UNTIL_BOTH_LOCKS_INDEPENDENTLY_ACCEPTED')


def downstream_disposition(decision, decision_sha):
    if decision not in NON_EXPANSION | {'EXPAND_CONFIRMATION'}:
        raise ValueError('Unknown route decision')
    if len(decision_sha) != 64 or any(c not in '0123456789abcdef' for c in decision_sha):
        raise ValueError('Decision hash required')
    return {stage: dict(status='NOT_RUN', reason=decision+'; exact B05B independent review required',
        decision_sha256=decision_sha, effect=None, dispatch_authorized=False) for stage in ('B06A', 'B06B', 'B07')}


def freeze_program(config: Path, evidence: Path, out: Path) -> StageResult:
    start_cpu = time.process_time()
    start = time.time()
    require_new_output(out)
    if Path(out).exists():
        raise FileExistsError('Refuse overwrite: '+str(out))
    from .assets import verify_plan_config
    if verify_plan_config(config) != PLAN:
        raise ValueError('Exact approved B05B plan required')
    cfg = yaml.safe_load(Path(config).read_text(encoding='utf8'))
    data, reads = load_accepted_evidence(evidence)
    reviews = []
    for stage, commit in [('B04B', B04B), ('B05A', BASE)]:
        path = Path('workflow/reviews')/('lw-nemesis-stage1-'+stage+'-R01')/'response.md'
        text = path.read_text(encoding='utf8')
        for marker in ['COMMIT_SHA: '+commit, 'PLAN_VERSION: '+PLAN, 'DECISION: APPROVED', 'EVIDENCE_STATUS: SUFFICIENT']:
            if marker not in text:
                raise PermissionError('Exact upstream independent review missing: '+stage)
        reviews.append(dict(stage=stage, commit=commit, path=str(path), sha256=sha(path), text=text))
    ci = data['bootstrap']['results']['same_ID_block']
    forecast = forecast_confirmation(data['cost'], data['risk'], data['ledger']['project_actual'])
    forecast['alternative960'] = forecast_confirmation(data['cost'], data['risk'], data['ledger']['project_actual'], s_confirm=960)
    # The B05A benchmark measured only bounded development A00 minimum-work roles.
    # Show the available proxy without passing it off as full confirmation costs.
    t_roles = [r for r in data['challenge_minimum']['roles'] if r['method']=='A00_MATCHED' and r['player'] is not None]
    forecast['B06A_available_T_minimum_proxy'] = dict(source=INPUTS['challenge_minimum'],
        source_roles=len(t_roles), scaled_roles=72,
        cpu_core_hours_72T_proxy=3*sum(r['cpu_forecast_seconds'] for r in t_roles)/3600,
        gpu_hours_72T_proxy_before_throttle=3*sum(r['gpu_forecast_seconds'] for r in t_roles)/3600,
        multiplier_already_in_source=1.25, E3_full_cost=None, E2_full_cost=None,
        interpretation='Development100-update minimum proxy only; not measured24-player rows/full recipe or guarded CUDA wall time; not admission')
    decision = decide_route(data['pilot'], ci, budget=forecast['admitted'])
    # This accepted evidence only permits A00_ASSET closure. Never upgrade the untrained C01.
    if decision == 'EXPAND_CONFIRMATION':
        raise PermissionError('Full guarded downstream costs are unavailable; expansion prohibited')
    half_dev = max((ci[f+'_'+m]['ci95'][1]-ci[f+'_'+m]['ci95'][0])/2 for f in ('E3', 'E2') for m in ('G', 'S'))
    # Normal sqrt(n) rescaling is a descriptive planning heuristic, not a new bootstrap.
    half480 = half_dev*math.sqrt(256/480)*(2.638/1.96)
    half960 = half_dev*math.sqrt(256/960)*(2.638/1.96)
    sample = choose_sample_size(half480, half960, feasible_960=False)
    choices = dict(method='A00_ASSET', comparison_status=data['challenger']['decision'],
        historical_match=False, reuse_compute_match=False, sample_planning=sample,
        sample_status='NOT_ACTIVATED_NO_EXPANSION', sample_projection_rule='max development same-ID G/S CI halfwidth * sqrt(256/n) *2.638/1.96; heuristic only',
        model_contract=cfg['models'], engine=cfg['engine'], selection=cfg['nemesis'],
        confirmation=cfg['confirmation'], arena=cfg['arena'], data_sampling=cfg['panels'],
        robust_lambda=data['lambda']['selected_lambda'], primary_lambda=0,
        statistics='same target/source identity multiplicity then within-stratum games; six99.1666667%CI,20000bootstrap; no dev identities',
        arena_history_selection='root_seed1107|panel|game_id then ply/input_hash; opposite-color full prefixes; before H outcomes',
        registry_status='120_CONFIRMATION_MODELS_NOT_TRAINED', artifact_lock=None,
        accepted_asset_parameters=[{k: r.get(k) for k in ('model_id','status','base_sha','checkpoint_sha256','group','alpha','temperature','metadata')}
            for r in data['registry']['models'] if r['cohort'] != 'confirm'],
        code_hashes={p: sha(p) for p in ('src/nemesis/v2/freeze.py', 'src/nemesis/__main__.py')},
        config_sha256=sha(config), inputs={r['path']: r['sha256'] for r in reads})
    out = Path(out)
    out.mkdir(parents=True)
    write(out/'source_read_receipt.json', dict(inputs=reads, reviews=[{k:v for k,v in r.items() if k != 'text'} for r in reviews],
        source_hashes=choices['code_hashes'], confirmation_H_reads=0, model_loads=0, S_compare_outcome_reads=0))
    for review in reviews:
        (out/(review['stage']+'_accepted_review.md')).write_text(review['text'], encoding='utf8')
    gate = dict(stage='B05B', attempt_id='B05B-N01', decision=decision, selected_asset='A00_ASSET',
        method_comparison_status=data['challenger']['decision'], expansion_authorized=False,
        independent_review_required=True, plan_version=PLAN, base_commit=BASE, B04B_commit=B04B,
        reference_gates=data['pilot']['reference_gates'], development_GS=data['pilot']['summaries'],
        descriptive_intervals={k: ci[k] for k in ('E3_G', 'E3_S', 'E2_G', 'E2_S')},
        reason='All four valid descriptive G/S upper bounds <.005 for current A00 route; headroom is not a method upper bound',
        challenger_effects={k: data['challenger'][k] for k in ('pooled_nll_delta', 'direct_D_E3', 'direct_D_E2', 'specificity_delta_E3', 'specificity_delta_E2')},
        evaluator_aggregate_qualification=data['pilot']['reference_gates']['qualification'],
        group_base_diagnostics=data['group_base_diagnostics'], per_player_windows=data['windows'], budget_status=forecast['status'],
        forecast_sha256=None, confirmation_models_status='NOT_TRAINED', method_lock_created=False)
    write(out/'confirmation_cost_forecast.json', forecast)
    gate['forecast_sha256'] = sha(out/'confirmation_cost_forecast.json')
    write(out/'gate_decision.json', gate)
    decision_sha = sha(out/'gate_decision.json')
    write(out/'all_method_choices.json', choices)
    write(out/'holdout_access_audit.json', dict(confirmation_H_reads=0, S_compare_outcome_reads=0,
        model_loads=0, training_runs=0, input_allowlist=list(INPUTS.values()),
        registry_metadata_paths=['artifacts/V2/B03R/approved_assets.json', 'artifacts/B03/model_registry.json'],
        policy='Reject before open; metadata JSON only; confirmation no method/artifact locks',
        isolation='procedural same-user; not cryptographic', future_access_authorized=False))
    write(out/'guard_evidence.json', dict(status='COMPUTE_GUARD_PENDING', heavy_compute_allowed=False,
        implementation='NOT_IMPLEMENTED_IN_B05B_LIGHT_CLOSURE', parent_child_CPU_proof=None,
        memory_runtime_proof=None, reason='No heavy unit needed or admitted; native implementation/tiny validation optional permission unused',
        CPU_hard_cap_required_percent=20, priority_required='BelowNormal', threads_required=1,
        resident_tree_target_gib=4, available_physical_min_gib=6, commit_cap_installed=False,
        unrelated_processes_modified=False, router_guard_status_written=False))
    write(out/'stage_disposition.json', dict(stage='B05B', execution_status='DONE', research_status=decision,
        review_status='PENDING_REVIEW', downstream=downstream_disposition(decision, decision_sha),
        B08=dict(status='NOT_RUN', eligible_only_after_exact_B05B_review=True, dispatch_authorized=False),
        decision_sha256=decision_sha))
    # Explicit conservatively charged engineering/coordination allowance covers shell,
    # RED/GREEN tests, metadata checks, Git publication and final closure. No CUDA used.
    allowance_seconds = 1200
    ledger = dict(stage='B05B', attempt_id='B05B-N01', inherited_actual=data['ledger']['project_actual'],
        inherited_ledger_path=INPUTS['ledger'], inherited_ledger_sha256=sha(Path(evidence)/INPUTS['ledger']),
        stage_actual=dict(cpu_core_hours=(allowance_seconds+time.process_time()-start_cpu)/3600, gpu_hours=0),
        project_actual={}, remaining={}, caps=dict(stage_cpu_core_hours=1, stage_gpu_hours=.1,
            project_cpu_core_hours=52, project_gpu_hours=20, disk_gib=80, external_spend_usd=0),
        measurement_method='process_time(all threads) for freeze; explicit1200CPU-second allowance for all other B05B work',
        work_units=[dict(work_unit_id='B05B-N01-freeze', owner_stage='B05B', input_hash=gate['forecast_sha256'],
            start_unix=start, end_unix=time.time(), cpu_seconds=time.process_time()-start_cpu, gpu_seconds=0)],
        allowances=[dict(work_unit_id='B05B-N01-engineering-test-publication-allowance', owner_stage='B05B',
            cpu_seconds=allowance_seconds, gpu_seconds=0, reason='Conservative allowance, not measured training',
            input_hash=sha(config), start_unix=start, end_unix=time.time())], per_role_cost=[],
        cold_warm='No model/cache loads; metadata only', A03_audit_overlay_recharged=False,
        prior_allowances_preserved_in_hashed_parent=True,
        peak_disk=dict(conservative_upper_gib=data['ledger']['disk']['peak_upper_bound_gib']+.01,
            method='Inherited55.48238283511251GiB upper bound +.01GiB metadata allowance; no caches',
            cap_gib=80), external_spend_usd=0)
    # Use the single measured amount for exact arithmetic and replay.
    ledger['stage_actual']['cpu_core_hours'] = (allowance_seconds+ledger['work_units'][0]['cpu_seconds'])/3600
    reconcile_balances(ledger)
    write(out/'resource_ledger.json', ledger)
    report = f'''# B05B decision and method freeze\n\nResearch decision: {decision}. Review: PENDING_REVIEW.\n\nA00_ASSET is the only measured route. E3 G/S={data['pilot']['summaries']['E3']['G']}/{data['pilot']['summaries']['E3']['S']}; E2 G/S={data['pilot']['summaries']['E2']['G']}/{data['pilot']['summaries']['E2']['S']}. All four same-ID descriptive upper95 bounds are below .005. This supports stopping this measured route at the development gate; it is not formal24-person NOT_SUPPORTED or a method/global upper bound. Headroom remains above .005. C01 and A00_MATCHED remain untrained; METHOD_BUDGET_INCONCLUSIVE, null comparative effects and all failed/superseded trials preserved at accepted B05A. No A00 superiority claim.\n\nNo method.lock or artifact.lock is created. Choices are recorded for closure, sample480/N960 planning is not activated. Development CI scaling is a heuristic, not guaranteed power. Confirmation120 models, H and arena remain NOT_RUN. B08 requires exact independent B05B approval.\n\nDownstream cost report includes72T+24E3+24E2,4800 risk positions,24x24 matrices under both evaluators,960 natural roots,128 games at200ply.1.25 applied once;5% reserve and stage caps retained. CPU scenario totals exceed B06B/B07 gates; full training/E2/guardedGPU/memory/disk costs are unknown and fail closed. Forecasts are extrapolations, not invented measurements; no quota borrowing. Historical A01 warning remains. See confirmation_cost_forecast.json for every term and source_read_receipt.json for exact hashes.\n\nOFFICE_PRIORITY remains COMPUTE_GUARD_PENDING. No native guard implementation or enforcement is claimed; no heavy computation/tests/model loading/training/evaluation occurred. No other processes or Router state changed.\n\nInherited projectCPU15.553885904153187/GPU4.979192232880852. B05B CPU allowance1200seconds plus measured freeze; GPU0(noCUDA). Stage1CPU/.1GPU caps and lifecycle52/20/80GiB/$0 retained.\n\nLimitations: old B03 INCOMPLETE_NOT_ACCEPTED; acceptedB03R eight-player assets only; weakE2 group/per-column qualifications retained; pretrain overlap/license unknown; eight-person findings not population generalization; indexed-state frequency not reachability; StockfishU not human winrate; procedural isolation;1430board-only overlaps; CUDAFP32/cross-backend caveats. No upstream weights/private outcomes were reopened or rehashed by this metadata closure.\n'''
    (out/'stage_report.md').write_text(report, encoding='utf8')
    (out/'power_and_cost_report.md').write_text(report+'\nForecast arithmetic and halfwidth projections are in JSON; no new power simulation or full-game benchmark was admitted.\n', encoding='utf8')
    write(out/'acceptance_results.json', {f'B05B-AC0{i}': dict(status='PASS_CONDITIONAL_CLOSURE', reason=r) for i,r in enumerate([
        'A00 G/S/frequency/windows and costs considered, not NLL alone',
        'Choices recorded, no expansion lock/H access; sample planning descriptive only',
        'Current asset stop with C01 fair effects null; accepted failures retained',
        'All downstream units counted; unknown costs fail closed and scenarios not admitted',
        'No fake locks/weights/results; NOT_RUN reasons hash-bound; review before B08'],1)})
    (out/'commands.log').write_text('python -m nemesis v2 freeze --config '+str(config)+' --evidence '+str(evidence)+' --out '+str(out)+'\nexit0; metadata only\n', encoding='utf8')
    files = [dict(path=str(p.relative_to(out)).replace('\\','/'), sha256=sha(p), bytes=p.stat().st_size,
        visibility='git_evidence') for p in sorted(out.rglob('*')) if p.is_file()]
    write(out/'artifact_manifest.json', dict(stage='B05B', attempt_id='B05B-N01', plan_version=PLAN,
        base_commit=BASE, review_status='PENDING_REVIEW', files=files, manifest_self_excluded=True))
    return StageResult('B05B', 'DONE', decision, out, 0, decision+'; no expansion, independent review required')
