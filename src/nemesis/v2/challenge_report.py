"""Explicit missing-data tables for an allowed bounded resource closure."""
import pyarrow as pa
import pyarrow.parquet as pq
import json
from pathlib import Path
from nemesis.data.common import write_json, digest
from .types import StageResult
from .challenge import DEV


def write_missing_tables(out,reason):
    methods=('A00_ASSET','A00_MATCHED','C01')
    def table(name,rows,schema): pq.write_table(pa.Table.from_pylist(rows,schema=schema),out/name)
    prediction=[dict(method=m,player=p,window=w,metric=metric,value=None,status='NOT_RUN',reason=reason)
                for m in methods for p in DEV for w in ('H1','H2')
                for metric in ('NLL','Top1','Top3','multiclass_Brier','risk_Brier')]
    table('human_prediction_comparison.parquet',prediction,pa.schema([
        ('method',pa.string()),('player',pa.string()),('window',pa.string()),('metric',pa.string()),
        ('value',pa.float64()),('status',pa.string()),('reason',pa.string())]))
    schema=pa.schema([(k,pa.string()) for k in ('method','family','strategy_identity','target_identity')]+
                     [('delta',pa.float64()),('n_roots',pa.int64()),('status',pa.string()),('reason',pa.string())])
    for family in ('E3','E2'):
        rows=[dict(method=m,family=family,strategy_identity=i,target_identity=j,delta=None,n_roots=0,
                   status='NOT_RUN',reason=reason) for m in methods for i in DEV for j in DEV]
        table(f'{family}_compare_matrix.parquet',rows,schema)
    table('all_trials.parquet',[dict(method=m,role='G' if p is None else f'T_{p}_{s}',
        status='NOT_STARTED_BUDGET',updates=None,reason=reason) for m in ('A00_MATCHED','C01')
        for p,s in [(None,1729)]+[(p,s) for p in DEV for s in (1729,2718,3141)]],pa.schema([
            ('method',pa.string()),('role',pa.string()),('status',pa.string()),('updates',pa.int64()),('reason',pa.string())]))


def close_inconclusive(out,lock,folder):
    seal=json.loads((out/'train/phase_seal.json').read_text())
    if seal['research_status']!='METHOD_BUDGET_INCONCLUSIVE':
        raise PermissionError('This bounded closure cannot evaluate an unimplemented complete challenge')
    reason='Fair comparison not run: mandatory shared engine forecast fails R4 CPU1 quota and5%reserve'
    write_missing_tables(out,reason)
    write_json(out/'paired_method_effects.json',dict(status='NOT_RUN',reason=reason,
        D={'E3':None,'E2':None},delta_S={'E3':None,'E2':None},paired_CI=None,
        shared_baseline='GENERIC_A00_ASSET',own_group_gain=None,
        identity_specific_gain=None,generic_gain=None,NLL_delta=None,
        useful_opportunity_frequency={'value':None,'conditional_denominator_planned':128,
            'natural_denominator_B04B':512,'natural_eligible_B04B':105,
            'S_compare_evaluated_roots':0,'reason':reason},
        S_compare_outcome_reads=0,confirmation_H_reads=0,
        original_A00_B04B_evidence_path='artifacts/V2/B04B/pilot_decision.json',
        original_A00_B04B_evidence_sha256=digest('artifacts/V2/B04B/pilot_decision.json'),
        frequency_not_transferred_from_old_asset=True,models_complete=False))
    write_json(out/'stage_disposition.json',dict(stage='B05A',attempt_id=lock['run_id'],
        execution_status='DONE',research_status='METHOD_BUDGET_INCONCLUSIVE',review_status='PENDING_REVIEW',
        exit_code=0,reason=reason,plan_version=lock['plan_version'],base_commit=lock['base_commit'],
        confirmation_expansion_authorized=False,full_challenge_implemented=False,
        actual_scope='C01 engineering + complete fair lock + minimum feasibility check + allowed bounded resource closure',
        permitted_downstream_uses=['Independent B05A review','After approval B05B may summarize inconclusive method comparison alongside B04B asset evidence'],
        forbidden_claims=['C01 utility upgrade','C01 method failure','global method upper bound','same-budget historical A00 win','confirmation complete']))
    write_json(folder/'phase_seal.json',dict(lock_sha256=digest(out/'challenge.lock.json'),run_id=lock['run_id'],
        research_status='METHOD_BUDGET_INCONCLUSIVE',outcome_access_count=0,
        source_train_seal_sha256=digest(out/'train/phase_seal.json')))
    return StageResult('B05A','DONE','METHOD_BUDGET_INCONCLUSIVE',out,0,reason)
