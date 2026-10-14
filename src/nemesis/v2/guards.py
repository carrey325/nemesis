"""Reject legacy expansion before creating paths, meters or model loaders."""
from pathlib import Path
import yaml

def v2_bound():
    path=Path('workflow/project_config.yaml')
    if path.exists():
        config=yaml.safe_load(path.read_text(encoding='utf8'))
        if config.get('active_batch_index')=='nemesis-stage1-plan/BATCH_INDEX.json':
            return True
    proposal=Path('nemesis-stage1-plan/configs/experiment.v2.proposed.yaml')
    return proposal.exists() and yaml.safe_load(proposal.read_text(encoding='utf8')).get('schema_version')==2

def reject_legacy_training():
    if v2_bound():
        raise PermissionError('V2 binding forbids legacy auto-training; confirmation requires accepted B05B/B06A gates')

def require_new_output(out):
    out=Path(out).resolve()
    legacy=Path('artifacts/B03').resolve()
    if out==legacy or out.is_relative_to(legacy):
        raise PermissionError('V2 validation cannot write original B03 evidence')
    root=Path('artifacts/V2').resolve()
    if not out.is_relative_to(root): raise ValueError('V2 evidence must be under artifacts/V2')
