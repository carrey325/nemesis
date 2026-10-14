from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class StageResult:
    stage_id: str
    execution_status: str
    research_status: str
    artifact_dir: Path
    exit_code: int
    reason: str

@dataclass(frozen=True)
class AssetBundle:
    registry_path: Path
    registry_hash: str
    source_commit: str
    legacy_plan_version: str
    approved_asset_ids: tuple[str,...]

@dataclass(frozen=True)
class RootContext:
    root_id: str
    game_id: str
    stratum: tuple[str,str,str]
    initial_fen: str
    fen6: str
    history_uci: tuple[str,...]
    root_color: bool
    input_hash: str

@dataclass(frozen=True)
class CandidateBundle:
    root_id: str
    moves: tuple[str,...]
    safe_moves: tuple[str,...]
    cp_root: dict[str,float|None]
    reply_values: dict[str,dict[str,float]]
    engine_hash: str
    budget_hash: str
    status: str

@dataclass(frozen=True)
class Selection:
    root_id: str
    method_id: str
    identity: str
    move_uci: str
    selection_hash: str
    source_policy_hash: str

@dataclass(frozen=True)
class AuditRecord:
    root_id: str
    method_id: str
    target_id: str
    evaluator_family: str
    q: float
    loss_cp: float|None
    audit_hash: str
    loss_reason: str|None = None
    stronger_reference_loss_cp: float|None = None
