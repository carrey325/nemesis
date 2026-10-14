"""Shared records from INTERFACES v1.0; algorithms belong to later Batches."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PositionContext:
    fen6: str
    history_uci: tuple[str, ...]
    initial_fen: str
    root_color: bool
    self_elo: int
    oppo_elo: int
    tc_base: int
    tc_increment: int
    clock_before: float | None
    history_hash: str


@dataclass(frozen=True)
class MoveDistribution:
    moves_uci: tuple[str, ...]
    probabilities: tuple[float, ...]
    model_id: str
    model_hash: str
    input_hash: str


@dataclass(frozen=True)
class EngineBudget:
    nodes: int
    multipv: int
    threads: int
    hash_mb: int
    options_hash: str


@dataclass(frozen=True)
class EngineEvaluation:
    cp: int | None
    mate: int | None
    wdl: tuple[int, int, int] | None
    expected_score: float
    pov: bool
    actual_nodes: int
    elapsed_seconds: float
    budget_hash: str
    engine_hash: str
    status: str


@dataclass(frozen=True)
class ReplyEvaluation:
    move_uci: str
    evaluation: EngineEvaluation


@dataclass(frozen=True)
class CandidateEvaluation:
    move_uci: str
    objective: EngineEvaluation
    cp_loss: float | None
    safe: bool
    replies: tuple[ReplyEvaluation, ...]


@dataclass
class DecisionTrace:
    root_id: str
    identity_id: str | None
    method: str
    selected_move: str
    candidates: list
    objective_loss: float | None
    expected_utilities: list
    disagreement: list
    fallback_reason: str | None
    input_hash: str
    decision_model_hashes: list[str]
    evidence_hash: str


@dataclass(frozen=True)
class ProbeResult:
    root_id: str
    targeted_identity: str
    evaluated_identity: str
    evaluator_family: str
    method: str
    selected_move: str
    audit_Q: float
    audit_cp_loss: float
    source_game_id: str
    source_window: str
    decision_hash: str
    evaluator_hash: str


@dataclass
class StageDisposition:
    stage: str
    branch: str
    execution_status: str
    artifact_status: str
    research_status: str
    reason: str
    permitted_downstream_uses: list[str] = field(default_factory=list)
    actual_commands: list[str] = field(default_factory=list)
    artifact_manifest: str = "artifact_manifest.json"
    ready: bool = False
    exit_code: int = 4
