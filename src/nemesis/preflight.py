"""Bounded B00 checks using immutable, local source records.

Provision sources before running; this module never silently resolves main.
The only network operation in B00 is one bounded, public PGN request unless a
verified receipt for that same request is supplied in the local runtime lock.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

from .types import StageDisposition


class IntegrityError(ValueError):
    """A source differs from its immutable lock."""


class ResourceLimit(RuntimeError):
    """The stage cannot continue within its declared cap."""


def require_revision(revision: str | None) -> str:
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("An immutable 40-character commit revision is required")
    return revision


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_file(record: dict) -> Path:
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise IntegrityError(f"SHA256 mismatch: {path}")
    return path


def cpu_core_hours(elapsed_seconds: float, threads: int) -> float:
    if elapsed_seconds < 0 or threads < 1:
        raise ValueError("Nonnegative duration and positive thread count required")
    return elapsed_seconds * threads / 3600


def forecast_conflicts(estimates: dict, stage_caps: dict) -> list[dict]:
    forecast = estimates["B07_384_games_200plies_cpu_core_hours_planning"]
    cap = stage_caps["B07"]["cpu_core_hours"]
    if forecast > cap:
        return [{"stage": "B07", "forecast_cpu_core_hours": forecast,
                 "approved_cpu_core_hours_cap": cap,
                 "status": "FORECAST_EXCEEDS_CAP_NOT_ACTUAL_OVERRUN"}]
    return []


A01_PATH = Path("nemesis-stage1-plan/amendments/A01_B07_FORECAST_DEFERRAL.md")
A01_SHA256 = "981f6eadb62f37ab59713a22cde64d8cf42f7229a55864a49439bfff7e0f037b"


def apply_forecast_gate(disposition: StageDisposition, conflicts: list[dict],
                        amendment_path: Path = A01_PATH) -> list[dict]:
    """Apply only A01's retained B00 forecast warning, after runtime success.

    This cannot convert a source/runtime/resource failure to success or authorize
    another forecast, an actual overrun, or B05 freeze. Keep conflicts intact.
    """
    if (disposition.stage != "B00" or disposition.execution_status != "DONE"
            or disposition.artifact_status != "VALID" or not disposition.ready
            or disposition.exit_code != 0 or not conflicts):
        return []
    approved = amendment_path.is_file() and sha256(amendment_path) == A01_SHA256
    deferred, blocking = [], []
    for conflict in conflicts:
        forecast = conflict.get("forecast_cpu_core_hours")
        if (approved and conflict.get("stage") == "B07"
                and conflict.get("approved_cpu_core_hours_cap") == 16
                and conflict.get("status") == "FORECAST_EXCEEDS_CAP_NOT_ACTUAL_OVERRUN"
                and isinstance(forecast, (int, float))
                and 17.684779919749424 <= forecast <= 17.855287777152512):
            deferred.append({**conflict, "gate_status": "DEFERRED_TO_B05",
                             "amendment_path": str(amendment_path),
                             "amendment_sha256": A01_SHA256, "recheck_stage": "B05",
                             "freeze_allowed_by_deferral": False})
        else:
            blocking.append(conflict)
    if blocking:
        disposition.execution_status, disposition.ready, disposition.exit_code = "BLOCKED", False, 4
        disposition.reason = "Runtime checks passed; unapproved resource forecast conflict requires user decision: " + json.dumps(blocking)
        disposition.permitted_downstream_uses = ["inspect B00 evidence", "resolve forecast conflict; no dependent stage dispatch"]
    else:
        disposition.reason = ("Required B00 runtime checks passed; A01 defers the retained B07 forecast "
                              "warning to mandatory B05 measurement before freeze. Actual caps unchanged; "
                              "no scientific effectiveness tested.")
        disposition.permitted_downstream_uses = ["independent B00 review", "B01 only after Router acceptance",
                                                "B05 must reassess B07 before freeze; pause if still over cap"]
    return deferred


def policy_head_lock(model, module_path: str) -> dict:
    try:
        module = model.get_submodule(module_path)
    except AttributeError as exc:
        raise ValueError(f"Policy head is absent: {module_path}") from exc
    params = list(module.named_parameters())
    if not params or sum(p.numel() for _, p in params) == 0:
        raise ValueError("Policy head has no trainable parameters")
    return {
        "module_path": module_path,
        "module_type": type(module).__name__,
        "parameter_names": [f"{module_path}.{name}" for name, _ in params],
        "parameter_shapes": {name: list(p.shape) for name, p in params},
        "parameter_count": sum(p.numel() for _, p in params),
        "backbone_training_mode": "eval",
        "freeze_rule": f"only {module_path} parameters require gradients",
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def smoke_positions(count: int = 32) -> list[dict]:
    """Deterministic synthetic legal walk, with complete real prior moves."""
    import chess
    rng = random.Random(20261007)
    board = chess.Board()
    history = []
    result = []
    for _ in range(count):
        if board.is_game_over():
            board = chess.Board()
            history = []
        result.append({"initial_fen": chess.STARTING_FEN, "fen": board.fen(),
                       "history_uci": list(history), "self_elo": 1900, "oppo_elo": 1900})
        move = rng.choice(sorted(board.legal_moves, key=lambda m: m.uci()))
        board.push(move)
        history.append(move.uci())
    return result


def validate_distribution(board, moves: list[str], probabilities: list[float]) -> None:
    if len(moves) != len(set(moves)) or set(moves) != {m.uci() for m in board.legal_moves}:
        raise IntegrityError("Distribution does not cover exactly all legal moves")
    if len(moves) != len(probabilities) or any(not math.isfinite(p) or p < 0 for p in probabilities):
        raise IntegrityError("Invalid probabilities")
    if abs(sum(probabilities) - 1) > 1e-6:
        raise IntegrityError("Probability mass differs from 1")


def benchmark_model(record: dict, positions: list[dict], device: str) -> tuple[dict, dict]:
    """One model at a time, FP32, official encoding; no B02 predictor API."""
    import chess
    import torch
    require_revision(record["revision"])
    checkpoint = verify_file(record)
    torch.set_num_threads(1)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    if record["family"] == "maia3":
        from maia3.uci import parse_args
        from maia3.models import MAIA3Model
        from maia3.dataset import tokenize_board, get_historical_tokens, get_legal_moves_mask
        from maia3.utils import get_all_possible_moves, mirror_move
        cfg = parse_args(["--model", record["alias"], "--checkpoint-path", str(checkpoint),
                          "--device", device, "--no-use-amp", "--use-uci-history"])
        model = MAIA3Model(cfg)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict({k.replace("smolgen", "gab"): v for k, v in state.items()}, strict=True)
        model.to(device).eval()
        vocabulary = get_all_possible_moves()
        move_map = {move: i for i, move in enumerate(vocabulary)}
        architecture = {"class": "maia3.models.MAIA3Model", "config": record["config"],
                        "history_length": cfg.history, "include_time_info": cfg.include_time_info,
                        "policy_paths": ["proj_sq_from", "proj_sq_to", "promo_bias_proj"],
                        "vocabulary_size": len(vocabulary), "policy_pov": "side_to_move"}

        def predict(position):
            board = chess.Board(position["initial_fen"])
            history = deque([tokenize_board(board)], maxlen=cfg.history)
            for move in position["history_uci"]:
                board.push_uci(move)
                history.append(tokenize_board(board))
            if board.fen() != position["fen"]:
                raise IntegrityError("Historical reconstruction mismatch")
            tokens = get_historical_tokens(history, cfg, 180, 0, 180, 0).unsqueeze(0).to(device)
            elo = torch.tensor([1900], dtype=torch.long, device=device)
            with torch.inference_mode():
                logits, _, _ = model(tokens, elo, elo)
                mask = get_legal_moves_mask(board, move_map).to(device)
                probs = logits[0].float().masked_fill(~mask, -torch.inf).softmax(-1)
            indices = mask.nonzero().flatten().tolist()
            moves = [vocabulary[i] if board.turn else mirror_move(vocabulary[i]) for i in indices]
            return board, moves, probs[indices].tolist()
    elif record["family"] == "maia2":
        from maia2.main import MAIA2Model
        from maia2.utils import parse_args, get_all_possible_moves, create_elo_dict, mirror_move
        from maia2.train import load_model_state_dict
        from maia2.inference import preprocessing, _masked_softmax
        cfg = parse_args(verify_file(record["config_file"]))
        vocabulary = get_all_possible_moves()
        move_map = {move: i for i, move in enumerate(vocabulary)}
        elo_map = create_elo_dict()
        model = MAIA2Model(len(vocabulary), elo_map, cfg)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        load_model_state_dict(model, ckpt["model_state_dict"])
        model.to(device).eval()
        architecture = {"class": "maia2.main.MAIA2Model", "config": vars(cfg),
                        "policy_head": policy_head_lock(model, "fc_1"),
                        "vocabulary_size": len(vocabulary), "policy_pov": "side_to_move",
                        "official_value_output_pov": "white_expected_score",
                        "history_length": 1, "include_time_info": False}

        def predict(position):
            board = chess.Board(position["fen"])
            tensor, self_elo, oppo_elo, legal = preprocessing(board.fen(), 1900, 1900, elo_map, move_map)
            with torch.inference_mode():
                logits, _, _ = model(tensor.unsqueeze(0).to(device),
                                     torch.tensor([self_elo], device=device),
                                     torch.tensor([oppo_elo], device=device))
                probs = _masked_softmax(logits, legal.unsqueeze(0).to(device))[0]
            indices = legal.nonzero().flatten().tolist()
            moves = [vocabulary[i] if board.turn else mirror_move(vocabulary[i]) for i in indices]
            return board, moves, probs[indices].tolist()
    else:
        raise ValueError("Unsupported model family")
    if device == "cuda":
        torch.cuda.synchronize()
    cold = time.perf_counter() - start
    rows = []
    for i, position in enumerate(positions):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        board, moves, probs = predict(position)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        validate_distribution(board, moves, probs)
        rows.append({"position_index": i, "elapsed_seconds": elapsed,
                     "moves_uci": moves, "probabilities": probs})
    architecture["total_parameters"] = sum(p.numel() for p in model.parameters())
    result = {"status": "PASS", "positions": len(rows), "device": device, "precision": "FP32",
              "checkpoint_sha256": record["sha256"], "cold_load_seconds": cold,
              "first_inference_seconds": rows[0]["elapsed_seconds"],
              "warm_median_seconds": statistics.median(r["elapsed_seconds"] for r in rows[1:]) if len(rows) > 1 else None,
              "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
              "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
              "outputs": rows}
    del model, ckpt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result, architecture


def benchmark_stockfish(record: dict, positions: list[dict], config: dict) -> dict:
    import chess
    import chess.engine
    import psutil
    path = verify_file(record)
    for net in record["nnue_files"]:
        verify_file(net)
    options = {"Threads": config["threads"], "Hash": config["hash_mb"],
               "Skill Level": config["skill_level"], "UCI_LimitStrength": config["limit_strength"],
               "UCI_ShowWDL": config["show_wdl"], "SyzygyPath": ""}
    rows = []
    start = time.perf_counter()
    engine = chess.engine.SimpleEngine.popen_uci(str(path), timeout=30)
    try:
        engine.ping()  # isready/readyok after popen_uci's uci/uciok
        engine.configure(options)
        available = {k: {"type": v.type, "default": v.default, "min": v.min, "max": v.max, "var": v.var}
                     for k, v in engine.options.items()}
        for i, position in enumerate(positions):
            board = chess.Board(position["initial_fen"])
            for move in position["history_uci"]:
                board.push_uci(move)
            if board.fen() != position["fen"]:
                raise IntegrityError("Stockfish historical reconstruction mismatch")
            engine.configure({"Clear Hash": None})
            tick = time.perf_counter()
            info = engine.analyse(board, chess.engine.Limit(nodes=config["forced_candidate_nodes"]), multipv=1)[0]
            elapsed = time.perf_counter() - tick
            if not info.get("pv") or info["pv"][0] not in board.legal_moves or not info.get("nodes"):
                raise IntegrityError("Illegal or empty engine output")
            score = info["score"].pov(board.turn)
            wdl = info["wdl"].pov(board.turn)
            rows.append({"position_index": i, "nodes": info["nodes"], "elapsed_seconds": elapsed,
                         "pv": [m.uci() for m in info["pv"]], "cp": score.score(), "mate": score.mate(),
                         "wdl": list(wdl), "expected_score": wdl.expectation(), "pov": board.turn})
        child = psutil.Process(engine.transport.get_pid()).cpu_times()
        child_cpu_seconds = child.user + child.system
    finally:
        engine.quit()
    seconds = sum(r["elapsed_seconds"] for r in rows)
    return {"status": "PASS", "uci": True, "isready": True, "positions": len(rows),
            "requested_nodes_per_position": config["forced_candidate_nodes"], "options": options,
            "available_options": available, "wall_seconds": time.perf_counter() - start,
            "process_cpu_seconds": child_cpu_seconds,
            "search_cpu_core_hours": cpu_core_hours(seconds, config["threads"]),
            "nodes_per_second": sum(r["nodes"] for r in rows) / seconds, "outputs": rows}


def check_endpoint(record: dict) -> dict:
    import chess.pgn
    receipt_path = verify_file(record["receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf8"))
    if receipt.get("status") != 200 or receipt.get("params", {}).get("max") != 1:
        raise FileNotFoundError("Required bounded PGN endpoint did not return HTTP 200")
    pgn = verify_file(record)
    stream = io.StringIO(pgn.read_text(encoding="utf8"))
    game = chess.pgn.read_game(stream)
    if game is None or game.errors or chess.pgn.read_game(stream) is not None:
        raise IntegrityError("Endpoint must contain exactly one parseable game")
    board = game.board()
    for move in game.mainline_moves():
        if move not in board.legal_moves:
            raise IntegrityError("Illegal PGN move")
        board.push(move)
    return {"status": "PASS", "http_status": 200, "games": 1, "plies": board.ply(),
            "content_sha256": record["sha256"], "receipt_sha256": record["receipt"]["sha256"],
            "parameters": receipt["params"], "request_utc": receipt["utc"],
            "endpoint_template": "https://lichess.org/api/games/user/{local_username}",
            "raw_receipt_local_only": str(receipt_path),
            "rate_policy": "one request at a time; on 429 wait >=60s and honor Retry-After; at most 3 retries",
            "not_a_cohort_game": True}


def run_preflight(config_path: Path, out_dir: Path) -> StageDisposition:
    if out_dir.exists():
        raise FileExistsError(f"Output already exists: {out_dir}; use a new run directory")
    out_dir.mkdir(parents=True)
    command = subprocess.list2cmdline([sys.executable, "-m", "nemesis", "preflight", "--config", str(config_path), "--out", str(out_dir)])
    disposition = StageDisposition("B00", "main", "BLOCKED", "INCOMPLETE", "NOT_TESTED", "preflight incomplete", actual_commands=[command])
    environment = {"utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                   "executable": sys.executable, "os": platform.platform(), "cpu": platform.processor(),
                   "logical_cpus": os.cpu_count(), "gpu": "unknown",
                   "nemesis_code_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    sources, architecture, benchmarks = {}, {}, {"models": {}, "stockfish": None, "endpoint": None}
    process_start, gpu_seconds, engine_cpu_seconds = time.process_time(), 0.0, 0.0
    try:
        import yaml
        import psutil
        import torch
        config = yaml.safe_load(config_path.read_text(encoding="utf8"))
        lock_path = Path(os.environ.get("NEMESIS_RUNTIME_LOCK", "vendor/runtime_lock.json"))
        lock = json.loads(lock_path.read_text(encoding="utf8"))
        sources = lock
        for record in lock["source_files"]:
            verify_file(record)
        for record in lock["models"].values():
            require_revision(record["revision"])
            require_revision(record["code_revision"])
            verify_file(record)
        if set(lock["models"]) != {"maia3_5m", "maia3_23m", "maia2_blitz"}:
            raise IntegrityError("Exactly the three required model roles must be locked")
        if sys.version_info[:2] != (3, 12):
            raise ImportError("B00 requires the verified Python 3.12 environment")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        environment.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                           ram_bytes=psutil.virtual_memory().total, physical_cpu_cores=psutil.cpu_count(logical=False),
                           cuda_available=torch.cuda.is_available(), device=device,
                           gpu=torch.cuda.get_device_name() if device == "cuda" else "unknown",
                           vram_bytes=torch.cuda.get_device_properties(0).total_memory if device == "cuda" else None,
                           observed_hardware=lock.get("environment_evidence"),
                           runtime_lock_sha256=sha256(lock_path), config_sha256=sha256(config_path))
        positions = smoke_positions()
        write_json(out_dir / "smoke_inputs.json", positions)
        benchmarks["input_sha256"] = sha256(out_dir / "smoke_inputs.json")
        caps = config["resources"]["stage_caps"]["B00"]
        setup_cpu = lock.get("setup_resource_usage", {}).get("cpu_core_hours_upper_bound", 0)

        def guard():
            used_cpu = setup_cpu + (time.process_time() - process_start + engine_cpu_seconds) / 3600
            if used_cpu >= caps["cpu_core_hours"] or gpu_seconds / 3600 >= caps["gpu_hours"]:
                raise ResourceLimit("B00 resource cap reached")
            if sum(p.stat().st_size for p in Path.cwd().rglob("*") if p.is_file()) / 2**30 >= config["resources"]["disk_gib_cap"]:
                raise ResourceLimit("Project disk cap reached")

        for name in ["maia3_5m", "maia3_23m", "maia2_blitz"]:
            guard()
            tick = time.perf_counter()
            try:
                benchmarks["models"][name], architecture[name] = benchmark_model(lock["models"][name], positions, device)
            finally:
                if device == "cuda":
                    gpu_seconds += time.perf_counter() - tick
            write_json(out_dir / "runtime_benchmark.json", benchmarks)
            guard()
        benchmarks["stockfish"] = benchmark_stockfish(lock["stockfish"], positions, config["engine"])
        engine_cpu_seconds = benchmarks["stockfish"]["process_cpu_seconds"]
        benchmarks["endpoint"] = check_endpoint(lock["endpoint"])
        guard()
        nps = benchmarks["stockfish"]["nodes_per_second"]
        engine = config["engine"]
        legal_reply_mean = statistics.mean(len(row["moves_uci"]) for row in benchmarks["models"]["maia3_23m"]["outputs"])
        per_root = engine["nomination_nodes"] + engine["candidate_k"] * engine["forced_candidate_nodes"] + engine["candidate_k"] * legal_reply_mean * engine["reply_nodes"]
        benchmarks["estimates"] = {
            "label": "ESTIMATE_NOT_MEASURED_EXPERIMENT", "observed_legal_moves_mean_proxy_for_replies": legal_reply_mean,
            "nodes_per_probe_root": per_root,
            "B04_smoke_32_roots_cpu_core_hours": per_root * 32 / nps / 3600 * engine["threads"],
            "B05_dev_480_roots_cpu_core_hours": per_root * 480 / nps / 3600 * engine["threads"],
            "B06_480_roots_shared_cache_cpu_core_hours": per_root * 480 / nps / 3600 * engine["threads"],
            "B06_960_roots_shared_cache_cpu_core_hours": per_root * 960 / nps / 3600 * engine["threads"],
            "B07_384_games_200plies_cpu_core_hours_planning": (engine["nomination_nodes"] + 0.75 * (per_root - engine["nomination_nodes"])) * 384 * 100 / nps / 3600 * engine["threads"],
            "training_gpu_hours": None, "training_cpu_core_hours": None, "training_feasibility_proven": False,
            "assumptions": ["microbench nps extrapolated; real middlegame branching and throughput unknown",
                            "all legal replies; 5 candidates; exact specified budgets; probe caches shared across identities",
                            "arena planning: 100 Nemesis plies per 200-ply game, 1/4 SF_BEST nomination-only, 3/4 full scoring; no early endings",
                            "audit, root screening, calibration, human-risk labels and sensitivity add work",
                            "training estimate remains unknown until B03/B05 measurement"],
            "disk_gib_planning": {"raw_pgn_and_processed": 8, "weights_and_environment": 12,
                                  "engine_cache_and_reports": 20, "total": 40, "status": "assumption"}}
        disposition.execution_status = "DONE"
        disposition.artifact_status = "VALID"
        disposition.ready = True
        disposition.exit_code = 0
        disposition.reason = "Required local B00 runtime checks passed; no scientific effectiveness tested"
        disposition.permitted_downstream_uses = ["independent B00 review", "B01 only after Router acceptance"]
        conflicts = forecast_conflicts(benchmarks["estimates"], config["resources"]["stage_caps"])
        benchmarks["resource_forecast_conflicts"] = conflicts
        benchmarks["deferred_forecast_gates"] = apply_forecast_gate(
            disposition, conflicts, config_path.parent.parent / "amendments/A01_B07_FORECAST_DEFERRAL.md")
    except ResourceLimit as exc:
        disposition.execution_status, disposition.exit_code, disposition.reason = "RESOURCE_LIMIT", 4, str(exc)
    except IntegrityError as exc:
        disposition.execution_status, disposition.exit_code, disposition.reason = "FAILED", 3, str(exc)
    except (FileNotFoundError, ImportError) as exc:
        disposition.reason = str(exc)
    except Exception as exc:
        disposition.execution_status, disposition.exit_code, disposition.reason = "FAILED", 5, f"{type(exc).__name__}: {exc}"
        import traceback
        (out_dir / "error.log").write_text(traceback.format_exc(), encoding="utf8")
    benchmarks["actual_resource_usage"] = {
        "runtime_process_cpu_seconds": time.process_time() - process_start,
        "engine_process_cpu_seconds": engine_cpu_seconds,
        "gpu_allocated_wall_hours_upper_bound": gpu_seconds / 3600,
        "runtime_cpu_core_hours": (time.process_time() - process_start + engine_cpu_seconds) / 3600,
        "external_spend_usd": 0,
        "measurement": "CPU user+system processor seconds; GPU wall time while model loaded, conservative; setup separately locked"}
    for filename, value in [("environment.json", environment), ("source_lock.json", sources),
                            ("architecture_lock.json", architecture), ("runtime_benchmark.json", benchmarks),
                            ("stage_disposition.json", asdict(disposition))]:
        write_json(out_dir / filename, value)
    (out_dir / "commands.log").write_text(command + f"\nexit_code={disposition.exit_code}\n", encoding="utf8")
    report = f"# B00 preflight\n\nExecution: {disposition.execution_status}\n\n{disposition.reason}\n\n" + \
             "Measured: see runtime_benchmark.json. Source confirmed: see source_lock.json. " + \
             "Training duration and scientific effectiveness remain unknown.\n"
    (out_dir / "preflight_report.md").write_text(report, encoding="utf8")
    (out_dir / "stage_report.md").write_text(report, encoding="utf8")
    manifest = [{"path": p.name, "sha256": sha256(p), "size_bytes": p.stat().st_size,
                 "format": p.suffix.lstrip("."), "table_rows": None, "source": "B00 local preflight",
                 "generation_command": command} for p in sorted(out_dir.iterdir()) if p.is_file()]
    write_json(out_dir / "artifact_manifest.json", manifest)
    return disposition
