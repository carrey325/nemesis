"""Export an allowlisted, aggregate-only presentation from a research checkout.

No PGN, names, model weights, per-position predictions or runtime locks are copied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("research", type=Path)
    args = parser.parse_args()
    research = args.research.resolve()
    receipts = []

    def read(name):
        path = research / name
        raw = path.read_bytes()
        receipts.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest()})
        return json.loads(raw)

    matrix = read("artifacts/B03/validation/dev_prediction_matrix.json")
    gate = read("artifacts/B03/dev_gate.json")
    train = read("artifacts/B03/training_summary.json")
    qv = read("artifacts/V2/B05A/qv_real_model_evidence.json")
    boot = read("artifacts/V2/B04B/bootstrap.json")
    baseline = read("artifacts/B02/metrics_summary.json")
    dataset_report = research / "artifacts/B01/stage_report.md"
    receipts.append({"path": "artifacts/B01/stage_report.md", "sha256": hashlib.sha256(dataset_report.read_bytes()).hexdigest()})
    prediction_to_decision = read("artifacts/V2/B04B/prediction_to_decision.json")
    players = sorted({r["player_id"] for r in matrix})

    def nll(model, player):
        rows = [r for r in matrix if r["model_id"] == model and r["player_id"] == player]
        assert len(rows) == 2 and {r["split"] for r in rows} == {"H1", "H2"}
        return sum(r["nll"] * r["games"] for r in rows) / sum(r["games"] for r in rows)

    people = []
    identity = []
    for p in players:
        own = nll("T_ensemble_" + p, p)
        others = [nll("T_ensemble_" + q, p) for q in players if q != p]
        group = nll("G", p)
        people.append(dict(player=p, group_nll=group, personal_nll=own,
                           nll_gain=group-own, wrong_identity_mean_nll=sum(others)/7,
                           personal_rank=1+sum(x < own for x in others)))
    for source in players:
        identity.append([nll("G", target)-nll("T_ensemble_"+source, target) for target in players])
    assert abs(sum(p["nll_gain"] for p in people)/8-gate["t_mean_nll_improvement"]) < 1e-12

    histories = []
    for t in train:
        if t["model_id"].startswith("T_"):
            histories.append({k: t[k] for k in ("model_id", "steps", "epochs", "stopping")})

    effects = {}
    for family in ("E3", "E2"):
        summary = read(f"artifacts/V2/B04B/{family}_PERSONAL_PLAIN_summary.json")
        effects[family] = {"gain": summary["G"], "specificity": summary["S"],
            "gain_by_player": summary["per_target_G"],
            "ci95_fixed_players": boot["results"]["fixed_identity_game_only"][family+"_G"]["ci95"],
            "ci95_player_and_game": boot["results"]["same_ID_block"][family+"_G"]["ci95"],
            "specificity_ci95_player_and_game": boot["results"]["same_ID_block"][family+"_S"]["ci95"]}

    results = dict(schema_version=1,
        dataset=dict(players=32, games=146156, samples=2540865,
                     development_players=8, development_games=7409, development_decisions=122593),
        methods=dict(adapter_parameters=17936, lora_parameters=16384, lora_rank=4,
                     checkpoints=43, precision="CUDA FP32", personal_seeds=[1729,2718,3141]),
        prediction=dict(players=people, identity_gain_matrix=identity,
                        aggregation="decisions within game, games within player, equal player weights",
                        mean_nll_gain=gate["t_mean_nll_improvement"], original_baseline=baseline["overall"]),
        calibration_histories=histories, decision=effects,
        prediction_to_decision=prediction_to_decision["players"],
        lora_engineering=dict(samples=qv["samples"], precision=qv["precision"], device=qv["device"],
            trainable_parameters=qv["trainable_parameters"],
            zero_probability_error=qv["zero_probability_error"],
            grad_eval_probability_error=qv["grad_eval_probability_error"],
            cached_prefix_probability_error=qv["cached_prefix_probability_error"],
            frozen_base_unchanged=qv["base_before"]==qv["base_after"],
            K_and_GAB_unchanged=qv["K_and_GAB_unchanged"],
            resume_parameter_error=qv["resume_parameter_error"],
            optimizer_rng_step_restored=qv["optimizer_rng_step_restored"],
            injected_fault_step=qv["injected_fault_step"], overfit=qv["overfit"],
            scope="32-sample CUDA engineering fixture; method-effect comparison not run"))
    evidence = ROOT / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "results.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8", newline="\n")

    copied = []
    files = sorted((research / "src").rglob("*.py"))
    files += [research / "tests/unit" / name for name in
              ("test_adapters.py", "test_qv_lora.py", "test_calibration.py")]
    for path in files:
        relative = path.relative_to(research)
        target = ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copied.append(dict(path=relative.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    commit = subprocess.check_output(["git", "-C", str(research), "rev-parse", "HEAD"], text=True).strip()
    provenance = dict(research_repository="carrey325/nemesis-dev", research_commit=commit,
        evidence_sources=receipts, unmodified_source_files=copied,
        export_scope="Anonymous player-level summaries, training histories, first-party source and component tests.")
    (evidence / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(f"Exported {len(people)} players, {len(histories)} training histories, {len(copied)} source/test files.")


if __name__ == "__main__":
    main()
