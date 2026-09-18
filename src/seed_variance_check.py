"""Lab 2 Task 3 — seed variance check for the two closest-ranked trials.

Run:  python -m src.seed_variance_check

The main sweep (src/tune_remote.py) ran every hyperparameter combination on a single
seed (20260101), so the 0.0012 val_roc_auc gap between its top two trials — trial-01
(n_estimators=100, max_depth=4, min_samples_leaf=5) and trial-07 (n_estimators=300,
max_depth=4, min_samples_leaf=5) — could be real signal or pure seed noise. There was
no way to tell from one seed each.

This script re-runs both configs across 4 seeds each (42, 123, 2024, 7) as real Vertex
custom training jobs via the same cloudlayer.submit_training() path — same spot
instance, same submit -> wait -> download -> log pipeline as the main sweep (reusing
its run_trial() directly). Logged to the SAME MLflow experiment (itcs355-lab2-remote)
but tagged seed_variance_check=true so these 8 runs don't mix with the main 12 when
scripts/compare_runs.py ranks by val_roc_auc.

If the standard deviation across seeds for either config is larger than the 0.0012
gap, the sweep's ranking between trial-01 and trial-07 is not distinguishable from
seed noise.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import mlflow

from src import config, costs
from src.train import git_commit
from src.tune_remote import _git_short_sha, run_trial

CONFIGS = {
    "trial-01": {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 5},
    "trial-07": {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 5},
}
SEEDS = [42, 123, 2024, 7]
MACHINE_TYPE = "n1-standard-4"
EXPERIMENT = "itcs355-lab2-remote"
CHECKPOINT = Path("reports/seed_variance_checkpoint.json")


def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return {"completed": {}}


def save_checkpoint(state: dict) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(state, indent=2))


def main() -> None:
    cfg = config.load()
    from cloudlayer.factory import get_adapter
    adapter = get_adapter(cfg)
    rate = costs.hourly_rate(cfg.provider, MACHINE_TYPE, spot=True)

    trials = [(name, params, seed) for name, params in CONFIGS.items() for seed in SEEDS]
    print(f"Seed variance check: {len(trials)} trials "
          f"({len(CONFIGS)} configs x {len(SEEDS)} seeds), {MACHINE_TYPE} spot "
          f"(~{rate:.2f} THB/hr, est. ~{len(trials) * 0.15:.2f} THB total)")

    image_uri = adapter.push_image(f"itcs355-lab1:{_git_short_sha()}")
    print(f"image: {image_uri}")

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    state = load_checkpoint()
    results: dict[str, list[dict]] = {name: [] for name in CONFIGS}

    for i, (name, params, seed) in enumerate(trials):
        key = json.dumps({"config": name, **params, "seed": seed}, sort_keys=True)
        if key in state["completed"]:
            print(f"{name} seed={seed}: already done, skipping (resumed from checkpoint)")
            results[name].append(state["completed"][key])
            continue

        print(f"{name} seed={seed}: submitting {params} ...")
        try:
            outcome = run_trial(adapter, cfg, image_uri, params, seed, MACHINE_TYPE, i + 100)
        except Exception as exc:
            print(f"{name} seed={seed}: FAILED ({exc!r}) — not marked complete, will retry on next run")
            continue

        metrics = outcome["metrics"]
        duration_h = outcome["wall_s"] / 3600.0
        trial_cost = duration_h * rate

        with mlflow.start_run(run_name=f"seedcheck-{name}-seed{seed}"):
            mlflow.log_params({**params, "seed": seed, "instance": MACHINE_TYPE, "spot": True})
            mlflow.log_metrics({
                "val_roc_auc": metrics["val_roc_auc"], "val_pr_auc": metrics["val_pr_auc"],
                "test_roc_auc": metrics["test_roc_auc"], "test_pr_auc": metrics["test_pr_auc"],
                "duration_s": round(outcome["wall_s"], 3),
                "cost_thb": round(trial_cost, 4),
            })
            mlflow.set_tags({
                "git_commit": git_commit(),
                "data_fingerprint": metrics.get("data_fingerprint", ""),
                "lab": "2",
                "training_job_id": outcome["job_id"],
                "training_target": "vertex-custom-job",
                "seed_variance_check": "true",
                "config_label": name,
            })

        record = {"val_roc_auc": metrics["val_roc_auc"], "test_roc_auc": metrics["test_roc_auc"],
                   "cost_thb": trial_cost}
        state["completed"][key] = record
        save_checkpoint(state)
        results[name].append(record)
        print(f"{name} seed={seed}: val_roc_auc={metrics['val_roc_auc']:.4f} "
              f"test_roc_auc={metrics['test_roc_auc']:.4f} cost={trial_cost:.4f} THB")

    print("\n=== Seed variance summary ===")
    summary: dict[str, dict] = {}
    for name, recs in results.items():
        if len(recs) < 2:
            print(f"{name}: only {len(recs)} result(s) logged — not enough to compute std")
            continue
        val_vals = [r["val_roc_auc"] for r in recs]
        test_vals = [r["test_roc_auc"] for r in recs]
        summary[name] = {
            "val_mean": statistics.mean(val_vals), "val_std": statistics.stdev(val_vals),
            "test_mean": statistics.mean(test_vals), "test_std": statistics.stdev(test_vals),
        }
        print(f"{name} (n={len(recs)}): val_roc_auc mean={summary[name]['val_mean']:.4f} "
              f"std={summary[name]['val_std']:.4f} | test_roc_auc mean={summary[name]['test_mean']:.4f} "
              f"std={summary[name]['test_std']:.4f}")

    if "trial-01" in summary and "trial-07" in summary:
        gap = abs(summary["trial-01"]["val_mean"] - summary["trial-07"]["val_mean"])
        max_std = max(summary["trial-01"]["val_std"], summary["trial-07"]["val_std"])
        print(f"\nval_roc_auc gap between trial-01 and trial-07 means: {gap:.4f}")
        print(f"largest seed std among the two configs: {max_std:.4f}")
        if max_std > gap:
            print("std > gap: the main sweep's ranking is NOT distinguishable from seed noise.")
        else:
            print("gap > std: the main sweep's ranking may reflect a real difference, not just noise.")


if __name__ == "__main__":
    main()
