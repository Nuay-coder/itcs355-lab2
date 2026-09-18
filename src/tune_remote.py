"""Lab 2 Task 2 — budgeted hyperparameter study on managed (Vertex AI) training jobs.

Run:  python -m src.tune_remote --trials 12                 # estimate only, no cloud calls
      python -m src.tune_remote --trials 12 --execute        # push image, submit real jobs

Unlike src/tune.py, which fits models in-process and only *estimates* what that would
cost on real infra, this script drives actual cloudlayer.submit_training() calls — one
real Vertex AI custom training job per trial, on spot capacity. Only the adapter
interface is used here (submit_training, wait_training, download, push_image); nothing
in this module imports a provider SDK.

Each trial's metrics are computed by src/train.py inside the container and pushed to
BLOB_URI by entrypoint.sh (keyed by TRIAL_KEY, set via the `trial_key` arg below), then
pulled back here with adapter.download() for MLflow logging — no log-scraping.

The budget is enforced, not advisory, exactly like src/tune.py: the loop stops
submitting new trials once projected spend would exceed --budget-thb. A trial that
fails or is preempted (spot capacity can vanish mid-run) is simply never marked
complete in the checkpoint, so re-running this script resumes it — an interruption
costs one trial's wall time, not the sweep.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path

import mlflow

from src import config, costs, seeds
from src.train import git_commit
from src.tune import SEARCH_SPACE, grid


def estimate_cost(n_trials: int, hourly_rate_thb: float, minutes_per_trial: float) -> float:
    """Pre-flight estimate, in THB. Vertex bills for the whole job wall clock —
    provisioning plus training, not just active compute — so minutes_per_trial should
    come from an observed job (see --minutes-per-trial), not the model fit's own time.
    """
    return n_trials * hourly_rate_thb * (minutes_per_trial / 60.0)


def _git_short_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=config.REPO_ROOT,
        )
        return out.stdout.strip()
    except Exception:
        return "dev"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ITCS355 Lab 2 Task 2 — remote budgeted study")
    p.add_argument("--trials", type=int, default=12, help="minimum 12 for the lab")
    p.add_argument("--max-trials", type=int, default=None,
                    help="Cap how many NEW trials THIS invocation submits — for smoke-testing "
                         "the submit/wait/download/log/checkpoint pipeline on 1-2 trials before "
                         "committing to the full sweep. Already-completed trials from the "
                         "checkpoint don't count against this; re-run without it (or raise it) "
                         "to submit the rest.")
    p.add_argument("--budget-thb", type=float, default=150.0)
    p.add_argument("--machine-type", default="n1-standard-4")
    p.add_argument("--minutes-per-trial", type=float, default=5.0,
                    help="Estimate only — not a timeout. Tune this from an observed run.")
    p.add_argument("--seed", type=int, default=seeds.DEFAULT_SEED)
    p.add_argument("--experiment", default="itcs355-lab2-remote")
    p.add_argument("--checkpoint", type=Path, default=Path("reports/tune_remote_checkpoint.json"))
    p.add_argument("--image", default="itcs355-lab1")
    p.add_argument("--tag", default=None, help="Defaults to the current git short SHA.")
    p.add_argument("--execute", action="store_true",
                    help="Actually push the image and submit Vertex jobs. Without this "
                         "flag, only the cost estimate is printed.")
    return p.parse_args()


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"completed": [], "spent_thb": 0.0}


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def run_trial(adapter, cfg, image_uri: str, params: dict, seed: int, machine_type: str, i: int) -> dict:
    trial_key = f"lab2-sweep-{i:02d}-{uuid.uuid4().hex[:8]}"
    container_args = {
        **params,
        "seed": seed,
        "metrics-out": "reports/metrics.json",
        "machine_type": machine_type,
        "use_spot": True,
        "trial_key": trial_key,
    }
    started = time.perf_counter()
    job_id = adapter.submit_training(image_uri, container_args)
    result = adapter.wait_training(job_id)  # raises RuntimeError on FAILED / CANCELLED
    wall_s = time.perf_counter() - started

    local_metrics = Path(f"reports/_remote_trial_{i:02d}_metrics.json")
    adapter.download(f"{cfg.blob_uri}/training-jobs/{trial_key}/metrics.json", str(local_metrics))
    metrics = json.loads(local_metrics.read_text())
    local_metrics.unlink(missing_ok=True)

    return {"job_id": job_id, "wall_s": wall_s, "metrics": metrics, **result}


def main() -> None:
    args = parse_args()
    cfg = config.load()
    from cloudlayer.factory import get_adapter
    adapter = get_adapter(cfg)
    seed = seeds.set_all(args.seed)

    candidates = grid(SEARCH_SPACE)[: args.trials]
    this_run = min(args.max_trials, len(candidates)) if args.max_trials is not None else len(candidates)
    rate = costs.hourly_rate(cfg.provider, args.machine_type, spot=True)
    estimate = estimate_cost(this_run, rate, args.minutes_per_trial)

    print(f"Planned trials:      {len(candidates)} total  (varying {', '.join(SEARCH_SPACE)})"
          + (f" — capped to {this_run} this invocation via --max-trials" if args.max_trials is not None else ""))
    print(f"Instance:            {args.machine_type} spot "
          f"(~{rate:.2f} THB/hr, {costs.SPOT_FACTOR:.0%} of on-demand)")
    print(f"Assumed wall time:   {args.minutes_per_trial:.1f} min/trial "
          "(observed warm-start Vertex job: ~3.3 min; first job of a session runs "
          "colder, +5-10 min)")
    print(f"Estimated cost:      {estimate:.2f} THB for {this_run} trial(s)   "
          f"(budget: {args.budget_thb:.2f} THB)")
    print(f"Estimated wall time: ~{this_run * args.minutes_per_trial:.0f} min "
          "(sequential — one job submitted at a time)")
    if estimate > args.budget_thb:
        print("WARNING: estimate exceeds budget — reduce --trials/--max-trials or re-check --minutes-per-trial.")

    if not args.execute:
        print("\nDry run only — no image was pushed and no job was submitted. "
              "Pass --execute to run the sweep for real.")
        return

    tag = args.tag or _git_short_sha()
    image_uri = adapter.push_image(f"{args.image}:{tag}")
    print(f"\nimage: {image_uri}")

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment)

    state = load_checkpoint(args.checkpoint)
    skipped: list[dict] = []
    run_count = 0

    for i, params in enumerate(candidates):
        key = json.dumps(params, sort_keys=True)
        if key in state["completed"]:
            print(f"trial {i}: already done, skipping (resumed from checkpoint)")
            continue
        if state["spent_thb"] >= args.budget_thb:
            skipped.append(params)
            continue
        if args.max_trials is not None and run_count >= args.max_trials:
            remaining = len(candidates) - i
            print(f"trial {i}: --max-trials={args.max_trials} reached for this invocation, "
                  f"stopping ({remaining} configuration(s) left — re-run to continue)")
            break

        print(f"trial {i}: submitting {params} ...")
        try:
            outcome = run_trial(adapter, cfg, image_uri, params, seed, args.machine_type, i)
        except Exception as exc:
            run_count += 1  # counts against --max-trials: a failed attempt still used a submission
            print(f"trial {i}: FAILED ({exc!r}) — not marked complete, will retry on next run")
            continue

        metrics = outcome["metrics"]
        duration_h = outcome["wall_s"] / 3600.0
        trial_cost = duration_h * rate
        state["spent_thb"] += trial_cost

        with mlflow.start_run(run_name=f"remote-trial-{i:02d}"):
            mlflow.log_params({**params, "seed": seed, "instance": args.machine_type, "spot": True})
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
            })

        state["completed"].append(key)
        save_checkpoint(args.checkpoint, state)
        run_count += 1
        print(f"trial {i}: val_roc_auc={metrics['val_roc_auc']:.4f} "
              f"cost={trial_cost:.4f} THB  cumulative={state['spent_thb']:.4f}")

    print(f"\nspent {state['spent_thb']:.4f} of {args.budget_thb} THB")
    if skipped:
        print(f"BUDGET EXHAUSTED — {len(skipped)} configurations not run:")
        for s in skipped:
            print(f"  {s}")


if __name__ == "__main__":
    main()
