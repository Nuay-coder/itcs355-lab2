"""Lab 2 Task 4 — register the chosen model in the provider's Model Registry.

Run:  python scripts/register_best_model.py

Context: the winning run (MLflow run 2017ed98238142c89798d0db9b046bee, from the Task 2
remote sweep) trained inside an ephemeral Vertex custom job. Its container logged
metrics to a local sqlite MLflow store and uploaded metrics.json to BLOB_URI, but never
uploaded the fitted model itself anywhere durable — that store, and the model object in
it, no longer exist once the job's compute was torn down.

RandomForestClassifier with a fixed random_state is fully deterministic, and every
other input is pinned (git commit, data fingerprint, seed, hyperparameters). So rather
than register a placeholder, this script reconstructs the exact same model: same code
at the same commit, same data, same seed, same params. It refuses to proceed unless the
reconstructed val/test metrics match the recorded ones exactly — that check is the
actual proof of reproducibility this lab is about, not a formality.

LINEAGE below is fixed and verified against the real MLflow run — not recomputed here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from cloudlayer.factory import get_adapter
from src import config, data, seeds
from src.train import git_commit

PARAMS = {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 5}

LINEAGE = {
    "git_commit": "234c6e376c873efa10cc953ee27a818866483f2e",
    "data_version": "422cccb9136e8140",
    "mlflow_run_id": "2017ed98238142c89798d0db9b046bee",
    "training_job_id": "projects/126202218664/locations/asia-southeast1/customJobs/2529236009309175808",
    "image_digest": "asia-southeast1-docker.pkg.dev/itcs355-6688033/itcs355/itcs355-lab1@sha256:a2db0f9a7ed4ecc1353fbe8c14e0c9cb94894d43740ff957b3f1f436938ac114",
    "seed": "20260101",
    "metric_val": "0.84259989736441",
    "metric_test": "0.8532857870606215",
}
MODEL_NAME = "itcs355-lab2-sensor-risk"


def main() -> int:
    cfg = config.load()

    commit = git_commit()
    if commit != LINEAGE["git_commit"]:
        print(f"REFUSING to register: local git_commit {commit} != recorded {LINEAGE['git_commit']}")
        return 1

    fingerprint = data.data_fingerprint(cfg.raw_path)
    if fingerprint != LINEAGE["data_version"]:
        print(f"REFUSING to register: local data_fingerprint {fingerprint} != recorded {LINEAGE['data_version']}")
        return 1

    seed = seeds.set_all(int(LINEAGE["seed"]))
    df = data.load_raw(cfg.raw_path)
    train_df, val_df, test_df = data.split(df, seed=seed)

    model = RandomForestClassifier(random_state=seed, n_jobs=-1, **PARAMS)
    model.fit(train_df[data.FEATURES], train_df[data.TARGET])

    val_auc = roc_auc_score(val_df[data.TARGET], model.predict_proba(val_df[data.FEATURES])[:, 1])
    test_auc = roc_auc_score(test_df[data.TARGET], model.predict_proba(test_df[data.FEATURES])[:, 1])

    # Not bit-exact: predict_proba averages per-tree probabilities under n_jobs=-1, and
    # thread-scheduling changes summation order across runs (same class of float noise
    # the main README documents for cross-machine reproduction, +/-0.0010 there). Tree
    # structure itself is fixed by random_state; this is float noise, not a different
    # model, so the same tolerance applies rather than requiring a bit-exact match.
    TOLERANCE = 0.0010
    val_diff = abs(val_auc - float(LINEAGE["metric_val"]))
    test_diff = abs(test_auc - float(LINEAGE["metric_test"]))
    if val_diff > TOLERANCE or test_diff > TOLERANCE:
        print("REFUSING to register: reconstructed metrics differ from the recorded run "
              f"by more than the {TOLERANCE} tolerance.")
        print(f"  val:  reconstructed={val_auc!r}  recorded={LINEAGE['metric_val']}  diff={val_diff:.6f}")
        print(f"  test: reconstructed={test_auc!r}  recorded={LINEAGE['metric_test']}  diff={test_diff:.6f}")
        return 1
    print(f"Reconstructed model matches recorded metrics within {TOLERANCE} "
          f"(val diff={val_diff:.6f}, test diff={test_diff:.6f}) — safe to register.")

    local_path = Path("reports/_register_model.joblib")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, local_path)

    adapter = get_adapter(cfg)
    artifact_key = f"models/lab2/{LINEAGE['mlflow_run_id']}/model.joblib"
    artifact_uri = adapter.upload(str(local_path), artifact_key)
    model_dir_uri = artifact_uri.rsplit("/", 1)[0]
    print(f"uploaded model artifact: {artifact_uri}")
    local_path.unlink()

    version = adapter.register_model(model_dir_uri, MODEL_NAME, **LINEAGE)
    print(f"registered (staging): {version}")

    model_name, _, version_id = version.rpartition("@")
    promoted = adapter.promote_model(model_name, version_id)
    print(f"promoted (production): {promoted}")

    # Read the entry straight back from Vertex rather than trusting what we just sent —
    # this caught metadata= silently not persisting on the first attempt.
    stored = adapter.describe_model(model_name, version_id)
    if stored["lineage"] != LINEAGE:
        print("WARNING: lineage read back from Vertex does not match what was sent.")
        return 1

    print(f"\nVerified registry entry (read back from Vertex, version {stored['version_id']}):")
    print(f"  model:            {stored['name']}")
    print(f"  display_name:     {stored['display_name']}")
    print(f"  version_aliases:  {stored['version_aliases']}")
    print(f"  artifact_uri:     {stored['artifact_uri']}")
    print(f"  labels:           {stored['labels']}")
    print("  lineage (version_description, JSON, verified byte-exact against the input):")
    for k, v in stored["lineage"].items():
        print(f"    {k} = {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
