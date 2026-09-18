"""Lab 2 Task 5 — prove the registered model can be reloaded from the registry, not a
local file, and still scores real rows.

Run:  python scripts/reload_check.py
      python scripts/reload_check.py --model <resource-id-or-full-name> --version <alias-or-id>

Pulls the model straight from Vertex AI Model Registry: GcpAdapter.describe_model()
resolves --model@--version (default: the Task 4 model, alias "production") to its
artifact_uri and lineage, then adapter.download() fetches the model bytes fresh from
GCS into a throwaway temp file every run — never a previously-cached local copy. The
temp file is deleted as soon as the model is loaded into memory.

This is the lab's quiet test: a model that fails to unpickle here, six months from now,
almost always fails because of a custom class defined in the training script rather
than an installed package — see the except block below.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.metrics import roc_auc_score

from cloudlayer.factory import get_adapter
from src import config, data, seeds

DEFAULT_MODEL = "projects/126202218664/locations/asia-southeast1/models/437231793901404160"
DEFAULT_VERSION = "production"
TOLERANCE = 0.0010  # matches reports/lab2-comparison.md and the README's n_jobs=-1 correction


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ITCS355 Lab 2 Task 5 — reload check")
    p.add_argument("--model", "--name", dest="model", default=DEFAULT_MODEL,
                   help="Full Vertex Model resource name, or a bare numeric model ID.")
    p.add_argument("--version", default=DEFAULT_VERSION, help="Version ID or alias.")
    p.add_argument("--n-rows", type=int, default=5)
    return p.parse_args()


def _load_model(adapter, artifact_uri: str):
    with tempfile.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / "model.joblib"
        adapter.download(f"{artifact_uri}/model.joblib", str(local_path))
        try:
            return joblib.load(local_path)
        except (ModuleNotFoundError, AttributeError) as exc:
            print(f"UNPICKLING FAILED: {exc!r}")
            print("This is almost always a custom class defined in the training script "
                  "(e.g. src/train.py) that isn't importable in this environment — check "
                  "whether the class/module this error names lives in project code rather "
                  "than an installed package, and whether it's on sys.path here.")
            raise
        # local_path is deleted on TemporaryDirectory exit regardless — nothing cached.


def main() -> int:
    args = parse_args()
    cfg = config.load()
    adapter = get_adapter(cfg)

    model_ref = args.model if args.model.startswith("projects/") \
        else f"projects/{cfg.project_id}/locations/{cfg.region}/models/{args.model}"
    entry = adapter.describe_model(model_ref, args.version)
    print(f"Pulled from registry: {entry['name']} (version {entry['version_id']}, "
          f"aliases {entry['version_aliases']})")
    print(f"artifact_uri: {entry['artifact_uri']}")

    model = _load_model(adapter, entry["artifact_uri"])
    print("Model unpickled successfully.\n")

    seed = seeds.set_all(int(entry["lineage"]["seed"]))
    df = data.load_raw(cfg.raw_path)
    _, val_df, test_df = data.split(df, seed=seed)

    sample = test_df.head(args.n_rows)
    proba = model.predict_proba(sample[data.FEATURES])[:, 1]
    print(f"Scored {len(sample)} held-out test rows (freshly reloaded model):")
    for (idx, row), p in zip(sample.iterrows(), proba):
        print(f"  row {idx}: predicted_proba={p:.4f}  actual={row[data.TARGET]}")

    val_auc = roc_auc_score(val_df[data.TARGET], model.predict_proba(val_df[data.FEATURES])[:, 1])
    test_auc = roc_auc_score(test_df[data.TARGET], model.predict_proba(test_df[data.FEATURES])[:, 1])
    logged_val = float(entry["lineage"]["metric_val"])
    logged_test = float(entry["lineage"]["metric_test"])
    val_diff = abs(val_auc - logged_val)
    test_diff = abs(test_auc - logged_test)

    print(f"\nFull held-out set, reloaded model vs. registry-logged metrics "
          f"(tolerance {TOLERANCE}):")
    print(f"  val_roc_auc:  reloaded={val_auc:.6f}  logged={logged_val:.6f}  "
          f"diff={val_diff:.6f}  {'PASS' if val_diff <= TOLERANCE else 'FAIL'}")
    print(f"  test_roc_auc: reloaded={test_auc:.6f}  logged={logged_test:.6f}  "
          f"diff={test_diff:.6f}  {'PASS' if test_diff <= TOLERANCE else 'FAIL'}")

    return 0 if max(val_diff, test_diff) <= TOLERANCE else 1


if __name__ == "__main__":
    raise SystemExit(main())
