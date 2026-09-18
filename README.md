# ITCS355 Lab 1 — Reproducible Training

> **Course materials live in [`course/`](course/README.md)** — syllabus, slides, the faculty
> specification, all five lab handouts, and the project brief. Every document is Markdown and
> renders on GitHub, diagrams included. New to the repo? Start with the
> [portability reference](course/reference/cloud-portability-reference.md).
> Keep this block when you edit the rest of this file; it is not part of the Lab 1 deliverable.

Predicting machine failure within 7 days from sensor readings. The model is not the point;
whether a stranger can reproduce it is.

> **This README is graded.** A grader with Docker and nothing else from your setup runs one
> command and compares the result against the claim below.

---

## Reproduce

```bash
make reproduce
```

expected test_roc_auc: 0.8483 ± 0.0010

Runtime: about 40 seconds on 4 cores. No cloud account or credentials needed for this command —
that is deliberate, and it is why a grader can run it.

This tolerance covers cross-machine floating-point noise at a **fixed** seed (different CPUs sum
floats in different orders inside the same BLAS routine) — not seed-to-seed variance. `make
reproduce` pins the seed, so seed sensitivity is not the right thing to size the tolerance against;
sweeping the seed instead of the hyperparameters moves `test_roc_auc` by far more than this,
because the seed also reshuffles which machines land in which split. The five runs below hold the seed fixed and vary the model instead, which is why their spread
(0.8433–0.8532) is wider than the claimed tolerance without contradicting it.
---

## The problem

240 machines, 25 readings each, 6 sensor features, binary target `failed_within_7d` with a
positive rate near 12%.

Machines have persistent characteristics — a hot-running machine reads hot in every row. So the
train/validation/test split is **grouped by `machine_id`**: every reading from one machine lands
in exactly one partition. Splitting row-wise instead lets the model memorise the machine and
reports a validation score that will never survive production. `tests/test_data.py` asserts this
property holds, and Lab 4 turns it into a CI gate.

Bringing your own dataset is allowed. Replace `scripts/make_dataset.py`, update the schema in
`src/data.py`, and keep every test passing.

---

## Layout

```
src/          Layer 1 — provider-neutral. No SDKs, no bucket names, no absolute paths.
cloudlayer/   Layer 3 — the only place a provider SDK may be imported.
scripts/      Dataset generation, cloud check, portability audit, metric verification.
tests/        Data contract tests and split property tests.
```

`src/config.py` is the single point of environment knowledge. Everything else reads from it.
`make portability-audit` enforces the rule; it fails the build if a provider string appears in
`src/` or `tests/`.

---

## Setup

```bash
cp cloud.env.example cloud.env      # fill in, never commit
make setup
make cloud-check                    # eight slots, all PASS
make data                           # generate the dataset
make test                           # all tests passing
```

Post your `make cloud-check` output in the course channel before Session 1.

---

## Provider: GCP

`CLOUD_PROVIDER=gcp`. The adapter (`cloudlayer/gcp.py`) implements the three Lab 1 methods:

- `upload` / `download` — `google-cloud-storage`, against `BLOB_URI` (`gs://itcs355-6688033/itcs355`)
- `push_image` — `gcloud auth configure-docker`, then `docker push`, returning the
  digest-pinned reference (`registry/repo@sha256:...`), not a tag

```bash
make image-push        # image reaches Artifact Registry, digest-pinned
dvc init && dvc remote add -d storage ${BLOB_URI}/dvc
dvc add data/raw && dvc push
```

Pushed image (verified pullable with `docker pull --platform linux/amd64 <ref>` from a
different machine architecture than it was built on):

```
asia-southeast1-docker.pkg.dev/itcs355-6688033/itcs355/itcs355-lab1@sha256:4417e4d00d724f4cbb38fc64350b3e789fc6b3d7d8000696d985a50dad5ab9b3
```

`dvc push` completed against `gs://itcs355-6688033/itcs355/dvc`; `dvc pull` round-trips
`data/raw/sensors.csv` back to fingerprint `422cccb9136e8140`.

Five tracked runs vary the model, not the seed (`itcs355-lab1` experiment, seed `20260101`
throughout):

| run | n_estimators | max_depth | min_samples_leaf | val_roc_auc | test_roc_auc |
|---|---|---|---|---|---|
| shallow-100-4 | 100 | 4 | 5 | 0.8426 | 0.8532 |
| baseline | 200 | 8 | 5 | 0.8364 | 0.8483 |
| compact-150-6 | 150 | 6 | 5 | 0.8433 | 0.8494 |
| wide-250-10 | 250 | 10 | 5 | 0.8394 | 0.8438 |
| deep-300-12 | 300 | 12 | 5 | 0.8357 | 0.8433 |

Test performance falls roughly monotonically as depth grows alongside tree count —
`max_depth=4`/100 trees generalises best in this study (0.8532), while `max_depth=12`/300 trees
drops to 0.8433, suggesting depth is the more overfitting-prone knob here even when paired with
more trees to average over. The production default (`max_depth=8`, `n_estimators=200`, the
Dockerfile `CMD`) sits in the middle of that trend.
---

## Reproducibility trade-off

The digest pin is what I would drop first. Hashed dependencies and the digest pin both guard
against something changing under me without a commit to blame; between the two, the digest pin is
the narrower, cheaper guarantee to lose, because `pip install --require-hashes` already fails
loudly and immediately if a wheel doesn't match — the same failure mode I'd be trading away, just
one layer up. Losing the digest pin means `python:3.11-slim` can move between my build and the
grader's with nothing to bisect. Losing the seed is worse than either: it doesn't just cost
comparability between runs, it silently changes which machines fall in each split, which is a
correctness bug wearing a reproducibility costume.

---

## Checklist before you submit — Lab 1

- [x] `make reproduce` works from a fresh clone, on a machine that is not yours
- [x] `make verify` passes against your claim line
- [x] `make test` — all tests pass
- [x] `make portability-audit` — clean
- [x] Image builds for `linux/amd64` and is pushed, digest-pinned
- [x] `dvc push` completed; a grader can `dvc pull`
- [x] Five or more tracked runs with params, metrics, data fingerprint, and commit SHA
- [x] Every submission placeholder above is filled in (the course-materials block at the top stays)
- [x] `git log -p | grep -i -E "secret|password|AKIA|BEGIN PRIVATE"` returns nothing

That last check is not optional. A credential in Git history is an automatic deduction in this
course, and rotating it is your responsibility, not the grader's.

---

## Lab 2 — Experiment Tracking and Model Registry

- Moved training onto Vertex AI managed compute (`submit_training`/`wait_training`,
  `cloudlayer/gcp.py`, `aiplatform_v1` — the low-level client, not the convenience
  wrapper). Getting the first submission through surfaced the submit-time vs. run-time
  identity gap the lab warns about: the API was disabled on the project, and the job's
  run-time identity needs a real service account, not the user account that submits
  it — `cloud.env` now carries `TRAINING_SERVICE_ACCOUNT` separately from
  `IDENTITY_REF` for that reason. The job reads/writes only through `BLOB_URI`
  (`entrypoint.sh` runs `dvc pull` before training if the data isn't already present).
- Ran a 12-trial hyperparameter sweep (`n_estimators`, `max_depth`, `min_samples_leaf`)
  on spot compute, checkpointed and resumable — proven for real, not just in theory,
  when two trials hit a network error mid-sweep and a plain rerun picked them up
  cleanly from the checkpoint.
- Followed up with a seed-variance check on the sweep's top two candidates: the
  seed-to-seed spread turned out to be several times larger than the gap that had
  separated them on a single seed, so that ranking wasn't real signal.
- Registered the selected model in Vertex AI Model Registry with full lineage, and
  promoted it through a staging step (below).
- Proved the registered model reloads and scores correctly straight from the registry —
  no local file, no cache.

**Cost breakdown**

| Source | Jobs | Cost (THB) |
|---|---:|---:|
| Task 2 main sweep (MLflow-tracked, spot) | 12 | 1.7858 |
| Task 3 seed-variance check (MLflow-tracked, spot) | 8 | 1.1396 |
| Task 1 troubleshooting job (on-demand, `SERVICE_DISABLED`-era failure) | 1 | 1.8468 |
| Task 1 final validation job (on-demand) | 1 | 0.4140 |
| Orphaned Task 2 trial (succeeded on Vertex, never logged) | 1 | 0.1419 |
| **Total** | **23** | **5.3281** |

**5.3281 of 150 THB budget.** Reconciled directly against every real Vertex job
(`gcloud ai custom-jobs list`) cross-checked against MLflow's logged `cost_thb` — not
the sweep scripts' own running totals alone, which caught one succeeded-but-never-logged
job and don't cover the two Task 1 jobs that predate cost
logging. Model Registry storage and API calls from Task 4/5 are not
separately itemized — a few KB of GCS storage and free-tier API calls, below the
precision this reconciliation is tracking at.

Trial-by-trial numbers, the full comparison table, and the model-selection
justification are in [`reports/lab2-comparison.md`](reports/lab2-comparison.md).

### Task 4 — Model Registry

Registered via `cloudlayer.gcp.GcpAdapter.register_model()` (Vertex AI Model Registry,
`aiplatform_v1.ModelServiceClient` — the lower-level client, same style as
`submit_training`/`wait_training`) by running `python scripts/register_best_model.py`.

**Version string:** `projects/126202218664/locations/asia-southeast1/models/437231793901404160@1`

**The artifact sitting at `artifact_uri` is not the original job's own output.** That job's compute
had already been torn down by the time Task 4 started, and its container's MLflow store —
including the fitted model it produced — was ephemeral and never uploaded anywhere durable, so
there was no original file left to register. `register_best_model.py` retrains a fresh model
instead — same git commit, same data fingerprint, same seed, same hyperparameters as the original
run — uploads *that* retrained model to `artifact_uri` itself, and only calls `register_model()` if
the retrained val/test metrics land within 0.0010 of the ones already recorded in MLflow for the
original run. They did, to within 0.00006 (see the correction above): a `RandomForestClassifier`
with a fixed `random_state` reproduces the same model, not bit-for-bit, but within that verified
tolerance.

`training_job_id` and `mlflow_run_id` below still identify the *original* job and run — the one
whose metrics justified selecting this configuration in Task 3 — not the source of the registered
artifact's bytes. Those are deliberately different facts: which run decided this was the model to
register, versus which run produced the specific file at `artifact_uri` today.

Lineage (all eight fields, read back from Vertex and verified byte-exact against the input — not
just what was sent, since `Model.metadata` turned out to silently not persist for a
custom-uploaded model and had to be moved to `version_description` as JSON instead):

| field | value |
|---|---|
| `git_commit` | `234c6e376c873efa10cc953ee27a818866483f2e` |
| `data_version` | `422cccb9136e8140` |
| `mlflow_run_id` | `2017ed98238142c89798d0db9b046bee` |
| `training_job_id` | `projects/126202218664/locations/asia-southeast1/customJobs/2529236009309175808` |
| `image_digest` | `asia-southeast1-docker.pkg.dev/itcs355-6688033/itcs355/itcs355-lab1@sha256:a2db0f9a7ed4ecc1353fbe8c14e0c9cb94894d43740ff957b3f1f436938ac114` |
| `seed` | `20260101` |
| `metric_val` | `0.84259989736441` |
| `metric_test` | `0.8532857870606215` |

The model-selection justification (why this run and not the sweep's raw highest-scoring one) is in
[`reports/lab2-comparison.md`](reports/lab2-comparison.md).

Promoted through a staging step via `GcpAdapter.promote_model()`: registered with alias `staging`,
then moved to `production` (Vertex's `MergeVersionAliases`, adding `production` and dropping
`staging` in one call).

#### Who should be allowed to promote, and on what evidence

Not the person who trained or tuned the model. In a real organisation, promotion from staging to
production should sit with a role that has no stake in the model looking good — an ML platform or
MLOps lead, or a designated model-risk approver — separate from the data scientist who built it,
the same segregation-of-duties reasoning that keeps code review separate from the author's own
sign-off. For anything with regulatory or safety exposure, promotion should require two sign-offs,
not one.

Before approving, that person should require, in writing (a PR, ticket, or registry comment —
something that outlives Vertex's own mutable alias state):

1. **Reproducibility proof** — exactly what this section demonstrates: pinned git commit, data
   version, and seed reconstruct the claimed metrics within a stated tolerance. A model that can't
   be rebuilt from its own lineage record is not eligible for promotion, full stop.
2. **A challenger comparison against the current production model on the same held-out data** —
   training-time val/test metrics are not enough; the candidate has to be evaluated against what
   it would actually replace.
3. **Seed/variance evidence that the improvement is real, not noise** — Task 3's seed-variance
   check on this same sweep found two configurations 0.0012 apart on `val_roc_auc` with a
   seed-to-seed standard deviation nearly 7x larger than that gap. A promotion decided on a single
   seed's ranking, with no variance check on record, should be rejected on that basis alone.
4. **Cost impact** — training cost, and the retraining cadence it implies, reviewed against the
   budget it will actually run under in production (not the lab budget).
5. **A monitoring and rollback plan already wired up** — the alias this model is being promoted to
   swaps atomically, but someone has to be watching the metric that would tell you to swap back.

---

## Checklist before you submit — Lab 2

- [x] `submit_training()` and `wait_training()` implemented and working
- [x] 12+ trials on discounted compute, checkpointed, all tracked
- [x] Interruption survived and resumed — evidence in logs
- [x] Comparison artifact in `reports/`
- [x] 200-word justification covering all four required points
- [x] Model registered with all eight lineage fields
- [x] Promotion step performed, with a note on who should own it
- [x] `reload_check.py` runs from the registry and scores rows
- [x] Cost recorded per trial, total under 150 THB
- [ ] `make teardown` run — **attempted, blocked by environment**: `teardown()` isn't implemented in `GcpAdapter` yet 
(it's a Lab 5 deliverable per `cloudlayer/base.py`); `make teardown` fails immediately with `NotImplementedError` 
before any deletion logic runs. 

---

## Notes for the grader

### Lab 1

`make reproduce` runs entirely offline against the DVC-tracked, deterministically-generated
dataset — no cloud credentials required for that command. The RandomForest is fit with
`n_jobs=-1`; scikit-learn seeds each tree from the master `random_state` independently of thread
scheduling, so *tree structure* is unaffected.

### Lab 2

`make image-push` and `dvc push` require `gcloud auth login` as `jinnaput.jaiphoom@gmail.com`
against project `itcs355-6688033`; the Artifact Registry repo and GCS bucket are provisioned under
that project's default region (`asia-southeast1`).

**Correction from Lab 2 Task 4:** `predict_proba` output is not bit-exact across runs under
`n_jobs=-1` — reconstructing the registered model (same commit, same data, same seed) reproduced
`val_roc_auc`/`test_roc_auc` to within 0.00006, not exactly. Averaging per-tree probabilities in
parallel is sensitive to thread-scheduling-dependent summation order; this is float noise, not a
different model, and is well inside the tolerance this README already claims above, but the
determinism claim as originally written was too strong.
