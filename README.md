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
because the seed also reshuffles which machines land in which split. The five runs below hold the
seed fixed and vary the model instead, which is why their spread (0.8417–0.8543) is wider than the
claimed tolerance without contradicting it.

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
| baseline | 200 | 8 | 5 | 0.8364 | 0.8483 |
| shallow-depth-4 | 200 | 4 | 5 | 0.8405 | 0.8543 |
| deep-depth-16 | 200 | 16 | 5 | 0.8361 | 0.8417 |
| more-trees-500 | 500 | 8 | 5 | 0.8386 | 0.8482 |
| regularized-leaf-20 | 200 | 8 | 20 | 0.8453 | 0.8507 |

`max_depth=16` overfits relative to the shallower trees; `min_samples_leaf=20` recovers most of
that gap through regularisation instead. `max_depth=4` generalises best in this study — the
production default (`max_depth=8`, the Dockerfile `CMD`) trades a little of that for a model less
sensitive to which machines happen to fall in the training split.

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

## Notes for the grader

`make reproduce` runs entirely offline against the DVC-tracked, deterministically-generated
dataset — no cloud credentials required for that command. `make image-push` and `dvc push`
require `gcloud auth login` as `jinnaput.jaiphoom@gmail.com` against project `itcs355-6688033`;
the Artifact Registry repo and GCS bucket are provisioned under that project's default region
(`asia-southeast1`). The RandomForest is fit with `n_jobs=-1`; scikit-learn seeds each tree from
the master `random_state` independently of thread scheduling, so this does not affect
determinism.

---

## Checklist before you submit

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
