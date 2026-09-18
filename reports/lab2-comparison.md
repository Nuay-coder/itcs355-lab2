# Lab 2 — Run comparison

Experiment `itcs355-lab2-remote` · 12 trials · total spend 1.7858 THB

`thb_per_point` is cost per percentage point of val_roc_auc above the worst trial. Cheap improvements rank low; expensive improvements rank high, however good the headline number is.

Sorted by `val_roc_auc` (`rank_val`). `rank_test` ranks the same trials by `test_roc_auc` instead — when `rank_val` == 1 doesn't line up with `rank_test` == 1, the highest-validation trial is not the highest-test one.

|   rank_val | run_id   |   val_roc_auc |   test_roc_auc |   rank_test |   cost_thb |   n_estimators |   max_depth |   min_samples_leaf |   thb_per_point |
|-----------:|:---------|--------------:|---------------:|------------:|-----------:|---------------:|------------:|-------------------:|----------------:|
|          1 | 2017ed98 |        0.8426 |         0.8533 |           3 |     0.1372 |            100 |           4 |                  5 |          0.085  |
|          2 | 5ff4c2e4 |        0.8424 |         0.8518 |           4 |     0.1762 |            100 |           4 |                  1 |          0.1106 |
|          3 | c26dbdb8 |        0.8411 |         0.8545 |           1 |     0.1573 |            300 |           4 |                  5 |          0.1075 |
|          4 | 38f61b69 |        0.8404 |         0.8537 |           2 |     0.1565 |            300 |           4 |                  1 |          0.1123 |
|          5 | 541c95fa |        0.8397 |         0.8466 |           8 |     0.1374 |            100 |           8 |                  5 |          0.1038 |
|          6 | b1d54d2b |        0.8377 |         0.8491 |           5 |     0.1374 |            300 |           8 |                  5 |          0.1223 |
|          7 | ae89490a |        0.8354 |         0.8431 |           9 |     0.1372 |            300 |          12 |                  5 |          0.1535 |
|          8 | 85fa0a43 |        0.8338 |         0.8478 |           7 |     0.1375 |            300 |           8 |                  1 |          0.1874 |
|          9 | 96366dd5 |        0.8322 |         0.8417 |          10 |     0.1374 |            100 |          12 |                  5 |          0.2394 |
|         10 | 9663690e |        0.8312 |         0.8488 |           6 |     0.1757 |            100 |           8 |                  1 |          0.3708 |
|         11 | 7221bb36 |        0.8268 |         0.8415 |          11 |     0.1378 |            100 |          12 |                  1 |          4.0721 |
|         12 | b3e301bb |        0.8265 |         0.8374 |          12 |     0.1582 |            300 |          12 |                  1 |         41.1976 |

## Seed variance check

8 runs: run 2017ed98 (n_estimators=100, max_depth=4, min_samples_leaf=5) and run c26dbdb8 (n_estimators=300, max_depth=4, min_samples_leaf=5), each re-trained on 4 seeds. Logged in MLflow experiment `itcs355-lab2-remote`, tag `seed_variance_check=true`.

| run_id   | base_run_id | seed |   val_roc_auc |   test_roc_auc |
|:---------|:------------|-----:|--------------:|---------------:|
| 43030840 | 2017ed98    |   42 |         0.8580 |         0.8328 |
| 877626ce | 2017ed98    |  123 |         0.8452 |         0.8569 |
| 7154c2fb | 2017ed98    | 2024 |         0.8532 |         0.8389 |
| a27e3101 | 2017ed98    |    7 |         0.8557 |         0.8526 |
| 26372e97 | c26dbdb8    |   42 |         0.8578 |         0.8352 |
| 3fad0e43 | c26dbdb8    |  123 |         0.8452 |         0.8573 |
| 65561202 | c26dbdb8    | 2024 |         0.8544 |         0.8406 |
| 9e38ba01 | c26dbdb8    |    7 |         0.8582 |         0.8536 |

| base_run_id | val_mean | val_std | test_mean | test_std |
|:------------|---------:|--------:|----------:|---------:|
| 2017ed98    |   0.8530 |  0.0056 |    0.8453 |   0.0113 |
| c26dbdb8    |   0.8539 |  0.0060 |    0.8467 |   0.0105 |

## Which model did you register, and why?

We register 2017ed98 (n_estimators=100, max_depth=4, min_samples_leaf=5), not the sweep's nominal winner on either metric.

1. Why this model rather than the highest-scoring one, if they differ

The main sweep separated 2017ed98 (val_roc_auc 0.8426) from c26dbdb8 (test_roc_auc leader, 0.8545) by 0.0012 on both metrics, each on one seed. A seed-variance check (4 seeds per config) found 2017ed98 val=0.8530 (std 0.0056) and c26dbdb8 val=0.8539 (std 0.0060): the 0.0009 gap between configs is ~6.7x smaller than the within-config seed noise. The two are statistically indistinguishable, so validation score alone cannot justify the pick. With performance tied, cost decides: 2017ed98 costs 0.085 THB per validation point versus c26dbdb8's 0.1075, ~26% cheaper.

2. The variance across seeds for your chosen configuration

Across seeds {42, 123, 2024, 7}: 2017ed98 val_roc_auc std=0.0056 (mean 0.8530), test_roc_auc std=0.0113 (mean 0.8453). This noise exceeds every hyperparameter-driven gap in the original 12-trial, single-seed sweep, so that ranking is not reliable alone.

3. What it costs to train, and to retrain monthly

Cost 0.1372 THB per training run (n1-standard-4, spot). Monthly retraining: ~0.14 THB, negligible against the 150 THB budget.

4. One way this choice could be wrong

The grid covers only 12 sparse combinations, and seed variance was checked on just 2 of them; an untested config could outperform both. All results come from one dataset snapshot and split, and may not generalize under data drift.

