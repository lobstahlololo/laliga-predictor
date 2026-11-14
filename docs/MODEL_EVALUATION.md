# Model evaluation protocol and measured results

This document describes the Phase 6 full-season evaluation and separates
**validation**, **test** and **live** performance. The final test season is
never used to choose a feature, hyperparameter, threshold, ensemble weight or
model.

## 1. Chronological protocol

| Window | Seasons | Rows (current cache) |
|---|---|---:|
| Train | `23/24` + `24/25` | 760 |
| Validation | `25/26` | 380 |
| Untouched test | `26/27` (in progress) | 119 |

- Hyperparameters come from recorded training metadata or declared defaults;
  they are not tuned on the test season.
- Ensemble weights are selected on validation predictions with a fixed simplex
  grid; test predictions are produced once after all choices are fixed.
- Walk-forward folds train only on seasons before their target season. Folds
  targeting `26/27` are excluded from this report.
- Every model is scored on the same rows with the same labels (`Home`, `Draw`,
  `Away`). Market probabilities are row-normalised to remove overround and are
  never described as a learned model.
- Log-loss, Brier and accuracy carry 1,000-draw percentile bootstrap intervals.
  Pairwise deltas versus market odds use a paired bootstrap.

Reproduce:

```bash
python main.py evaluation
# or without writing artifacts / fewer draws:
python -m src.evaluation.full_season --bootstrap 100 --no-save
```

Artifacts:

- `models/evaluation/full_season_metrics.csv`
- `models/evaluation/full_season_breakdowns.csv`
- `models/evaluation/full_season_walk_forward.csv`
- `models/evaluation/full_season_summary.json`
- `reports/full_season_evaluation.md`

## 2. Validation (`25/26`, 380 matches)

| Model | Accuracy | Log-loss | 95% CI | Brier | Macro F1 | Draw recall | ECE |
|---|---:|---:|---|---:|---:|---:|---:|
| Market odds | 55.0% | **0.9637** | 0.9220–1.0040 | 0.5710 | 0.3876 | 0.0% | 0.0449 |
| Ensemble (validation weights) | 52.9% | 0.9760 | 0.9330–1.0221 | 0.5786 | 0.3875 | 2.2% | 0.0498 |
| Dynamic Elo | 52.4% | 0.9803 | 0.9344–1.0273 | 0.5812 | 0.3658 | 0.0% | 0.0377 |
| Ensemble (equal weights) | 51.1% | 0.9826 | 0.9418–1.0271 | 0.5829 | 0.4484 | 19.4% | 0.0625 |
| Dixon-Coles | 52.1% | 0.9934 | 0.9542–1.0324 | 0.5896 | 0.3596 | 0.0% | 0.0472 |
| Independent Poisson | 52.1% | 0.9934 | 0.9543–1.0321 | 0.5896 | 0.3596 | 0.0% | 0.0414 |
| Random Forest (production reference) | 45.8% | 1.0134 | 0.9678–1.0584 | 0.6029 | 0.4501 | 44.1% | 0.0963 |
| XGBoost | 46.6% | 1.0559 | 0.9889–1.1236 | 0.6286 | 0.4438 | 34.4% | 0.1009 |

**Validation selects the ensemble with weights RF 0.00, XGB 0.20, Dynamic Elo
0.70, Independent Poisson 0.10** (validation log-loss 0.9760 vs 0.9826 for equal
weights). The production Random Forest is not replaced by this experiment.

## 3. Test (`26/27`, 119 matches, scored once)

| Model | Accuracy | Log-loss | 95% CI | Brier | Macro F1 | Draw recall | ECE |
|---|---:|---:|---|---:|---:|---:|---:|
| Market odds | **56.3%** | **0.9171** | 0.8379–1.0008 | **0.5405** | 0.4238 | 0.0% | 0.0819 |
| Dynamic Elo | 54.6% | 0.9255 | 0.8524–1.0023 | 0.5440 | 0.4094 | 0.0% | 0.0693 |
| Ensemble (validation weights) | 55.5% | 0.9257 | 0.8503–1.0054 | 0.5443 | 0.4181 | 0.0% | 0.0662 |
| Ensemble (equal weights) | 52.1% | 0.9364 | 0.8638–1.0148 | 0.5513 | 0.4399 | 9.7% | 0.0813 |
| Independent Poisson | 52.1% | 0.9623 | 0.8984–1.0322 | 0.5695 | 0.3897 | 0.0% | 0.0802 |
| Dixon-Coles | 52.1% | 0.9623 | 0.8984–1.0322 | 0.5695 | 0.3897 | 0.0% | 0.0802 |
| Random Forest (production) | 53.8% | 0.9624 | 0.8834–1.0512 | 0.5672 | 0.5379 | 58.1% | 0.0786 |
| XGBoost | 53.8% | 1.0036 | 0.8853–1.1354 | 0.5868 | 0.5113 | 32.3% | 0.1011 |

**No evaluated learned model beats market odds on test log-loss.** The
validation-selected ensemble is close to Dynamic Elo, but both trail the market.
The confidence intervals are wide and overlap; no improvement is claimed.

## 4. Walk-forward

Fourteen model/fold rows across folds 1 and 2:

- Fold 1: train `23/24` → test `24/25`.
- Fold 2: train `23/24` + `24/25` → test `25/26`.
- The final `26/27` season is excluded.
- The walk-forward ensemble uses equal weights only, because validation-fitted
  weights from a *later* season would be temporally invalid in earlier folds.

## 5. Breakdowns and small samples

`models/evaluation/full_season_breakdowns.csv` reports metrics by:

- season;
- team (home and away appearances);
- predicted side (Home Win / Away Win picks);
- confidence bin (`<40%`, `40–50%`, `50–60%`, `60–70%`, `>=70%`).

Rows with fewer than **20** matches are marked `insufficient_sample` and contain
no metric values. A team's per-season sample on the synthetic test data is small,
so those rows are not evidence.

## 6. Production and live performance

- Production registry version: `random_forest-20261105T224238Z`; `registry verify`
  passes.
- No Phase 6 challenger is promoted. Promotion still requires strictly better
  validation log-loss and no Brier regression.
- The ledger currently has **17 prediction rows, 3 unique fixtures, 0 settled**.
  Live log-loss/Brier/calibration are therefore reported as
  `insufficient_data`, `live_performance_reliable=false`, and no live conclusion
  is drawn.
- Monitoring reports prediction counts, missing predictions, invalid
  probabilities, overdue settlements, freshness, fixture coverage and drift.
  A `confirmed` drift label requires at least 100 settled predictions and a
  supportive bootstrap interval.

## 7. Limitations

- The test season has 119 of 380 matches; test metrics are provisional.
- Football is noisy; close models have overlapping intervals.
- Market odds are an input to the tree models, so they are not independent
  forecasts, and no betting edge is claimed.
- No historical xG or player availability exists in this repository; no
  improvement from them is claimed.
- No closing-line or flat-stake out-of-sample audit was performed.
- The Linux CI workflow is configured; it was not executed from this Windows
  environment, so no Linux pass is claimed here.

## Live performance (Phase 8)

`python -m src.monitoring.live_performance` writes
`reports/live_performance.json` and `reports/live_performance.md`. The report
keeps three samples separate and never pools them:

| Split | Source | Meaning |
|---|---|---|
| Validation | production model metadata | model-selection window |
| Historical test | production model metadata | untouched test season |
| Live | append-only ledger | settled real predictions |

Every live block reports log-loss, multiclass Brier score, accuracy, macro-F1,
per-class precision/recall, draw recall and expected calibration error (ECE),
with its sample size. Below `MONITORING_MIN_SETTLED` (30) the block is marked
`insufficient_data` and must not be read as a conclusion.

Live metrics are also grouped by season and by model version, so a model change
cannot be hidden by pooled numbers and earlier versions keep their own record.
Void predictions (postponed/cancelled/abandoned fixtures) are excluded from all
metrics; corrected results are scored with the corrected official score, while
the previous record remains in `data/processed/result_corrections.jsonl`.

The API exposes the same report at `/performance/live`, and the dashboard's
Fixture Operations page displays it next to the refresh manifest and settled
results. Live performance never triggers retraining, promotion or a registry
write.
