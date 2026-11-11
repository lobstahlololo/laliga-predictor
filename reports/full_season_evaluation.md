# La Liga Predictor — Full-Season Out-of-Sample Evaluation

Generated: 2026-11-11T00:28:17+00:00

## Protocol

- Train seasons: 23/24, 24/25
- Validation season: 25/26
- Untouched test season: 26/27
- Model selection and ensemble weights use validation only. Test used for selection: **False**.
- Metrics carry bootstrap confidence intervals; paired deltas are against row-normalised market probabilities.
- A test point estimate is not a claim of improvement on its own.

## Headline

- Best **validation** learned model: **Ensemble (validation)** (log-loss 0.9760, Brier 0.5786, accuracy 52.9%).
- Best **test** log-loss: **Market odds** (0.9171). This is confirmatory, not a selection signal.
- Market test log-loss: **0.9171** (Brier 0.5405, accuracy 56.3%).
- Production reference (unchanged): `random_forest-20261105T224238Z` (Random Forest).

## Validation

| model | n_samples | accuracy | log_loss | log_loss_ci_low | log_loss_ci_high | brier_score | macro_f1 | draw_recall | ece |
|---|---|---|---|---|---|---|---|---|---|
| Market odds | 380 | 55.0% | 0.9637 | 0.9220 | 1.0040 | 0.5710 | 0.388 | 0.0% | 0.0449 |
| Ensemble (validation) | 380 | 52.9% | 0.9760 | 0.9330 | 1.0221 | 0.5786 | 0.388 | 2.2% | 0.0498 |
| Dynamic Elo | 380 | 52.4% | 0.9803 | 0.9344 | 1.0273 | 0.5812 | 0.366 | 0.0% | 0.0377 |
| Ensemble (equal) | 380 | 51.1% | 0.9826 | 0.9418 | 1.0271 | 0.5829 | 0.448 | 19.4% | 0.0625 |
| Dixon-Coles | 380 | 52.1% | 0.9934 | 0.9542 | 1.0324 | 0.5896 | 0.36 | 0.0% | 0.0472 |
| Independent Poisson | 380 | 52.1% | 0.9934 | 0.9543 | 1.0321 | 0.5896 | 0.36 | 0.0% | 0.0414 |
| Random Forest | 380 | 45.8% | 1.0134 | 0.9678 | 1.0584 | 0.6029 | 0.45 | 44.1% | 0.0963 |
| XGBoost | 380 | 46.6% | 1.0559 | 0.9889 | 1.1236 | 0.6286 | 0.444 | 34.4% | 0.1009 |

## Test (single scoring pass)

| model | n_samples | accuracy | accuracy_ci_low | accuracy_ci_high | log_loss | log_loss_ci_low | log_loss_ci_high | brier_score | macro_f1 | draw_recall | ece | delta_log_loss_vs_market | delta_ci_low | delta_ci_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Market odds | 119 | 56.3% | 47.1% | 65.5% | 0.9171 | 0.8379 | 1.0008 | 0.5405 | 0.424 | 0.0% | 0.0819 | 0.0000 | 0.0000 | 0.0000 |
| Dynamic Elo | 119 | 54.6% | 45.4% | 63.9% | 0.9255 | 0.8524 | 1.0023 | 0.5440 | 0.409 | 0.0% | 0.0693 | 0.0084 | -0.0178 | 0.0359 |
| Ensemble (validation) | 119 | 55.5% | 46.2% | 64.7% | 0.9257 | 0.8503 | 1.0054 | 0.5443 | 0.418 | 0.0% | 0.0662 | 0.0086 | -0.0147 | 0.0322 |
| Ensemble (equal) | 119 | 52.1% | 43.7% | 60.5% | 0.9364 | 0.8638 | 1.0148 | 0.5513 | 0.44 | 9.7% | 0.0813 | 0.0193 | -0.0105 | 0.0460 |
| Independent Poisson | 119 | 52.1% | 42.9% | 61.3% | 0.9623 | 0.8984 | 1.0322 | 0.5695 | 0.39 | 0.0% | 0.0802 | 0.0451 | 0.0099 | 0.0786 |
| Dixon-Coles | 119 | 52.1% | 42.9% | 61.3% | 0.9623 | 0.8984 | 1.0322 | 0.5695 | 0.39 | 0.0% | 0.0802 | 0.0451 | 0.0099 | 0.0786 |
| Random Forest | 119 | 53.8% | 45.4% | 62.2% | 0.9624 | 0.8834 | 1.0512 | 0.5672 | 0.538 | 58.1% | 0.0786 | 0.0452 | -0.0086 | 0.0924 |
| XGBoost | 119 | 53.8% | 44.5% | 62.2% | 1.0036 | 0.8853 | 1.1354 | 0.5868 | 0.511 | 32.3% | 0.1011 | 0.0864 | 0.0019 | 0.1726 |

## Walk-forward (test season excluded)

| fold | test_season | model | n_samples | accuracy | log_loss | brier_score | macro_f1 | draw_recall | ece |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 24/25 | Market odds | 380 | 54.5% | 0.9534 | 0.5640 | 0.407 | 0.0% | 0.0388 |
| 1 | 24/25 | Ensemble (equal) | 380 | 51.6% | 0.9845 | 0.5864 | 0.443 | 11.3% | 0.0514 |
| 1 | 24/25 | Random Forest | 380 | 53.4% | 0.9892 | 0.5877 | 0.496 | 24.7% | 0.0694 |
| 1 | 24/25 | Dynamic Elo | 380 | 52.1% | 0.9924 | 0.5889 | 0.379 | 0.0% | 0.0503 |
| 1 | 24/25 | Independent Poisson | 380 | 50.0% | 1.0230 | 0.6126 | 0.364 | 1.0% | 0.0246 |
| 1 | 24/25 | Dixon-Coles | 380 | 50.3% | 1.0234 | 0.6126 | 0.372 | 2.1% | 0.0321 |
| 1 | 24/25 | XGBoost | 380 | 47.4% | 1.0809 | 0.6451 | 0.445 | 26.8% | 0.1178 |
| 2 | 25/26 | Market odds | 380 | 55.0% | 0.9637 | 0.5710 | 0.388 | 0.0% | 0.0449 |
| 2 | 25/26 | Dynamic Elo | 380 | 52.4% | 0.9803 | 0.5812 | 0.366 | 0.0% | 0.0377 |
| 2 | 25/26 | Ensemble (equal) | 380 | 51.1% | 0.9826 | 0.5829 | 0.448 | 19.4% | 0.0625 |
| 2 | 25/26 | Dixon-Coles | 380 | 52.1% | 0.9934 | 0.5896 | 0.36 | 0.0% | 0.0472 |
| 2 | 25/26 | Independent Poisson | 380 | 52.1% | 0.9934 | 0.5896 | 0.36 | 0.0% | 0.0414 |
| 2 | 25/26 | Random Forest | 380 | 45.8% | 1.0134 | 0.6029 | 0.45 | 44.1% | 0.0963 |
| 2 | 25/26 | XGBoost | 380 | 46.6% | 1.0559 | 0.6286 | 0.444 | 34.4% | 0.1009 |

## Breakdowns

Per-season, per-team, predicted-side and confidence-bin rows are in `C:\Users\indor\laliga-predictor\models\evaluation\full_season_breakdowns.csv`. Rows with fewer than 20 matches are marked `insufficient_sample` and contain no metrics.

## Limitations

- The test season is in progress, so its sample is small and test metrics are provisional.
- Football outcomes are noisy; confidence intervals overlap for close models.
- Market probabilities are inputs to the tree models, so learned models are not independent forecasts.
- No betting edge is claimed: there is no closing-line or flat-stake out-of-sample audit here.
- Historical xG and player availability are unavailable in this repository; no improvement from them is claimed.

