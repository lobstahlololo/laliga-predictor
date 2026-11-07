# La Liga Predictor — Phase 3: Advanced Football Modelling

Generated: 2026-11-07T15:34:29+00:00
Production model version (unchanged): `random_forest-20261105T224238Z`

## Executive summary

- Best **test** log-loss among all evaluated models: **Market odds** (0.9192); accuracy 56.9%, Brier 0.5418, ECE 0.0891, draw recall 0.0%.
- Best **validation** log-loss: **Market odds** (0.9637). Validation, not test, drove model and ensemble decisions.
- Market odds test log-loss: **0.9192** (Brier 0.5418, accuracy 56.9%).
- No Phase 3 model is promoted to production in this run. The current production model is preserved; test metrics are confirmatory, not a selection signal.
- Validation-weighted ensemble test log-loss: **0.9279** (Brier 0.5461, accuracy 55.0%).
- Relative to the current production Random Forest reference, Dynamic Elo improved validation log-loss from 1.0131 to 0.9803 and test log-loss from 0.9837 to 0.9244. The validation-weighted ensemble improved validation to 0.9756 and test to 0.9279. Both challengers still trail market odds, so this is a reason for a controlled review, not an automatic deployment.

> **Result honesty:** No evaluated model beats market odds on test log-loss. This is a negative result and is reported without adjusting the evaluation.

## 1. Dynamic Elo

The Elo system updates ratings match-by-match, includes a fixed home advantage, uses a margin-of-victory multiplier and tracks rating deviation. All fixtures on a date read the pre-date state; updates are applied after the date is complete. Initial rating: 1500, initial RD: 350. The draw mapping is fitted on training log-loss only.

Training-fitted Elo draw parameters: `{"draw_base": 0.29999999999999993, "draw_scale": 300.0, "selection": "training log-loss grid; test not used"}`.

### Latest ratings (descriptive, through the latest cached match)

| rank | team | rating | rating_deviation | matches |
|---|---|---|---|---|
| 1 | Barcelona | 1784.14 | 86.8 | 125 |
| 2 | Real Madrid | 1754.68 | 86.8 | 125 |
| 3 | Ath Madrid | 1663.24 | 86.8 | 125 |
| 4 | Villarreal | 1590.07 | 86.7 | 125 |
| 5 | Betis | 1583.79 | 86.8 | 125 |
| 6 | Ath Bilbao | 1545.64 | 87.3 | 124 |
| 7 | Sociedad | 1516.88 | 86.8 | 125 |
| 8 | La Coruna | 1509.07 | 199.3 | 11 |
| 9 | Celta | 1505.50 | 86.8 | 125 |
| 10 | Vallecano | 1498.75 | 87.3 | 124 |
| 11 | Girona | 1492.81 | 86.4 | 114 |
| 12 | Valencia | 1486.61 | 86.8 | 125 |

### Elo performance

| split | accuracy | log_loss | brier_score | draw_recall | ece |
|---|---|---|---|---|---|
| Validation | 0.5237 | 0.9803 | 0.5812 | 0.0000 | 0.0377 |
| Test | 0.5505 | 0.9244 | 0.5431 | 0.0000 | 0.0635 |

## 2. Poisson and Dixon-Coles

Expected goals are multiplicative functions of league home/away base rates, team attack and team defence. Observations are time-decayed, team estimates are shrunk toward the league average, and clubs with fewer than the minimum matches fall back explicitly to league average. The Dixon-Coles low-score correction is fitted on the training likelihood, not on test.

- Selected Poisson configuration: `{"max_goals": 10, "half_life_days": 365.0, "prior_strength": 2.0, "min_matches": 3, "dixon_coles": true, "rho_min": -0.15, "rho_max": 0.15, "rho_steps": 31, "iterations": 80, "minimum_weight": 0.01}`.
- Fitted rho: **+0.0100**.
- League home base: 1.532 goals; away base: 1.141 goals.
- Example fixture: **Alaves vs Getafe**; expected goals 1.01–0.79; probabilities H/D/A 40.0%/32.3%/27.7%; most likely score 1-0.

### Attack and defence estimates

| team | attack | defence | matches | effective_matches | known |
|---|---|---|---|---|---|
| Barcelona | 1.788 | 0.782 | 114 | 52.0 | True |
| Real Madrid | 1.500 | 0.717 | 114 | 52.0 | True |
| Villarreal | 1.388 | 1.012 | 114 | 52.2 | True |
| Ath Madrid | 1.251 | 0.829 | 114 | 52.0 | True |
| Betis | 1.126 | 0.967 | 114 | 52.0 | True |
| Celta | 1.072 | 1.021 | 114 | 52.0 | True |
| Sociedad | 1.021 | 1.079 | 114 | 52.0 | True |
| Elche | 0.992 | 1.119 | 38 | 29.8 | True |
| Levante | 0.960 | 1.159 | 38 | 29.9 | True |
| Ath Bilbao | 0.958 | 0.928 | 114 | 51.9 | True |
| Girona | 0.929 | 1.060 | 114 | 52.0 | True |
| Osasuna | 0.907 | 1.016 | 114 | 52.0 | True |

### Poisson and Dixon-Coles performance

| split | model | accuracy | log_loss | brier_score | draw_recall | ece |
|---|---|---|---|---|---|---|
| Validation | Dixon-Coles | 0.5263 | 0.9922 | 0.5886 | 0.0000 | 0.0532 |
| Validation | Independent Poisson | 0.5263 | 0.9922 | 0.5885 | 0.0000 | 0.0455 |
| Test | Independent Poisson | 0.5505 | 0.9572 | 0.5661 | 0.0000 | 0.0813 |
| Test | Dixon-Coles | 0.5505 | 0.9574 | 0.5662 | 0.0000 | 0.0774 |

## 3. Model comparison

All rows use the same chronological windows. Validation rows train on 23/24+24/25; test rows refit on 23/24+24/25+25/26 and score 26/27 once. Market odds are a benchmark, not a learned model.

### Validation

| model | accuracy | log_loss | brier_score | draw_recall | ece |
|---|---|---|---|---|---|
| Market odds | 0.5500 | 0.9637 | 0.5710 | 0.0000 | 0.0449 |
| Ensemble (validation weights) | 0.5289 | 0.9756 | 0.5782 | 0.0430 | 0.0487 |
| Dynamic Elo | 0.5237 | 0.9803 | 0.5812 | 0.0000 | 0.0377 |
| Ensemble (equal) | 0.5158 | 0.9820 | 0.5824 | 0.2043 | 0.0610 |
| Dixon-Coles | 0.5263 | 0.9922 | 0.5886 | 0.0000 | 0.0532 |
| Independent Poisson | 0.5263 | 0.9922 | 0.5885 | 0.0000 | 0.0455 |
| Elo + Random Forest | 0.4842 | 1.0014 | 0.5976 | 0.2903 | 0.0937 |
| Poisson + Random Forest | 0.4553 | 1.0051 | 0.5977 | 0.3871 | 0.0904 |
| Random Forest | 0.4579 | 1.0131 | 0.6027 | 0.4409 | 0.0967 |
| Logistic regression | 0.4868 | 1.0295 | 0.6065 | 0.4731 | 0.1044 |
| Elo + XGBoost | 0.4842 | 1.0367 | 0.6195 | 0.1505 | 0.1032 |
| Poisson + XGBoost | 0.4947 | 1.0497 | 0.6240 | 0.3871 | 0.1043 |
| XGBoost | 0.4658 | 1.0526 | 0.6268 | 0.3763 | 0.1093 |

### Test

| model | accuracy | log_loss | brier_score | draw_recall | ece |
|---|---|---|---|---|---|
| Market odds | 0.5688 | 0.9192 | 0.5418 | 0.0000 | 0.0891 |
| Dynamic Elo | 0.5505 | 0.9244 | 0.5431 | 0.0000 | 0.0635 |
| Ensemble (validation weights) | 0.5505 | 0.9279 | 0.5461 | 0.0000 | 0.0866 |
| Ensemble (equal) | 0.5046 | 0.9440 | 0.5567 | 0.0370 | 0.0964 |
| Independent Poisson | 0.5505 | 0.9572 | 0.5661 | 0.0000 | 0.0813 |
| Dixon-Coles | 0.5505 | 0.9574 | 0.5662 | 0.0000 | 0.0774 |
| Logistic regression | 0.5046 | 0.9608 | 0.5679 | 0.5185 | 0.1144 |
| Elo + Random Forest | 0.4862 | 0.9689 | 0.5742 | 0.5185 | 0.0891 |
| Random Forest | 0.5046 | 0.9837 | 0.5811 | 0.5556 | 0.0888 |
| Poisson + Random Forest | 0.4954 | 0.9851 | 0.5820 | 0.5556 | 0.0999 |
| XGBoost | 0.5138 | 1.0224 | 0.5985 | 0.2963 | 0.1155 |
| Elo + XGBoost | 0.4679 | 1.0329 | 0.6143 | 0.2963 | 0.0931 |
| Poisson + XGBoost | 0.4771 | 1.0612 | 0.6229 | 0.2963 | 0.1182 |

## 4. Walk-forward performance

The existing tree-model walk-forward rows are combined with new market, logistic, Elo, Poisson and Dixon-Coles rows. Folds train only on seasons before their target season; the final test season is excluded.

| fold | test_season | model | n_samples | log_loss | brier_score | accuracy | draw_recall | ece |
|---|---|---|---|---|---|---|---|---|
| 1 | 24/25 | Market odds | 380 | 0.9534 | 0.5640 | 0.5447 | 0.0000 | 0.0388 |
| 1 | 24/25 | Dynamic Elo | 380 | 0.9924 | 0.5889 | 0.5211 | 0.0000 | 0.0503 |
| 1 | 24/25 | Random Forest | 380 | 0.9926 | 0.5895 | 0.5316 | 0.2474 | 0.0648 |
| 1 | 24/25 | Logistic regression | 380 | 1.0066 | 0.5864 | 0.5211 | 0.2990 | 0.0769 |
| 1 | 24/25 | Independent Poisson | 380 | 1.0230 | 0.6126 | 0.5000 | 0.0103 | 0.0246 |
| 1 | 24/25 | Dixon-Coles | 380 | 1.0234 | 0.6126 | 0.5026 | 0.0206 | 0.0321 |
| 1 | 24/25 | XGBoost | 380 | 1.0793 | 0.6438 | 0.4632 | 0.2577 | 0.1196 |
| 2 | 25/26 | Market odds | 380 | 0.9637 | 0.5710 | 0.5500 | 0.0000 | 0.0449 |
| 2 | 25/26 | Dynamic Elo | 380 | 0.9803 | 0.5812 | 0.5237 | 0.0000 | 0.0377 |
| 2 | 25/26 | Dixon-Coles | 380 | 0.9934 | 0.5896 | 0.5211 | 0.0000 | 0.0472 |
| 2 | 25/26 | Independent Poisson | 380 | 0.9934 | 0.5896 | 0.5211 | 0.0000 | 0.0414 |
| 2 | 25/26 | Random Forest | 380 | 1.0131 | 0.6027 | 0.4579 | 0.4409 | 0.0967 |
| 2 | 25/26 | Logistic regression | 380 | 1.0295 | 0.6065 | 0.4868 | 0.4731 | 0.1044 |
| 2 | 25/26 | XGBoost | 380 | 1.0526 | 0.6268 | 0.4658 | 0.3763 | 0.1093 |

## 5. Ensemble

Weights are non-negative, sum to one, and were selected by a fixed simplex grid on validation predictions only. Equal weights are included in the grid, so the selected ensemble cannot be worse than equal weighting on validation.

- Selected weights: `{"XGBoost": 0.2, "Dynamic Elo": 0.7, "Independent Poisson": 0.1}`.
- Validation log-loss: 0.9756; equal-weight validation log-loss: 0.9820.
- Test log-loss: 0.9279.

| model | accuracy | log_loss | brier_score | draw_recall | ece |
|---|---|---|---|---|---|
| Ensemble (validation weights) | 0.5505 | 0.9279 | 0.5461 | 0.0000 | 0.0866 |
| Ensemble (equal) | 0.5046 | 0.9440 | 0.5567 | 0.0370 | 0.0964 |

## 6. Market odds analysis

Raw implied probabilities are `1 / decimal odds`; overround is removed by row-normalising those probabilities. The market remains the primary benchmark. Model-versus-market differences are descriptive, not a claim of a betting edge.

### Overround by season

| season | matches | mean_overround | median_overround | min_overround | max_overround |
|---|---|---|---|---|---|
| 23/24 | 380 | 0.0540 | 0.0537 | 0.0310 | 0.1045 |
| 24/25 | 380 | 0.0565 | 0.0558 | 0.0418 | 0.1226 |
| 25/26 | 380 | 0.0561 | 0.0552 | 0.0384 | 0.1096 |
| 26/27 | 109 | 0.0550 | 0.0545 | 0.0382 | 0.0762 |

### Model versus market

| model | model_log_loss | market_log_loss | delta_log_loss | delta_log_loss_ci_low | delta_log_loss_ci_high | mean_abs_probability_difference | beats_market_log_loss |
|---|---|---|---|---|---|---|---|
| Market odds | 0.9192 | 0.9192 | 0.0000 | -0.0000 | 0.0000 | 0.0000 | False |
| Dynamic Elo | 0.9244 | 0.9192 | 0.0052 | -0.0236 | 0.0339 | 0.0402 | False |
| Ensemble (validation weights) | 0.9279 | 0.9192 | 0.0087 | -0.0183 | 0.0332 | 0.0366 | False |
| Ensemble (equal) | 0.9440 | 0.9192 | 0.0248 | -0.0071 | 0.0534 | 0.0439 | False |
| Independent Poisson | 0.9572 | 0.9192 | 0.0380 | 0.0004 | 0.0735 | 0.0575 | False |
| Dixon-Coles | 0.9574 | 0.9192 | 0.0382 | 0.0003 | 0.0739 | 0.0578 | False |
| Logistic regression | 0.9608 | 0.9192 | 0.0416 | -0.0163 | 0.1029 | 0.0776 | False |
| Elo + Random Forest | 0.9689 | 0.9192 | 0.0497 | -0.0006 | 0.1000 | 0.0738 | False |
| Random Forest | 0.9837 | 0.9192 | 0.0645 | 0.0164 | 0.1160 | 0.0760 | False |
| Poisson + Random Forest | 0.9851 | 0.9192 | 0.0659 | 0.0140 | 0.1186 | 0.0758 | False |
| XGBoost | 1.0224 | 0.9192 | 0.1032 | 0.0159 | 0.1912 | 0.1008 | False |
| Elo + XGBoost | 1.0329 | 0.9192 | 0.1137 | 0.0252 | 0.1972 | 0.1017 | False |
| Poisson + XGBoost | 1.0612 | 0.9192 | 0.1420 | 0.0580 | 0.2259 | 0.0956 | False |

### Picks where model and market disagree

| model | n | model_log_loss | market_log_loss | model_accuracy | market_accuracy | model_beats_market |
|---|---|---|---|---|---|---|
| Market odds | 0 |  |  |  |  | False |
| Random Forest | 45 | 1.0970 | 1.0381 | 0.3556 | 0.5111 | False |
| XGBoost | 32 | 1.1002 | 1.0212 | 0.3750 | 0.5625 | False |
| Logistic regression | 48 | 1.1466 | 1.0352 | 0.3542 | 0.5000 | False |
| Dynamic Elo | 10 | 1.1710 | 1.1090 | 0.1000 | 0.3000 | False |
| Independent Poisson | 13 | 1.2141 | 1.1419 | 0.1538 | 0.3077 | False |
| Dixon-Coles | 13 | 1.2174 | 1.1419 | 0.1538 | 0.3077 | False |
| Elo + Random Forest | 46 | 1.0899 | 1.0433 | 0.3261 | 0.5217 | False |
| Elo + XGBoost | 39 | 1.1531 | 1.0299 | 0.2564 | 0.5385 | False |
| Poisson + Random Forest | 46 | 1.1030 | 1.0415 | 0.3261 | 0.5000 | False |
| Poisson + XGBoost | 31 | 1.1846 | 0.9961 | 0.2581 | 0.5806 | False |
| Ensemble (equal) | 14 | 1.0860 | 1.0019 | 0.1429 | 0.6429 | False |
| Ensemble (validation weights) | 8 | 1.1377 | 1.0991 | 0.1250 | 0.3750 | False |

### Validation-fitted market + model stack

This stack asks whether the component probabilities add information conditional on the market. It is fitted on validation only and scored once on test; it is not deployed.

| model | log_loss | brier_score | accuracy | draw_recall | market_log_loss | delta_log_loss_vs_market | delta_log_loss_ci_low | delta_log_loss_ci_high |
|---|---|---|---|---|---|---|---|---|
| Market + RF/XGB/Elo/Poisson stack | 0.9167 | 0.5421 | 0.5780 | 0.1111 | 0.9192 | -0.0025 | -0.0299 | 0.0279 |

## 7. Prediction uncertainty

Confidence is the largest class probability, entropy is the normalized spread across outcomes, agreement is one minus mean pairwise total-variation distance between model families, and historical reliability combines Elo matches, Poisson match counts and known-team status. These are descriptive diagnostics, not calibrated intervals.

- Fixtures: **109**; mean confidence 51.6%; mean entropy 0.900; mean agreement 0.878.
- Uncertainty levels: `{"high": 76, "low": 23, "medium": 10}`.
- Historical reliability: `{"high": 77, "low": 32}`.
- Spearman correlation between entropy and realised row log-loss: 0.4665554628857381.

## 8. Reproducibility

- Data fingerprint and feature hashes are recorded in every experiment JSON.
- Random seed: 42.
- Ensemble weights: selected on validation only, then applied unchanged to test.
- Poisson time-decay and shrinkage: selected on validation only, then refit on train+validation for test.
- Elo ratings are chronological; no post-match rating enters a pre-match feature.
- Experiment index: `models/experiments/index.csv`.
- Test season was scored once after all choices; it was not used to tune a model, feature or weight.

## 9. Limitations

- The test season has 109 of 380 matches, so differences of a few thousandths in log-loss are noisy.
- Elo starts every club at 1500; promoted clubs are not given a separate prior.
- Poisson estimates are multiplicative and do not model lineups, injuries, red cards, xG or shot quality.
- The Dixon-Coles correction is fitted on the training likelihood, not a separate validation window.
- Market odds are also inputs to the tree models, so the learned models are not independent forecasts.
- No betting edge is claimed. A flat-stake or closing-line audit on genuinely unseen fixtures would be required.
- The ensemble grid can overfit the validation season; the equal-weight reference is reported for that reason.
- Uncertainty levels are descriptive and not calibrated probability intervals.

