# La Liga Predictor — Phase 2: Model Intelligence & Performance

Generated: 2026-11-06T17:51:04+00:00
Model version: `random_forest-20261105T224238Z`

## Executive summary

- Feature recommendation: **base_19**. No candidate produced a statistically credible improvement for both models on validation and walk-forward; the deployed 19-feature set is retained.
- Calibration selected on validation: Random Forest=sigmoid, XGBoost=sigmoid. This is diagnostic, not an automatic deployment.

## 1. Baselines vs learned models

All rows use the same chronological train/validation/test seasons. Baselines are fixed before test evaluation; test metrics were not used to choose a model.

### Test

| model | accuracy | log_loss | brier_score | draw_recall | ece |
|---|---|---|---|---|---|
| Market odds | 0.5688 | 0.9192 | 0.5418 | 0.0000 | 0.0891 |
| Logistic regression | 0.5046 | 0.9608 | 0.5679 | 0.5185 | 0.1144 |
| Historical frequencies | 0.4587 | 1.0641 | 0.6428 | 0.0000 | 0.0138 |
| Home always | 0.4587 | 14.9562 | 1.0826 | 0.0000 | 0.3609 |
| Majority class | 0.4587 | 14.9562 | 1.0826 | 0.0000 | 0.3609 |

### Validation

| model | accuracy | log_loss | brier_score | draw_recall | ece |
|---|---|---|---|---|---|
| Market odds | 0.5500 | 0.9637 | 0.5710 | 0.0000 | 0.0449 |
| Logistic regression | 0.4868 | 1.0295 | 0.6065 | 0.4731 | 0.1044 |
| Historical frequencies | 0.4895 | 1.0509 | 0.6332 | 0.0000 | 0.0316 |
| Home always | 0.4895 | 14.1064 | 1.0211 | 0.0000 | 0.3404 |
| Majority class | 0.4895 | 14.1064 | 1.0211 | 0.0000 | 0.3404 |

## 2. Feature experiments

Each candidate adds one feature group to the base 19. The retention rule requires lower validation log-loss, lower mean walk-forward log-loss and a 95% paired bootstrap upper bound below zero for both tree models. The test season was not used. The interval is deliberately conservative: a small point estimate that cannot be separated from noise is rejected.

| model | feature_set | n_features | validation_log_loss | walk_forward_log_loss | delta_validation_log_loss | delta_validation_ci_low | delta_validation_ci_high | delta_walk_forward_log_loss |
|---|---|---|---|---|---|---|---|---|
| Random Forest | all_extended | 46 | 1.0053 | 0.9984 | -0.0078 | -0.0209 | 0.0041 | -0.0045 |
| XGBoost | all_extended | 46 | 1.0619 | 1.0701 | 0.0093 | -0.0320 | 0.0477 | 0.0042 |
| Random Forest | base_19 | 19 | 1.0131 | 1.0029 | 0.0000 | nan | nan | 0.0000 |
| XGBoost | base_19 | 19 | 1.0526 | 1.0659 | 0.0000 | nan | nan | 0.0000 |
| Random Forest | elo | 22 | 1.0067 | 1.0036 | -0.0064 | -0.0177 | 0.0039 | 0.0008 |
| XGBoost | elo | 22 | 1.0352 | 1.0724 | -0.0174 | -0.0400 | 0.0053 | 0.0065 |
| Random Forest | goal_trends | 23 | 1.0116 | 0.9999 | -0.0016 | -0.0078 | 0.0046 | -0.0029 |
| XGBoost | goal_trends | 23 | 1.0576 | 1.0639 | 0.0050 | -0.0182 | 0.0282 | -0.0021 |
| Random Forest | head_to_head | 21 | 1.0077 | 0.9994 | -0.0054 | -0.0125 | 0.0011 | -0.0035 |
| XGBoost | head_to_head | 21 | 1.0380 | 1.0575 | -0.0146 | -0.0336 | 0.0030 | -0.0084 |
| Random Forest | long_form | 25 | 1.0116 | 1.0026 | -0.0015 | -0.0096 | 0.0068 | -0.0003 |
| XGBoost | long_form | 25 | 1.0706 | 1.0645 | 0.0180 | -0.0090 | 0.0464 | -0.0014 |
| Random Forest | rest_detail | 22 | 1.0124 | 0.9984 | -0.0008 | -0.0070 | 0.0052 | -0.0045 |
| XGBoost | rest_detail | 22 | 1.0654 | 1.0693 | 0.0128 | -0.0078 | 0.0328 | 0.0034 |
| Random Forest | strength_of_opposition | 22 | 1.0188 | 1.0006 | 0.0057 | -0.0027 | 0.0143 | -0.0022 |
| XGBoost | strength_of_opposition | 22 | 1.0707 | 1.0683 | 0.0181 | -0.0067 | 0.0435 | 0.0024 |
| Random Forest | venue_form | 25 | 1.0067 | 0.9987 | -0.0064 | -0.0144 | 0.0019 | -0.0041 |
| XGBoost | venue_form | 25 | 1.0376 | 1.0588 | -0.0150 | -0.0376 | 0.0071 | -0.0072 |

**Decision:** `base_19`. No candidate produced a statistically credible improvement for both models on validation and walk-forward; the deployed 19-feature set is retained.

Head-to-head and venue form were the closest candidates. Every candidate had a validation interval that included zero; for head-to-head the upper bounds were +0.0011 for Random Forest and +0.0030 for XGBoost. None of the candidates is retained.

xG was not tested: the cached Football-Data.co.uk extract has no reliable historical xG column, so adding one would mean inventing data. Elo, strength of schedule, venue form and 10-match form were tested and rejected under the same rule.

## 3. Calibration

Methods were selected on a chronological validation holdout and then scored once on test. Test metrics below compare all pre-declared methods for transparency, but selection did not use test.

| model | calibration | selected_on_validation | log_loss | brier_score | ece | accuracy | draw_recall |
|---|---|---|---|---|---|---|---|
| Random Forest | sigmoid | True | 0.9641 | 0.5722 | 0.0548 | 0.5505 | 0.0000 |
| Random Forest | isotonic | False | 0.9812 | 0.5793 | 0.0684 | 0.5413 | 0.0000 |
| Random Forest | none | False | 0.9837 | 0.5811 | 0.0888 | 0.5046 | 0.5556 |
| XGBoost | sigmoid | True | 0.9860 | 0.5890 | 0.0610 | 0.4954 | 0.0000 |
| XGBoost | isotonic | False | 1.0022 | 0.5903 | 0.0741 | 0.5138 | 0.0000 |
| XGBoost | none | False | 1.0224 | 0.5985 | 0.1155 | 0.5138 | 0.2963 |

Calibration comparison is diagnostic. No calibrator is applied automatically by this experiment run. On test, the validation-selected calibrator collapses draw recall to about zero for Random Forest, XGBoost. Treat that as a decision-quality regression even where log-loss/Brier improve.

## 4. Explainability

Permutation importance is measured on validation; native tree importance comes from the final train+validation fit; XGBoost TreeSHAP is measured on validation. None of these is causal evidence.

| model | feature | importance_mean | importance_std |
|---|---|---|---|
| Random Forest | implied_prob_home | 0.02519 | 0.00814 |
| Random Forest | implied_prob_away | 0.02287 | 0.00846 |
| Random Forest | implied_prob_draw | 0.01033 | 0.00348 |
| Random Forest | home_form_total_shots | 0.00210 | 0.00126 |
| Random Forest | home_form_shots_on_target | 0.00142 | 0.00130 |
| Random Forest | home_form_goals_scored | 0.00107 | 0.00233 |
| Random Forest | away_form_shots_on_target | 0.00055 | 0.00119 |
| Random Forest | home_goal_diff_5m | 0.00030 | 0.00164 |
| Random Forest | away_form_total_shots | 0.00029 | 0.00237 |
| Random Forest | home_advantage | 0.00000 | 0.00000 |
| XGBoost | implied_prob_away | 0.04941 | 0.01658 |
| XGBoost | implied_prob_draw | 0.03526 | 0.01033 |
| XGBoost | home_form_total_shots | 0.01922 | 0.00702 |
| XGBoost | away_form_shots_on_target | 0.00650 | 0.00378 |
| XGBoost | home_form_goals_scored | 0.00423 | 0.00444 |
| XGBoost | implied_prob_home | 0.00328 | 0.00932 |
| XGBoost | home_short_goals_conceded | 0.00219 | 0.00260 |
| XGBoost | home_goal_diff_5m | 0.00172 | 0.00756 |
| XGBoost | home_form_shots_on_target | 0.00091 | 0.00355 |
| XGBoost | away_short_goals_conceded | 0.00005 | 0.00420 |

## 5. Monitoring

Current monitoring verdict: **insufficient_data**. 

| name | status | message |
|---|---|---|
| missing_inputs | ok | Required inputs are present. |
| feature_drift | ok | Largest feature shift is +0.61 SD for home_form_total_shots over 60 recent matches. |
| team_distribution | ok | Largest team-appearance share shift is 5.7% for La Coruna (recent 60 matches). |
| prediction_distribution | insufficient_data | Only 10 valid logged predictions; need at least 50 for a before/after comparison. |
| calibration_change | insufficient_data | Only 0 settled predictions with probabilities; need 30. |
| performance | insufficient_data | Only 0 settled predictions with probabilities; need 30 before judging degradation. |

Monitoring distinguishes `insufficient_data`, `warning` and `confirmed` degradation. The ledger currently has no settled predictions, so performance and calibration checks are correctly reported as insufficient rather than healthy.

## 6. Reproducibility

- Experiment index: `models/experiments/index.csv`.
- Each experiment JSON records data fingerprint, feature hash, split, hyperparameters, validation metrics and whether test was evaluated.
- Random seed: 42.
- Test metrics were recorded only after model/feature/calibration decisions.

## 7. Limitations

- The test season contains 109 of 380 matches, so test differences are noisy.
- Draw recall and calibration remain difficult; one season is not enough to establish a stable betting edge.
- Market odds are included as features, so the learned model is not an independent forecast of the market.
- Head-to-head and rest features have sparse early-season values; imputation is training-only but still a modelling compromise. Head-to-head did not clear the statistical bar and was rejected.
- The strict feature rule can reject a real but small effect; that is a deliberate bias toward not shipping noise.
- SHAP for Random Forest is optional; XGBoost native TreeSHAP is always available.
