# ⚽ La Liga Predictor — Football Prediction & Analytics Platform

An end-to-end, leakage-safe football-modelling platform for Spanish La Liga
match outcomes (**Home Win / Draw / Away Win**). It covers real data ingestion,
19 pre-kickoff features, chronological model selection, walk-forward
backtesting, probability-calibration diagnostics, dynamic Elo ratings,
Poisson/Dixon-Coles goal models, constrained ensembles, market-odds analysis,
prediction-uncertainty diagnostics, an append-only prediction ledger, a versioned
model registry, an automated update cycle, a FastAPI service, and an eight-page
Streamlit analytics dashboard.

The project is designed as both a working application and an honest ML
portfolio piece: **the test season is never used to choose a model**, accuracy
alone never decides a winner, and every reported number is measured from real
data rather than asserted.

> Predictions are probabilistic estimates for analytics and education. They are
> not guarantees and not betting advice.

---

## 1. What the platform does

- Downloads real La Liga results, shots, shots on target and Bet365 odds from
  [Football-Data.co.uk](https://www.football-data.co.uk/) (SP1).
- Discovers the upcoming La Liga fixtures from a public full-season calendar
  ([OpenFootball](https://github.com/openfootball/football.json), CC0, no key)
  merged with TheSportsDB/API-Football windows, then verifies completeness
  against that calendar instead of assuming a provider window is the schedule.
- Validates, de-duplicates and normalizes the raw dataset, with an incremental
  refresh that keeps cached data when a download fails.
- Builds 19 features using only information available before kickoff.
- Trains **Random Forest** and **XGBoost** multiclass classifiers.
- Tunes hyperparameters with time-ordered cross-validation.
- Selects a model on **validation log-loss only**, then scores both models once
  on an untouched test season.
- Reports accuracy, log-loss, multiclass Brier score, per-class
  precision/recall/F1, draw recall, confusion matrices, calibration curves and
  season-by-season results.
- Runs a rolling-origin walk-forward backtest.
- Stores every prediction with fixture ID, match date, timestamp, probabilities,
  model name/version, predicted outcome and later actual result.
- Reconciles results without overwriting prior records, and audits a
  hypothetical flat-stake value-bet ledger.
- Compares the learned models against fixed baselines (majority class, home
always, historical frequencies, market odds and multinomial logistic regression)
  on identical chronological windows.
- Tests candidate feature groups (10-match form, venue form, goal trends, rest,
  Elo, strength of schedule and head-to-head) against a strict validation +
  walk-forward + bootstrap rule, and keeps the 19 features when nothing clears it.
- Measures permutation importance, native tree importance, XGBoost TreeSHAP and
  fold stability without claiming causality.
- Investigates sigmoid and isotonic calibration on a validation holdout, with
  reliability diagrams, ECE, multiclass Brier score and per-outcome metrics.
- Adds a chronological dynamic Elo system with home advantage, margin-of-victory
  updates, rating deviation and rating-history snapshots.
- Fits time-decayed, regularised Poisson goal models, aggregates Poisson scoreline
  matrices into home/draw/away probabilities, and tests a Dixon-Coles low-score
  correction.
- Selects non-negative ensemble weights on validation only and compares the result
  with every component model and with market odds.
- Removes bookmaker overround, measures model-versus-market probabilities, and
  tests whether model signals add information beyond the market.
- Separates probability confidence, entropy, model agreement and historical/data
  reliability instead of treating a high probability as certainty.
- Tracks reproducible experiment records and adds evidence-aware monitoring that
  separates insufficient data, warnings and confirmed degradation.
- Serves a Streamlit dashboard with Home, Match Predictions, Model Performance,
  Team Analytics, Prediction History, Model Monitoring, Model Intelligence and
  Model Operations pages.
- Registers every trained model as an immutable version with metrics, periods,
  feature-set hash and an artifact snapshot; promotion requires measured
  validation evidence and rollback restores the previous snapshot.
- Runs an idempotent update cycle (data refresh → predictions → reconciliation →
  reports → monitoring) with retries, file logging, validation and optional
  webhook alerts for failures.
- Audits historical data quality by column and season, and audits fixture
  coverage/provider freshness without inventing missing fixtures.
- Compares Random Forest, XGBoost, Elo, Poisson/Dixon-Coles, ensembles and the
  market on a chronological full season with bootstrap intervals and
  sample-size-gated breakdowns.
- Monitors prediction counts, missing predictions, calibration, drift and data
  freshness, and refuses to present live metrics as conclusions until enough
  matches are settled.
- Exposes a FastAPI service for fixtures, single-match predictions, team
  statistics, performance and registry status, using the same prediction engine
  as the CLI and dashboard.
- Ships Docker and docker-compose files, GitHub Actions CI plus a scheduled update
  workflow, an environment template and a deployment guide.

---

## 2. Leakage and evaluation guarantees

These are the rules the code enforces:

| Rule | Implementation |
|---|---|
| No future information in features | Rolling statistics use `shift(1)` before rolling; rest days use the previous match only. |
| Chronological splits | Earlier seasons train, the penultimate season validates, the newest season tests. No random row split. |
| Time-ordered tuning | `RandomizedSearchCV` with `TimeSeriesSplit`, optimising `neg_log_loss`. |
| Validation-only selection | Lowest validation log-loss wins; Brier score breaks ties. The test set is not touched until after selection. |
| One-shot test | Both candidates are refit on train + validation and scored once on the test season. |
| Untouched final test | The test season is excluded from walk-forward diagnostics and from all tuning. |
| Preprocessing fitted on train only | Median imputation + `StandardScaler` are fitted on the training window (or the fold's training window). |
| Consistent train/inference features | A single `FeaturePreprocessor` and one canonical 19-feature order are used everywhere. |
| Elo chronology | Each fixture reads the ratings and uncertainty that existed before its date; all updates for a date are applied after the date is described. |
| Poisson chronology | Models are fitted only on matches on or before an `as_of` cutoff; pre-match features refit at month boundaries and predict each month from earlier matches only. |
| Ensemble selection | Weights are selected on validation predictions only; the untouched test season is not an input to the grid search. |
| Market benchmark | Raw `1 / odds` probabilities are explicitly row-normalised to remove overround before every comparison. |
| Append-only history | Prediction rows are never overwritten; a new timestamp/version appends a snapshot. Results fill blank fields only. |
| Evidence-gated promotion | The registry promotes only when validation log-loss is strictly better and validation Brier is not worse; test metrics are recorded but never decide a promotion. |
| Fixture integrity | Merging never deletes a stored fixture because a provider returned a partial window; conflicting kickoffs are reported, not silently resolved. |
| Coverage is measured | A provider "next events" window is reported as unknown completeness, never as a complete league schedule. |

---

## 3. Architecture

```text
Football-Data.co.uk (SP1 CSV)
          │
          ▼
[ src/data ]  download → validate → normalize teams → dedupe → chronological CSV
          │
          ▼
[ src/features ]  5m/3m rolling form → 19 features → train-only imputer/scaler
          │
          ▼
[ src/evaluation ] seasonal splits → TimeSeriesSplit tuning → validation selection
          │            → one-shot test → walk-forward → baselines/calibration/SHAP
          │            → Elo/Poisson comparison → market analysis → uncertainty
          ▼
[ src/models ]  Random Forest + XGBoost + dynamic Elo + Poisson/Dixon-Coles
          │            → production model unchanged; Phase 3 artifacts separate
          │
          ├──────────────► [ src/predictions ] CLI / dashboard inference → ledger
          │                                      │
          │                                      ▼
          │                              backfill_results.py (real results only)
          │
          ├──────────────► [ src/experiments ] experiment JSON + index.csv
          │
          └──────────────► [ src/monitoring ] drift/calibration/performance verdicts
          ▼
[ dashboard ]  seven Streamlit pages reading shared cached data access
```

---

## 4. Project structure

```text
laliga-predictor/
├── src/
│   ├── data/
│   │   ├── make_dataset.py       # fetch, merge, validate, atomic write
│   │   ├── validation.py         # schema, dedupe, team normalization, quality
│   │   ├── fixture_provider.py    # TheSportsDB/API-Football discovery + safe merge
│   │   ├── fixture_coverage.py    # coverage, conflicts, provider telemetry
│   │   ├── football_calendar.py  # season calendar cache + completeness check
│   │   ├── data_quality.py        # column/season quality audit + reports
│   │   └── historical_sources.py  # xG/availability source audit (no network)
│   ├── features/
│   │   ├── build_features.py     # rolling features + model frame + artifacts
│   │   └── preprocessing.py      # train-only median imputation + scaling
│   ├── models/
│   │   ├── train_model.py        # tuning, selection, test, walk-forward, plots
│   │   ├── elo.py                # chronological Elo, uncertainty, snapshots
│   │   ├── poisson.py            # time-decayed Poisson/Dixon-Coles goals
│   │   └── registry.py           # versioned artifacts, promotion, rollback
│   ├── evaluation/
│   │   ├── metrics.py            # log-loss, Brier, calibration, per-class
│   │   ├── splits.py             # chronological split + walk-forward folds
│   │   ├── backtesting.py        # rolling-origin runner
│   │   ├── baselines.py          # fixed + logistic baselines
│   │   ├── calibration.py        # sigmoid/isotonic, validation-holdout selection
│   │   ├── explainability.py     # permutation, tree, TreeSHAP, stability
│   │   ├── ensemble.py           # validation-only constrained weights
│   │   ├── market.py             # overround removal and market diagnostics
│   │   ├── uncertainty.py        # confidence/entropy/agreement/reliability
│   │   ├── phase2_plots.py       # baseline/experiment/importance/calibration plots
│   │   ├── phase3_plots.py       # Elo/Poisson/ensemble/market/uncertainty plots
│   │   ├── full_season.py        # Phase 6 full-season comparison + intervals
│   │   └── plots.py              # confusion/calibration/comparison charts
│   ├── experiments/
│   │   ├── tracking.py           # JSON records + index.csv
│   │   ├── runner.py             # Phase 2 orchestrator and CLI
│   │   └── phase3_runner.py      # Phase 3 orchestrator and CLI
│   ├── pipeline/
│   │   └── automation.py         # idempotent update cycle, retries, alerts
│   ├── monitoring/
│   │   ├── diagnostics.py        # drift, distribution, calibration, performance
│   │   └── prediction_monitor.py # Phase 6 counts, drift, freshness, coverage
│   ├── predictions/
│   │   ├── predictor.py          # shared inference engine
│   │   ├── fixtures.py           # fixtures.txt / upcoming_fixtures.csv parser
│   │   └── history.py            # versioned ledger, reconciliation, reports
│   ├── api/
│   │   └── main.py               # FastAPI read-only service
│   ├── utils/
│   │   ├── config.py             # paths, features, seasons, constants
│   │   └── teams.py              # aliases, normalization, fuzzy resolution
│   └── visualization/            # reserved for additional chart helpers
├── dashboard/
│   ├── app.py                    # Home page
│   ├── components.py             # shared styling/components
│   ├── data_access.py            # cached data/model/history access
│   └── pages/
│       ├── 1_Match_Predictions.py
│       ├── 2_Model_Performance.py
│       ├── 3_Team_Analytics.py
│       ├── 4_Prediction_History.py
│       ├── 5_Model_Monitoring.py
│       ├── 6_Model_Intelligence.py
│       └── 7_Model_Operations.py
├── tests/                        # 201 pytest tests
├── models/
│   ├── saved_models/             # best_model.pkl, per-model pkl, metadata JSON
│   ├── evaluation/               # metrics CSVs, PNGs, evaluation_summary.json
│   └── registry/                 # versioned snapshots + registry.json
├── data/
│   ├── raw/                      # laliga_matches.csv, upcoming_fixtures.csv
│   └── processed/                # preprocessor, scaler, split CSVs, ledger
├── reports/                      # ledger reports + pipeline_runs.jsonl
├── fixtures.txt                  # legacy/upcoming fixture input
├── main.py                       # cross-platform stage entrypoint
├── predict_upcoming.py           # CLI inference
├── backfill_results.py           # result reconciliation + flat-stake audit
├── run_pipeline.ps1              # Windows PowerShell orchestrator
├── Dockerfile                    # dashboard/API image
├── docker-compose.yml            # dashboard + API services
├── DEPLOYMENT.md                 # deployment and operations guide
├── .env.example                  # optional webhook/API configuration
├── requirements.txt
└── README.md
```

---

## 5. Setup

Python 3.10+ is required. The project is developed with Python 3.14, pandas 3.x,
scikit-learn 1.9 and xgboost 3.4.

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Bash / Git Bash on Windows

```bash
python -m venv venv
source venv/Scripts/activate
python -m pip install -r requirements.txt
```

No API keys or credentials are required. All paths are relative to the project
root and defined in `src/utils/config.py`.

---

## 6. Complete pipeline

Run every stage manually:

```powershell
venv\Scripts\python.exe -m src.data.make_dataset
venv\Scripts\python.exe -m src.features.build_features
venv\Scripts\python.exe -m src.models.train_model
venv\Scripts\python.exe predict_upcoming.py
venv\Scripts\python.exe backfill_results.py
venv\Scripts\python.exe -m src.predictions.reports
venv\Scripts\python.exe -m pytest
venv\Scripts\streamlit.exe run dashboard\app.py
```

Or use the cross-platform entrypoint:

```bash
python main.py data       # refresh + validate
python main.py features   # 19 features + preprocessor
python main.py train      # tune, select, test, backtest, plots
python main.py predict    # score fixtures + append history
python main.py backfill   # reconcile results + reports + audit
python main.py reports    # ledger CSV/Markdown reports
python main.py experiments # baselines, features, calibration, SHAP, monitoring
python main.py phase3     # Elo, Poisson/Dixon-Coles, ensembles, market, uncertainty
python main.py quality    # data-quality audit (JSON/CSV/Markdown)
python main.py calendar   # cache the season calendar and compare coverage
python main.py coverage   # fixture coverage + provider telemetry
python main.py sources    # historical xG/availability source audit
python main.py evaluation # full-season out-of-sample comparison with intervals
python main.py monitor    # live prediction monitoring report
python main.py update     # idempotent cycle: data, quality, coverage, predict, backfill, reports, monitoring
python main.py registry   # production registry status
python -m src.models.registry register   # register/promote/rollback/verify
python main.py api        # launch the FastAPI service
python main.py test       # pytest
python main.py dashboard  # launch Streamlit
python main.py all        # full pipeline (except dashboard/tests)
```

Or use the orchestrator:

```powershell
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1
# include the test suite:
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -RunTests
# run Phase 2 experiments as well:
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -RunExperiments
# run Phase 3 experiments as well:
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -RunPhase3
# run the automated update cycle as well:
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -RunUpdate
# skip inference:
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -SkipPrediction
```

### Stage-by-stage

| Command | What it does |
|---|---|
| `python -m src.data.make_dataset` | Refreshes seasons 23/24–26/27, validates, deduplicates, writes `data/raw/laliga_matches.csv` and `data/processed/data_quality.json`. Keeps cached rows when a download fails. |
| `python -m src.features.build_features` | Builds the 19-feature matrix, fits the preprocessor on the training season only, writes split CSVs and `data/processed/preprocessor.joblib`. |
| `python -m src.models.train_model` | Time-ordered tuning, validation selection, one-shot test, walk-forward backtest, plots, metadata/version. Registers the result as a **candidate**: live production artifacts change only through promotion (or the first-ever registration). |
| `python predict_upcoming.py [file]` | Scores fixtures and appends versioned rows to the ledger. |
| `python backfill_results.py` | Fills blank results from real data, writes `reports/`, and prints the flat-stake audit. |
| `python -m src.predictions.reports` | Regenerates ledger performance CSV/Markdown reports. |
| `python -m src.experiments.runner` | Baselines, feature ablations, calibration, explainability, experiment records and monitoring report. |
| `python -m src.experiments.phase3_runner` | Dynamic Elo, Poisson/Dixon-Coles, validation-selected ensembles, market diagnostics, uncertainty and the Phase 3 report. |
| `python -m src.data.data_quality` | Column/season missingness, duplicates, team-name and score/date validity, chronology, season sizes, odds and feature coverage. Writes JSON/CSV/Markdown. |
| `python -m src.data.fixture_coverage` | Team/season and upcoming coverage, duplicate/conflicting fixtures, provider freshness/failures. Never claims a partial provider window is complete. |
| `python -m src.data.historical_sources` | Audits xG/availability candidates with zero network requests and measures only a local timestamped extract. |
| `python -m src.evaluation.full_season` | Chronological comparison of RF, XGBoost, Elo, Poisson/Dixon-Coles, ensembles and market with bootstrap intervals and gated breakdowns. |
| `python -m src.monitoring.prediction_monitor` | Live counts, missing predictions, invalid probabilities, overdue settlements, calibration, drift, freshness and coverage with sample-size guards. |
| `python -m src.pipeline.automation` | One idempotent update cycle: refresh data, predict configured fixtures, reconcile results, regenerate reports and monitoring. Retries, logs to `logs/pipeline.log` and appends `reports/pipeline_runs.jsonl`. |
| `python -m src.models.registry` | `status`, `list`, `register`, `promote <version> --reason`, `rollback --reason`, `verify` against `models/registry/registry.json`. |
| `uvicorn src.api.main:app` | FastAPI service; OpenAPI docs at `/docs`. |
| `python -m pytest` | Runs the automated test suite. |
| `python -m streamlit run dashboard/app.py` | Launches the dashboard. |

---

## 7. Data pipeline

### Source

- Football-Data.co.uk `SP1` (La Liga) season sheets.
- Direct download first, with an optional `r.jina.ai` text-mirror fallback for
  networks that block the host.
- Current cached data: **1,259 matches**, 2023-08-11 → 2026-11-08, seasons
  `23/24`, `24/25`, `25/26` (380 each) and `26/27` (119 in progress).
- The Phase 1–3 result tables below were measured on the earlier 1,249/109
  snapshot and are retained as historical results; the Phase 6 comparison uses
  the current 1,259/119 snapshot.

### Cleaning rules

- Dates parsed and season labels derived from the date (`Aug–Jul` season).
- Team names canonicalised (`Atletico Madrid` → `Ath Madrid`,
  `Real Betis` → `Betis`).
- Stable fixture ID: `YYYYMMDD_Home_v_Away` (`match_id`).
- Exact duplicate fixtures removed; the process is idempotent.
- Rows with invalid dates, missing teams or unknown FTR codes are dropped.
- Scorelines that contradict the FTR result are dropped.
- Odds are coerced to numeric; implied probabilities are `1 / decimal odds`.
- Output is sorted chronologically and written atomically.

### Upcoming fixtures (discovered, then verified)

Fixtures are discovered automatically from the configured provider list
(`openfootball,thesportsdb` by default) and written to
`data/raw/upcoming_fixtures.csv` with stable IDs, kickoff timestamps, status and
source attribution. `python main.py calendar` caches the openfootball season
calendar, and the coverage audit then compares the stored fixtures with it:
measured on 2026-11-12, **312** fixtures covered **20/20** clubs and **260/260**
calendar fixtures (100%). Without a cached calendar the audit reports
`unknown_partial_provider_window` rather than claiming completeness. See
[`docs/FIXTURE_PROVIDERS.md`](docs/FIXTURE_PROVIDERS.md) for configuration and
merge rules.

The manual input formats below remain supported as an override, in priority
order:

1. `data/raw/upcoming_fixtures.csv`

```csv
home_team,away_team,match_date,odds_home,odds_draw,odds_away
Real Madrid,Valencia,2026-11-08,1.80,3.60,4.20
Barcelona,Sevilla,2026-11-09,,,
```

2. `fixtures.txt`

```text
# Home vs Away [| h d a] [| YYYY-MM-DD]
Real Madrid vs Valencia
Barcelona vs Sevilla | 1.55 4.10 6.50 | 2026-11-09
```

No fixture or date is generated by the project. If the date is unknown it is
shown as `TBD`; results are only reconciled from real played matches.

---

## 8. The 19 features

All rolling features use matches **before** the current kickoff. Form carries
across season boundaries because the team history is chronological.

| # | Feature | Calculation |
|---|---|---|
| 1–4 | `away_form_*` | 5-match mean of away team's goals scored, goals conceded, total shots, shots on target (previous matches only). |
| 5–6 | `away_short_*` | 3-match mean of away team's goals scored / conceded. |
| 7–10 | `home_form_*` | Same 5-match block for the home team. |
| 11–12 | `home_short_*` | Same 3-match momentum block for the home team. |
| 13 | `implied_prob_home` | `1 / odds_home` (Bet365 pre-match decimal odds). |
| 14 | `implied_prob_draw` | `1 / odds_draw`. |
| 15 | `implied_prob_away` | `1 / odds_away`. |
| 16 | `home_goal_diff_5m` | `home_form_goals_scored - home_form_goals_conceded`. |
| 17 | `away_goal_diff_5m` | `away_form_goals_scored - away_form_goals_conceded`. |
| 18 | `short_rest_advantage` | `1` when home rest ≥ 5 days and away rest ≤ 3 days, else `0`. |
| 19 | `home_advantage` | Constant `1` for the home side. |

Why these 19: they capture recent attacking/defensive form, short-term
momentum, venue, schedule congestion and the market's pre-match assessment. No
additional features were added without a statistical justification. The market
probabilities are deliberately included because they are a strong, legitimate
pre-kickoff baseline; the model is therefore not an independent betting edge.

Median imputation and standard scaling are fitted on the training window only.
The same fitted `FeaturePreprocessor` transforms validation, test and live
fixtures.

---

## 9. Model evaluation

### Splits

| Window | Seasons | Rows (current data) |
|---|---|---:|
| Train | `23/24` + `24/25` | 760 |
| Validation | `25/26` | 380 |
| Untouched test | `26/27` | 109 (season in progress) |

### Tuning

- Random Forest and XGBoost each search 24 random candidates.
- `TimeSeriesSplit(n_splits=3)` inside the training window — never shuffled.
- Objective: multiclass log-loss.
- Class imbalance: Random Forest uses `class_weight="balanced_subsample"`;
  XGBoost receives inverse-frequency sample weights.
- Balancing is evaluated rather than assumed: it raises draw recall but can
  worsen probability calibration, which the metrics expose.

### Selection and final scoring

1. Tune both models on the training window.
2. Score both on validation.
3. Select the lowest **validation log-loss** (Brier score breaks ties).
4. Refit both models on train + validation.
5. Score both **once** on the test season. The winner was already fixed.

### Walk-forward backtest

Rolling-origin folds over earlier seasons, excluding the untouched test season:

- Train `23/24` → test `24/25`
- Train `23/24` + `24/25` → test `25/26`

Each fold refits its own preprocessor and model.

### Metrics

- Accuracy
- Multiclass log-loss
- Multiclass Brier score
- Per-class precision, recall, F1 and support
- **Draw recall** (reported separately)
- Confusion matrix (counts + row-normalised percentages)
- Calibration curves and expected calibration error (ECE)
- Season-by-season and walk-forward results

---

## 10. Measured results (current data)

These values come from the artifacts produced by the current pipeline run
(model version `random_forest-20261105T224238Z`). They are not claims of future
performance.

### Validation (`25/26`, 380 matches)

| Model | Accuracy | Log-loss | Brier | Macro F1 | Draw recall | ECE |
|---|---:|---:|---:|---:|---:|---:|
| **Random Forest** | 45.8% | **1.0131** | **0.6027** | 0.4499 | 44.1% | 0.0967 |
| XGBoost | 46.6% | 1.0526 | 0.6268 | 0.4477 | 37.6% | 0.1093 |

### Untouched test (`26/27`, 109 matches)

| Model | Accuracy | Log-loss | Brier | Macro F1 | Draw recall | ECE |
|---|---:|---:|---:|---:|---:|---:|
| **Random Forest** (selected) | 50.5% | **0.9837** | **0.5811** | 0.5048 | 55.6% | 0.0888 |
| XGBoost | **51.4%** | 1.0224 | 0.5985 | 0.4861 | 29.6% | 0.1155 |

XGBoost has slightly higher test **accuracy**, but Random Forest was selected on
validation **log-loss** and also has better test log-loss, Brier score, macro F1
and draw recall. This is the intended demonstration that accuracy alone is not a
valid model-selection criterion.

### Walk-forward

| Fold | Test season | Model | Accuracy | Log-loss | Brier |
|---|---|---:|---:|---:|---:|
| 1 | `24/25` | Random Forest | 53.2% | 0.9926 | 0.5895 |
| 1 | `24/25` | XGBoost | 46.3% | 1.0793 | 0.6438 |
| 2 | `25/26` | Random Forest | 45.8% | 1.0131 | 0.6027 |
| 2 | `25/26` | XGBoost | 46.6% | 1.0526 | 0.6268 |

**Caveats:** the test season has only 109 of 380 matches; three-way football is
inherently noisy; draws are the hardest class; and the model includes market
odds, so these metrics should not be read as an independent betting edge. Small
differences between models are within noise.

### Phase 2: fixed baselines vs learned models

Market odds are a strong pre-kickoff baseline. On this test window the raw
market probabilities beat both learned models on log-loss; that is reported as a
negative result rather than hidden.

| Model | Validation log-loss | Test log-loss | Test accuracy | Test draw recall |
|---|---:|---:|---:|---:|
| **Market odds** | **0.9637** | **0.9192** | **56.9%** | 0.0% |
| Logistic regression | 1.0295 | 0.9608 | 50.5% | 51.9% |
| Random Forest (selected ML) | 1.0131 | 0.9837 | 50.5% | 55.6% |
| XGBoost | 1.0526 | 1.0224 | 51.4% | 29.6% |
| Historical frequencies | 1.0509 | 1.0641 | 45.9% | 0.0% |
| Home always / majority | 14.1064 | 14.9562 | 45.9% | 0.0% |

**Feature experiments.** Seven candidate groups were tested on validation and two
walk-forward folds, with the test season withheld. None cleared the strict rule
(lower validation log-loss, lower walk-forward log-loss and a 95% paired
bootstrap upper bound below zero for both tree models). Head-to-head and venue
form were closest, but every candidate's validation interval included zero;
head-to-head's upper bounds were +0.0011 (RF) and +0.0030 (XGBoost), so the
canonical 19-feature set is retained. xG was not tested because
the cached extract has no reliable historical xG column.

**Calibration.** Sigmoid was selected on a chronological validation holdout for
both models. On test it improved Random Forest log-loss from 0.9837 to 0.9641 and
ECE from 0.0888 to 0.0548, but draw recall collapsed from 55.6% to 0%. XGBoost
showed the same pattern (1.0224 → 0.9860 log-loss, 29.6% → 0% draw recall). The
calibrator is therefore **not deployed automatically**; improving probability
scores while losing an entire outcome class is a decision-quality regression.

**Explainability.** Market-implied probabilities dominate permutation importance
for both models, followed by shot-volume and goal form. XGBoost TreeSHAP shows
the same ranking. These are descriptive associations on validation, not causal
effects, and correlated features share credit.

**Monitoring.** The current ledger has 10 pending predictions and no settled
rows, so performance and calibration checks correctly return
`insufficient_data`. Data quality and feature drift are healthy (largest recent
shift +0.61 SD). A `confirmed` degradation label requires at least 100 settled
predictions, a material performance gap and a supportive bootstrap interval.

### Phase 3: dynamic Elo, Poisson/Dixon-Coles, ensembles and market analysis

Phase 3 adds fundamentally different football models without changing the
production model. The full measured report is in
[`reports/phase3_report.md`](reports/phase3_report.md).

**Dynamic Elo.** Ratings start at 1500, use a fixed 60-point home advantage, a
World Football Elo margin-of-victory multiplier, and a rating deviation that
falls with matches and rises with inactivity. All fixtures on a date read the
state before that date. The latest descriptive snapshot has Barcelona first
(1784.1), Real Madrid second (1754.7) and Atletico Madrid third (1663.2).

**Poisson/Dixon-Coles.** Attack and defence are multiplicative, time-decayed
and shrunk toward the league average. The validation-selected settings were a
365-day half-life and prior strength 2; the fitted home/away goal bases were
1.532/1.141 and the Dixon-Coles rho was +0.010. Sparse or unknown teams fall
back explicitly to the league average rather than borrowing another club's
rating.

**Validation (`25/26`, 380 matches)**

| Model | Accuracy | Log-loss | Brier | Draw recall | ECE |
|---|---:|---:|---:|---:|---:|
| **Market odds** | **55.0%** | **0.9637** | **0.5710** | 0.0% | 0.0449 |
| Ensemble (validation weights) | 52.9% | 0.9756 | 0.5782 | 4.3% | 0.0487 |
| Dynamic Elo | 52.4% | 0.9803 | 0.5812 | 0.0% | 0.0377 |
| Ensemble (equal weights) | 51.6% | 0.9820 | 0.5824 | 20.4% | 0.0610 |
| Dixon-Coles | 52.6% | 0.9922 | 0.5886 | 0.0% | 0.0532 |
| Independent Poisson | 52.6% | 0.9922 | 0.5885 | 0.0% | 0.0455 |
| Random Forest (production reference) | 45.8% | 1.0131 | 0.6027 | 44.1% | 0.0967 |

**Test (`26/27`, 109 matches)**

| Model | Accuracy | Log-loss | Brier | Draw recall | ECE |
|---|---:|---:|---:|---:|---:|
| **Market odds** | **56.9%** | **0.9192** | **0.5418** | 0.0% | 0.0891 |
| Dynamic Elo | 55.0% | 0.9244 | 0.5431 | 0.0% | 0.0635 |
| Ensemble (validation weights) | 55.0% | 0.9279 | 0.5461 | 0.0% | 0.0866 |
| Ensemble (equal weights) | 50.5% | 0.9440 | 0.5567 | 3.7% | 0.0964 |
| Independent Poisson | 55.0% | 0.9572 | 0.5661 | 0.0% | 0.0813 |
| Dixon-Coles | 55.0% | 0.9574 | 0.5662 | 0.0% | 0.0774 |
| Logistic regression | 50.5% | 0.9608 | 0.5679 | 51.9% | 0.1144 |
| Random Forest (production) | 50.5% | 0.9837 | 0.5811 | 55.6% | 0.0888 |
| XGBoost | 51.4% | 1.0224 | 0.5985 | 29.6% | 0.1155 |

**Ensemble.** The fixed simplex grid selected XGBoost 0.20, Dynamic Elo 0.70 and
Independent Poisson 0.10 (Random Forest 0.00). Validation log-loss was 0.9756
versus 0.9820 for equal weights, but the ensemble still finished behind market
odds on both validation and test. It is not deployed.

**Walk-forward.** Market odds were best in both earlier folds: 0.9534 on `24/25`
and 0.9637 on `25/26`. Dynamic Elo was second in both (0.9924 and 0.9803), ahead
of Random Forest (0.9926 and 1.0131). The pattern is stable, not a one-fold
accident.

**Market analysis.** Mean overround was 5.40% (`23/24`), 5.65% (`24/25`),
5.61% (`25/26`) and 5.50% (`26/27`). A validation-fitted stack of market plus
RF/XGB/Elo/Poisson probabilities scored 0.9167 test log-loss versus 0.9192 for
the market, but the paired bootstrap interval for the difference was
[-0.0299, +0.0279] and its Brier score was slightly worse (0.5421 vs 0.5418).
There is **no credible evidence** that the models add information beyond the
market on this window, and no betting edge is claimed.

**Uncertainty.** On the 109 test fixtures, mean confidence was 51.6%, mean
normalised entropy 0.900 and mean model agreement 0.878; 76 fixtures were
classified high uncertainty. Entropy correlated positively with realised row
log-loss (Spearman 0.467), which supports the diagnostic but does not make the
level a calibrated interval.

**Decision.** No experimental model is promoted automatically. The production
Random Forest is preserved because the challengers did not beat the market
benchmark and the test window is small. Dynamic Elo and the validation-weighted
ensemble do improve on the current Random Forest reference on validation and
test, so they are the strongest candidates for a controlled review, but they
still trail market odds. The Phase 3 run is explicit (`python main.py phase3`)
and writes a separate experiment record for every new model.

### Phase 6: data quality, coverage and full-season evaluation

Phase 6 adds no model architecture. It measures the data and the existing
models more strictly, and it separates validation, test and live performance.

**Data quality.** The current 1,259-match cache has 0 duplicate fixtures, 0
invalid scores/dates, 0 score/result contradictions and 0 missing odds. All 19
features are present for **98.57%** of rows; the largest per-feature missing
fraction is **1.19%** (early-season form warm-up), below the 5% threshold.
Reports: `reports/data_quality.json`, `reports/data_quality_by_column.csv`,
`reports/data_quality_by_season.csv`, `reports/data_quality.md`.

**Fixture coverage.** After Phase 7 the store holds **312** upcoming fixtures
from `openfootball` (311) and `thesportsdb` (1), covering **20 of 20** clubs with
0 duplicate IDs, 0 conflicting kickoffs and 0 missing dates. Verified against
the public `2026-27/es.1.json` calendar: **260** fixtures still to play, **260
matched**, **0 missing** (100%), and a schedule completeness of
`verified_against_calendar`. One fixture disagrees on the date (Athletic Bilbao
vs Espanyol: TheSportsDB 2026-11-20, calendar 2026-11-22) and is reported as
`rescheduled` rather than silently resolved. TheSportsDB alone returns 1 fixture
because its free `eventsnextleague` endpoint is capped at one event; the
calendar provider exists to make completeness measurable instead of assumed.
The reference season `25/26` contains Girona, Mallorca and Oviedo; `26/27`
contains La Coruna, Malaga and Santander, a real promotion/relegation change.
Reports: `reports/fixture_coverage.json`, `reports/fixture_coverage.md`.
Provider freshness/failure telemetry is written to
`data/processed/fixture_provider_status.json` for scheduled runs.

**Historical xG and player availability.** No timestamped local extract exists,
so both groups are reported as **unavailable**; no values are simulated and no
model uses them. The source audit made zero network requests. See
[`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

**Full-season evaluation (`26/27`, 119 matches).** The full measured report is
in [`docs/MODEL_EVALUATION.md`](docs/MODEL_EVALUATION.md) and
`reports/full_season_evaluation.md`.

| Model | Validation log-loss | Test log-loss | Test 95% CI | Test Brier | Test accuracy | Test draw recall |
|---|---:|---:|---|---:|---:|---:|
| Market odds | **0.9637** | **0.9171** | 0.8379–1.0008 | **0.5405** | **56.3%** | 0.0% |
| Dynamic Elo | 0.9803 | 0.9255 | 0.8524–1.0023 | 0.5440 | 54.6% | 0.0% |
| Ensemble (validation weights) | **0.9760** | 0.9257 | 0.8503–1.0054 | 0.5443 | 55.5% | 0.0% |
| Random Forest (production) | 1.0134 | 0.9624 | 0.8834–1.0512 | 0.5672 | 53.8% | 58.1% |
| XGBoost | 1.0559 | 1.0036 | 0.8853–1.1354 | 0.5868 | 53.8% | 32.3% |

Validation selected the ensemble with weights Random Forest 0.00, XGBoost 0.20,
Dynamic Elo 0.70 and Independent Poisson 0.10 (validation log-loss 0.9760 vs
0.9826 for equal weights). No learned model beats market odds on test log-loss,
and the intervals overlap. The production model and registry snapshot are
unchanged; no challenger is promoted. Walk-forward excludes the final test
season.

**Live monitoring.** The ledger currently has 17 prediction rows over 3 unique
fixtures and **0 settled** rows. Live calibration/log-loss/Brier are therefore
reported as `insufficient_data` with `live_performance_reliable=false`; no live
conclusion is presented. Reports: `reports/prediction_monitoring.json` and
`reports/prediction_monitoring.md`.

---

## 11. Generating predictions

```bash
python predict_upcoming.py
python predict_upcoming.py my_fixtures.txt
```

Output includes home/draw/away probabilities (summing to 100%), the most likely
outcome, confidence, expected goals, most-likely scorelines, recent form, rest
context and value-bet flags against the
supplied or historical average market probabilities.

Every valid prediction appends a row to
`data/processed/prediction_history.csv`:

| Column | Meaning |
|---|---|
| `fixture_id` | Stable `YYYYMMDD_Home_v_Away` (or `TBD_...` when date unknown). |
| `match_date` | Kickoff date when known. |
| `home_team`, `away_team` | Canonical club names. |
| `predicted_at` | Prediction timestamp (UTC). |
| `model_name`, `model_version` | Model identity for reproducibility. |
| `model_prob_home/draw/away` | Model probabilities (stored to 6 dp). |
| `predicted_outcome` | `Home Win`, `Draw` or `Away Win`. |
| `confidence` | Probability of the selected outcome. |
| `market_prob_home/draw/away` | Market benchmark used at prediction time. |
| `actual_outcome` | `2` home, `1` draw, `0` away — blank until played. |
| `actual_home_goals`, `actual_away_goals` | Real scoreline once settled. |
| `correct` | `1`/`0` once settled. |
| `settled_at` | Reconciliation timestamp. |

Legacy ledgers using `date_predicted` are migrated on read; exact duplicate rows
are removed, but distinct snapshots are preserved.

### Result reconciliation

```bash
python backfill_results.py
```

- Only blank result fields are filled.
- Exact fixture ID is preferred; otherwise the same home/away pairing is matched
  with a match date on/after the prediction date and on/before today.
- Earlier meetings and future fixtures are never used.
- Settled rows are immutable.
- The audit re-derives value bets with the same logic as inference and reports
  hypothetical flat-stake P&L, win rate and yield. This is a ledger audit, not a
  recommendation.

---

## 12. Dashboard

Launch:

```powershell
venv\Scripts\streamlit.exe run dashboard\app.py
# or
venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Pages:

- **Home** — project overview, latest metrics, upcoming fixtures, recent
  predictions, ledger accuracy.
- **Match Predictions** — fixture probabilities, confidence, form, rest context,
  expected goals, top scorelines, an uncertainty level, a team filter, value-bet
  flags and a custom matchup builder.
- **Model Performance** — validation/test comparison, confusion matrices,
  calibration curves, per-class metrics, season results, walk-forward,
  feature importance.
- **Team Analytics** — rolling form, goals, home/away splits, season history and
  side-by-side team comparison.
- **Prediction History** — filters, settled/pending status, accuracy trends,
  performance by season/outcome/team/model version, CSV export.
- **Model Monitoring** — data-quality warnings, staleness, ECE over time,
  prediction distribution, standardized feature drift, and evidence-aware
  diagnostic verdicts (`ok`, `insufficient_data`, `warning`, `confirmed`).
- **Model Intelligence** — fixed baselines, feature-group experiments and
  confidence intervals, validation-only calibration comparisons, permutation /
  tree / TreeSHAP importance, fold stability, dynamic Elo ratings and rating
  history, Poisson attack/defence estimates, expected goals, most-likely
  scorelines, scoreline heatmaps, Phase 3 model comparison, ensemble weights,
  market-versus-model differences and prediction uncertainty, plus the
  experiment ledger.
- **Model Operations** — production version and its validation/test evidence,
  every registered version, promotion eligibility, update-cycle history,
  monitoring verdicts and artifact health.

The dashboard is served by Streamlit 1.50+ and all eight pages are covered by
smoke tests. It is a local analytics application; use Docker or Streamlit
Community Cloud to host it.

---

## 13. API

```bash
python main.py api
# or
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | artifact presence and production version |
| GET | `/model` | production metadata, metrics and feature contract |
| GET | `/fixtures` | configured upcoming fixtures with full predictions |
| POST | `/predict` | one fixture: probabilities, expected goals, scorelines |
| GET | `/teams` | canonical team names |
| GET | `/teams/{team}/stats` | record, form, Elo rating, attack/defence |
| GET | `/performance` | ledger summary and grouped performance |
| GET | `/registry` | registered versions and production designation |
| GET | `/runs` | recent update-cycle records |

The API imports the same `predict_fixtures` engine, ledger functions and model
statistics as the CLI and dashboard; it does not re-implement a second forecast.
Unknown teams return `422` (or `404` for team statistics), partial odds and
malformed requests return `422`, and missing artifacts or an unreadable registry
return `503`. OpenAPI documentation is served at `/docs`.

---

## 14. Operations and deployment

One update cycle is `data → predict → backfill → reports → monitoring` and is
idempotent: downloads merge and de-duplicate, predictions are keyed by
fixture/timestamp/version, and reconciliation only fills blank results.

```bash
python main.py update
python -m src.pipeline.automation --steps data,backfill,monitoring --retries 2
```

Failures are logged to `logs/pipeline.log`, appended to
`reports/pipeline_runs.jsonl`, and posted to `LALIGA_ALERT_WEBHOOK` when that
environment variable is set. Daily scheduling is included as
`.github/workflows/update.yml`; `DEPLOYMENT.md` documents cron and Windows Task
Scheduler equivalents.

Model promotion is never automatic. Training registers a candidate and leaves live
inference pinned to the promoted registry snapshot, so a newer experiment cannot
change production by itself:

```bash
python -m src.models.registry register
python -m src.models.registry status
python -m src.models.registry promote <version> --reason "validation evidence"
python -m src.models.registry rollback --reason "observed regression"
python -m src.models.registry verify
```

Deployment artifacts: `Dockerfile`, `docker-compose.yml` (dashboard on 8501, API
on 8000, both with health checks), `.github/workflows/ci.yml` (byte-compile, full
test suite, registry verification), `.env.example`, and `DEPLOYMENT.md` (local,
Docker, Streamlit Community Cloud, scheduling, API and security notes).

Verification performed for this phase: full test suite, Phase 3 regeneration,
engine predictions with expected goals/scorelines, ledger reconciliation, all
eight Streamlit pages via `AppTest`, every API endpoint via `TestClient`, and a
fresh-virtualenv dependency install. Nothing was deployed to a public host from
this environment, and no deployment success is claimed.

---

## 15. Tests

```bash
python -m pytest
```

Current result: **201 passed**.

Coverage includes:

- Rolling feature calculations and current-match exclusion.
- Feature invariance when future matches are truncated (leakage test).
- Missing-value imputation fitted on training data only.
- Team normalization, fixture IDs, deduplication and validation.
- Chronological split disjointness and ordering.
- Walk-forward folds train only on earlier seasons.
- Metric correctness (log-loss, Brier, draw recall, calibration, confusion).
- Probability validity and predicted-label consistency.
- Ledger migration, append-only behaviour, reconciliation and immutability.
- End-to-end small training/selection run using synthetic data.
- Candidate-feature leakage when future matches are truncated.
- Fixed-baseline probability validity and calibration output validity.
- Unique experiment IDs across calibration variants and index reload.
- Monitoring states, including insufficient-data handling.
- Permutation-importance completeness on synthetic data.
- Elo chronological updates, uncertainty reduction and future-truncation invariance.
- Poisson scoreline normalization, home/draw/away aggregation and sparse-team fallback.
- Dixon-Coles low-score adjustment and pre-match Poisson feature leakage checks.
- Ensemble weight validity and the validation-only no-worse-than-equal guarantee.
- Market implied-probability extraction and overround removal.
- Prediction reproducibility for Elo and Poisson models.
- Registry snapshots, feature hashes, evidence-gated promotion and rollback.
- Update-cycle retries, failure handling, append-only run log and optional alerts.
- Engine expected goals, scoreline normalization, uncertainty fields and reproducibility.
- FastAPI health/model/predict/teams/performance/registry endpoints and error codes.
- Streamlit smoke tests for Home, Match Predictions and Model Operations.
- Data-quality anomaly detection, configurable thresholds and strict JSON/CSV/Markdown reports.
- Fixture coverage, duplicate/conflict detection, partial-provider preservation, stable IDs and provider telemetry.
- OpenFootball calendar normalisation, 20/20 team-name canonicalisation, full-season provider scopes, multi-provider priority lists and the 14-day/whole-season calendar comparison (missing, rescheduled, unexpected, per-team completeness).
- Historical xG/availability source audit with time-aware coverage and no network requests.
- Full-season evaluation protocol, bootstrap intervals, walk-forward exclusion of the final test season and small-sample breakdown refusal.
- Prediction monitoring counts, missing predictions, invalid probabilities and explicit insufficient-sample states.

Tests use a deterministic synthetic four-season dataset, so they require no
network access and do not modify the real ledger.

---

## 16. Reproducibility

- Global random seed: `42`.
- Season splits, feature names, windows and paths are centralized in
  `src/utils/config.py`.
- Model metadata records the version, seed, parameters, split sizes/dates,
  validation/test metrics and feature list.
- `models/saved_models/model_metadata.json` is the source of truth for the
  dashboard's selected model.
- Preprocessing is serialized with the model, so live predictions use the exact
  training statistics.

Re-running `train_model` produces a new timestamped model version; historical
ledger rows keep the old version, so performance can be compared across model
generations.

---

## 17. Known limitations

- **Fixture discovery is calendar-dependent.** Coverage can only be *verified*
  after `python main.py calendar` caches the openfootball season; that dataset is
  community-maintained and refreshed about once a day, so it can lag official
  announcements and may miss recent results (a past fixture then stays
  `scheduled`, which the audit shows but never treats as a result). Kickoff clock
  times are published for only part of the season; a date-only record merges into
  a timed one from another provider rather than being invented.
- **TheSportsDB's free tier returns one fixture.** `eventsnextleague` is capped
  at 1 event (20 on premium), so it cannot supply league-wide coverage alone.
  API-Football needs a key and its free plan does not cover the current season.
- **Registry promotion uses one validation window.** A challenger that wins on
  validation can still lose on test; the registry records both and never promotes
  automatically.
- **Scheduled updates do not commit artifacts.** The GitHub Actions workflow uploads
  reports and logs as workflow artifacts; accumulating a ledger needs durable
  storage or an explicit commit step.
- **Streamlit Community Cloud has an ephemeral filesystem.** Ledger rows written in
  the hosted app do not survive a restart.
- **The API is read-only and unauthenticated.** It cannot write the ledger or
  promote models; add TLS and authentication before exposing it publicly.
- **Registry snapshots duplicate artifacts.** Each version keeps its own model,
  preprocessor and metadata copy (about 1 MB per tree model), which is the cost of
  a working rollback path.
- **In-progress test season.** `26/27` currently has 109 of 380 matches, so test
  metrics are provisional.
- **Small historical sample.** Four seasons and two walk-forward folds are a
  limited basis for strong claims.
- **Market odds are a feature.** This is realistic and strong, but it means the
  model is not an independent edge over the market and may inherit closing-line
  bias from the Bet365 snapshot.
- **Draws remain difficult.** Draw recall and calibration are materially worse
  than for home/away outcomes.
- **No deployed post-calibration.** Calibration is measured on a validation
  holdout and compared once on test, but no calibrator is applied to live
  predictions. The selected sigmoid variants improve log-loss yet collapse draw
  recall to zero, so deployment needs an explicit decision rule first.
- **Limited feature set.** No expected goals (xG), lineups, injuries, travel or
  managerial changes. Elo, strength of schedule, venue form, rest detail and
  head-to-head were tested and rejected under the strict Phase 2 rule; that rule
  can reject a real but small effect.
- **Elo prior is uniform.** Every club starts at 1500 with rating deviation 350;
  promoted clubs are not given a separate prior, so early-season ratings are
  less informative.
- **Poisson assumptions are strong.** Goals are conditionally Poisson with
  multiplicative attack/defence and no red-card, lineup, game-state or shot-quality
  adjustment. The Dixon-Coles correction is fitted on the training likelihood,
  not a separate validation window.
- **Ensemble weights can overfit validation.** The equal-weight reference is
  reported, and the selected ensemble is not deployed.
- **No incremental market evidence.** A validation-fitted market + model stack
  had a test log-loss point estimate slightly below the market, but its bootstrap
  interval included zero and its Brier score was slightly worse. This is not a
  demonstrated edge.
- **Validation reuse.** Hyperparameters and model selection both use the
  validation season; a fully nested CV would give a stricter estimate.
- **Financial audit is hypothetical.** Flat-stake P&L ignores stakes limits,
  commission, line movement and availability. It is not betting advice.
- **No scheduled retraining.** The pipeline is manual or script-driven. The
  experiment runner is explicit (`python main.py experiments`), not part of
  every `main.py all` run.
- **Windows-first orchestration.** `run_pipeline.ps1` is PowerShell; the Python
  commands are cross-platform.
- **Provider fixture coverage is currently insufficient.** TheSportsDB returned
  only one upcoming fixture covering two teams. The audit reports the window as
  incomplete and does not fabricate the remaining schedule.
- **No historical xG or player availability.** No timestamped local extract
  exists; both optional feature groups are unavailable and untrained.
- **Live performance is unmeasured.** The ledger has 0 settled predictions, so
  live metrics are explicitly `insufficient_data`, not evidence.
- **Linux CI is configured, not executed here.** `ci.yml` runs on Ubuntu and
  Windows, but this Windows environment cannot execute GitHub Actions; no Linux
  pass is claimed.
- **Phase 1–3 tables use an earlier data snapshot.** They were measured on
  1,249 matches with 109 test rows; the Phase 6 evaluation uses 1,259/119.

---

## 18. Suggested next improvements

1. Add an automated fixtures/results provider so upcoming matches and
   reconciliation do not depend on manual input.
2. Add xG, lineups and availability data, with ablation tests before inclusion.
3. Decide whether calibrated probabilities or draw recall matter more for the
   product, then either deploy a constrained calibrator or keep the raw model.
4. Use nested walk-forward evaluation for hyperparameters as well as models.
5. Run and extend the configured GitHub Actions CI (Ubuntu + Windows), including the registry-integrity check on scheduled updates.
6. Add a scheduler/container for periodic data refresh and retraining.
7. Revisit feature candidates when a reliable xG source is available, using the
   same validation + walk-forward + bootstrap rule.
8. Extend the Poisson model with a hierarchical prior or promoted-team adjustment
   once more seasons are available.
9. Add a genuinely independent closing-line or flat-stake audit before making any
   betting-edge claim.
10. Require nested walk-forward evidence, not a single validation season, before a
    registry promotion.
11. Cache Poisson/Elo fits in the API once prediction latency matters.
12. Add authentication and rate limiting before exposing the API publicly.
13. Persist generated artifacts from the scheduled workflow to durable storage.
