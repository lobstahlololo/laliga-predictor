# 🇪🇸 La Liga Match Outcome Predictor & Value Bet Core

An end-to-end, automated machine learning platform engineered to predict La Liga match outcomes (Win/Draw/Loss). The architecture implements robust, leakage-safe rolling time windows, custom regularized multiclass tree structures, an automated value betting dashboard, and an idempotent post-match financial audit ledger.

---

## 🛠️ Architecture & Core Components

```text
       [ 1. Ingestion Layer ]   -->   Reads historical & ongoing seasons (Direct URL / Jina Text Mirror Proxy)
                │
                ▼
       [ 2. Feature Engine ]    -->   19-feature matrix (3m/5m rolling metrics + Goal Differentials + Rest Gaps)
                │
                ▼
       [ 3. Modeling Sandbox ]  -->   XGBoost (max_delta_step) & Random Forest (balanced_subsample) via CV Log-Loss
                │
                ▼
       [ 4. Inference Hub ]     -->   Fuzzy-name team resolution, manual/file batch ingestion, UTF-8 console graphs
                │
                ▼
       [ 5. Settlement Logs ]   -->   Idempotent financial ledger backfilling, flat-stake tracking, Yield % metrics
```

### 1. Advanced Feature Pipeline (`src/features/build_features.py`)
*   **Momentum Windows:** Computes 3-match and 5-match rolling means for goals scored, goals conceded, total shots, and shots on target using a `shift(1)` boundary to eliminate forward data leakage.
*   **Goal Dominance & Rest Gaps:** Tracks team efficiency via 5-match rolling goal differentials and calculates a binary short-rest feature (`short_rest_advantage = 1` if home team rest \(\ge\) 5 days and away team rest \(\le\) 3 days).
*   **Market Implication:** Ingests raw bookmaker odds and maps them to clean scaled feature columns representing historical implied market probability densities.

### 2. Tuned Multiclass Brains (`src/models/train_model.py`)
*   **Optimization Target:** Optimized strictly on multiclass **Log-Loss (`neg_log_loss`)** rather than Accuracy, forcing the model to respect and accurately evaluate draw probabilities instead of collapsing into majority class prediction.
*   **Imbalance Resolution:** Random Forest is regularized using `class_weight="balanced_subsample"`. XGBoost is dynamically balanced using inverse-frequency sample mapping and a hyperparameter search across `max_delta_step` (1–5) to prevent gradient updates from exploding on sparse Draw instances.

### 3. Production Inference Engine (`predict_upcoming.py`)
*   **Fault-Tolerant Resolution:** Utilizes an accent-insensitive alias map and `difflib` fuzzy text-matching to map user input anomalies back to the single source of truth database framework.
*   **Value Edge Detector:** Flags an explicit visual marker (`⭐ [VALUE BET DETECTED]`) if the model's calculated outcome probability exceeds the bookmaker's implied probability line by an absolute threshold of \(> 5.0\%\).

### 4. Post-Match Financial Settlement (`backfill_results.py`)
*   **Idempotent Auditing:** Programmatically cross-references the active `prediction_history.csv` against real match results as they finish, recording numerical outcome states (0, 1, 2).
*   **ROI Metrics Engine:** Tracks hypothetical flat-stake (\$10) deployment performance, logging Win Rate %, Total Profit/Loss, Capital Risked, and overall **Yield %**. Enforces strict chronological safety constraints (`match_date <= today`).

---

## 📂 Repository Structure

```text
laliga-predictor/
├── data/
│   ├── raw/               # Concatenated real match data (laliga_matches.csv)
│   └── processed/         # Processed CSV feature tables & feature_scaler.joblib
├── models/
│   └── saved_models/      # Serialized production pipeline weights (best_model.pkl)
├── src/
│   ├── data/              # Multi-season web parsing & text proxy pipelines
│   ├── features/          # 19-Feature matrix extraction engines
│   ├── models/            # Hyperparameter optimization & matrix visualizers
│   └── visualization/     # Local database logs & history compilers
├── backfill_results.py    # Monday post-match PnL settlement module
├── predict_upcoming.py    # Weekend interactive fixture inference script
├── run_pipeline.ps1       # 5-Stage PowerShell Orchestration Wrapper
├── requirements.txt       # Hard-pinned environment dependencies
└── README.md              # System documentation asset
```

---

## 🚀 Execution & Quickstart

### Environment Initialization
Ensure you have Python 3.10+ and Node.js (for global Freebuff access) installed on your system.

```powershell
# Create the virtual environment configuration
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install hard-pinned production dependencies
pip install -r requirements.txt
```

### Running the Orchestration Framework
The entire codebase has been wrapped inside a unified, validation-checked PowerShell orchestrator script. To scrape new data, preprocess features, cross-validate hyperparameters, predict upcoming games, and audit your financial ledger, execute:

```powershell
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1
```
