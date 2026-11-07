<#
.SYNOPSIS
    La Liga predictor - end-to-end pipeline orchestrator.

.DESCRIPTION
    Runs the modular pipeline with a hard stop on failure:
      0. Verify the virtual environment and core dependencies.
      1. src.data.make_dataset      - refresh real data, validate, de-duplicate.
      2. src.features.build_features - 19 leakage-safe features + preprocessor.
      3. src.models.train_model      - time-ordered tuning, validation-only
                                        selection, one-shot test evaluation,
                                        walk-forward backtest and plots.
      4. predict_upcoming.py         - score fixtures.txt / upcoming_fixtures.csv
                                        and append versioned history.
      5. backfill_results.py         - optional: reconcile real results and audit
                                        the flat-stake ledger.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File run_pipeline.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -SkipPrediction
#>

param(
    [switch]$SkipPrediction,
    [switch]$RunTests,
    [switch]$RunExperiments,
    [switch]$RunPhase3,
    [switch]$RunUpdate
)

# --- UTF-8 console so dashboard glyphs (box drawing, star) render natively ---
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
    Write-Host "[warn] Could not set UTF-8 console encoding." -ForegroundColor Yellow
}

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = Get-Location }

$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$Stages = @(
    @{ Name = "make_dataset";    Args = @("-m", "src.data.make_dataset") },
    @{ Name = "build_features";  Args = @("-m", "src.features.build_features") },
    @{ Name = "train_model";     Args = @("-m", "src.models.train_model") }
)

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host ">>> $Message" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# Stage 0: environment verification
# ---------------------------------------------------------------------------
Write-Step "Stage 0: verifying environment state"

if (-not (Test-Path $Python)) {
    Write-Host "[FAIL] Python not found at $Python - create the venv first." -ForegroundColor Red
    exit 1
}
Write-Host "[ok] interpreter: $Python"

& $Python -c "import pandas, numpy, sklearn, xgboost, joblib, matplotlib, requests"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] core dependencies missing - pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}
Write-Host "[ok] core dependencies importable"
Write-Host "[ok] environment verified" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Stages 1-3
# ---------------------------------------------------------------------------
$total = $Stages.Count
for ($i = 0; $i -lt $total; $i++) {
    $stage = $Stages[$i]
    Write-Step "Stage $($i + 1)/$($total): $($stage.Name)"
    $stageArgs = $stage.Args
    & $Python @stageArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] $($stage.Name) exited with code $LASTEXITCODE - pipeline halted." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "[ok] $($stage.Name)" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Optional Phase 2 model-intelligence suite
# ---------------------------------------------------------------------------
if ($RunExperiments) {
    Write-Step "Running Phase 2 experiments (baselines, features, calibration, explainability)"
    & $Python -m src.experiments.runner
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] Phase 2 experiments failed - pipeline halted." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "[ok] Phase 2 artifacts and report generated" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Optional Phase 3 suite
# ---------------------------------------------------------------------------
if ($RunPhase3) {
    Write-Step "Running Phase 3 experiments (Elo, Poisson/Dixon-Coles, ensembles, market analyses)"
    & $Python -m src.experiments.phase3_runner
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] Phase 3 experiments failed - pipeline halted." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "[ok] Phase 3 artifacts and report generated" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Optional automated update cycle (data -> predict -> backfill -> reports -> monitoring)
# ---------------------------------------------------------------------------
if ($RunUpdate) {
    Write-Step "Running automated update cycle (idempotent; retries enabled)"
    & $Python -m src.pipeline.automation --retries 2
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[warn] update cycle reported a failure - see logs/pipeline.log and reports/pipeline_runs.jsonl." -ForegroundColor Yellow
    } else {
        Write-Host "[ok] update cycle complete" -ForegroundColor Green
    }
}

# ---------------------------------------------------------------------------
# Optional test suite
# ---------------------------------------------------------------------------
if ($RunTests) {
    Write-Step "Running automated test suite"
    & $Python -m pytest
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] tests failed - pipeline halted." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "[ok] tests passed" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Stage 4: inference
# ---------------------------------------------------------------------------
if (-not $SkipPrediction) {
    Write-Step "Stage 4: prediction ledger update"
    & $Python "predict_upcoming.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[warn] prediction exited with code $LASTEXITCODE (no valid fixtures or incomplete history)." -ForegroundColor Yellow
    } else {
        Write-Host "[ok] predictions logged" -ForegroundColor Green
    }
}

# ---------------------------------------------------------------------------
# Stage 5 (optional, prompted): settle results + flat-stake audit
# ---------------------------------------------------------------------------
Write-Step "Stage 5 (optional): settlement & performance audit"
$answer = ""
try {
    $answer = Read-Host "Reconcile played results and audit the prediction ledger? (y/N)"
} catch {
    Write-Host "[warn] No interactive prompt available - skipping settlement." -ForegroundColor Yellow
}

if ($answer -match '^[Yy]') {
    & $Python "backfill_results.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[warn] backfill_results.py exited with code $LASTEXITCODE (core pipeline still succeeded)." -ForegroundColor Yellow
    } else {
        Write-Host "[ok] settlement & audit complete" -ForegroundColor Green
    }
} else {
    Write-Host "[skip] run later with: venv\Scripts\python.exe backfill_results.py" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Pipeline complete: data -> features -> models -> inference -> settlement." -ForegroundColor Green
Write-Host "Dashboard: venv\Scripts\streamlit.exe run dashboard\app.py" -ForegroundColor Cyan
exit 0
