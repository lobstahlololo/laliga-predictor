<#
.SYNOPSIS
    La Liga predictor - end-to-end pipeline orchestrator.

.DESCRIPTION
    Sequences the full pipeline natively in PowerShell:
      0. Verify environment state (venv Python, core deps, project files)
      1. src/data/make_dataset.py        - fetch clean data (Jina fallback intact)
      2. src/features/build_features.py  - regenerate feature shapes + scaler
      3. src/models/train_model.py       - refit tuned models (neg_log_loss)
      4. predict_upcoming.py             - launch inference for user testing
      5. backfill_results.py (optional)  - prompted: settle the ledger and
                                           calculate real-world ROI metrics

    Stops immediately if any stage fails (non-zero exit code), so a broken
    stage can never cascade into the next one.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File run_pipeline.ps1
#>

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
    @{ Name = "make_dataset";   Script = "src\data\make_dataset.py" },
    @{ Name = "build_features"; Script = "src\features\build_features.py" },
    @{ Name = "train_model";    Script = "src\models\train_model.py" },
    @{ Name = "predict";        Script = "predict_upcoming.py" }
)

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host ">>> $Message" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# Stage 0: environment verification
# ---------------------------------------------------------------------------
Write-Step "Stage 0/4: verifying environment state"

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

foreach ($stage in $Stages) {
    $path = Join-Path $ProjectRoot $stage.Script
    if (-not (Test-Path $path)) {
        Write-Host "[FAIL] missing script: $($stage.Script)" -ForegroundColor Red
        exit 1
    }
}
Write-Host "[ok] all pipeline scripts present"

if (-not (Test-Path (Join-Path $ProjectRoot "backfill_results.py"))) {
    Write-Host "[FAIL] missing script: backfill_results.py" -ForegroundColor Red
    exit 1
}
Write-Host "[ok] settlement script present"
Write-Host "[ok] environment verified" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Stages 1-4: sequential execution with hard stop on failure
# ---------------------------------------------------------------------------
$total = $Stages.Count
for ($i = 0; $i -lt $total; $i++) {
    $stage = $Stages[$i]
    Write-Step "Stage $($i + 1)/$($total): $($stage.Name) ($($stage.Script))"

    & $Python (Join-Path $ProjectRoot $stage.Script)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] $($stage.Name) exited with code $LASTEXITCODE - pipeline halted." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "[ok] $($stage.Name)" -ForegroundColor Green
}

Write-Host ""
Write-Host "Pipeline core complete: data -> features -> models -> inference." -ForegroundColor Green

# ---------------------------------------------------------------------------
# Stage 5 (optional, prompted): settle ledger + financial ROI audit
# ---------------------------------------------------------------------------
Write-Step "Stage 5/5 (optional): settlement & ROI audit"

$answer = ""
try {
    $answer = Read-Host "Run backfill_results.py to update your active ledger histories and calculate real-world ROI metrics? (y/N)"
} catch {
    Write-Host "[warn] No interactive prompt available - skipping settlement." -ForegroundColor Yellow
}

if ($answer -match '^[Yy]') {
    & $Python (Join-Path $ProjectRoot "backfill_results.py")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[warn] backfill_results.py exited with code $LASTEXITCODE (core pipeline still succeeded)." -ForegroundColor Yellow
    } else {
        Write-Host "[ok] settlement & ROI audit complete" -ForegroundColor Green
    }
} else {
    Write-Host "[skip] settlement skipped - run later with: venv\Scripts\python.exe backfill_results.py" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Pipeline complete: data -> features -> models -> inference -> settlement." -ForegroundColor Green
exit 0
