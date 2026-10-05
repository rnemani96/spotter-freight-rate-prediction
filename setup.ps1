# setup.ps1 — Spotter Freight Rate Prediction: One-Click Setup
# Run this from the project root: .\setup.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Spotter Freight Rate Prediction — Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Create virtual environment
if (-Not (Test-Path "$root\venv")) {
    Write-Host "[1/4] Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv "$root\venv"
    Write-Host "      Created: $root\venv" -ForegroundColor Green
} else {
    Write-Host "[1/4] Virtual environment already exists. Skipping." -ForegroundColor Gray
}

# 2. Upgrade pip
Write-Host "[2/4] Upgrading pip..." -ForegroundColor Yellow
& "$root\venv\Scripts\python.exe" -m pip install --upgrade pip --quiet

# 3. Install dependencies
Write-Host "[3/4] Installing requirements from requirements.txt..." -ForegroundColor Yellow
& "$root\venv\Scripts\pip.exe" install -r "$root\requirements.txt" --quiet
Write-Host "      All packages installed." -ForegroundColor Green

# 4. Verify data files exist
Write-Host "[4/4] Checking raw data files..." -ForegroundColor Yellow
$required = @(
    "data\raw\train-test.csv",
    "data\raw\validation.csv",
    "data\raw\december-chart-inputs.csv",
    "data\raw\validation-predictions-template.csv"
)
$missing = $false
foreach ($f in $required) {
    $p = Join-Path $root $f
    if (Test-Path $p) {
        Write-Host "      [OK] $f" -ForegroundColor Green
    } else {
        Write-Host "      [MISSING] $f" -ForegroundColor Red
        $missing = $true
    }
}
if ($missing) {
    Write-Host ""
    Write-Host "WARNING: Some data files are missing. Place them in data/raw/ and re-run." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "USAGE:" -ForegroundColor Yellow
Write-Host "  Activate venv  : .\venv\Scripts\activate"
Write-Host "  Fast predict   : python main.py --skip-cv --models lasso ridge catboost"
Write-Host "  Full CV run    : python main.py --skip-tuning"
Write-Host "  Full pipeline  : python main.py"
Write-Host ""
Write-Host "OUTPUTS (after running):"
Write-Host "  outputs\validation_predictions.csv   <- 12,000 row submission"
Write-Host "  outputs\december-chart-inputs.csv    <- 31-row December chart"
Write-Host "  outputs\scorer_results\candidate_december.png <- Rate chart"
Write-Host ""
