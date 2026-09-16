[CmdletBinding()]
param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$venvDirectory = Join-Path $PSScriptRoot ".venv"
$venvPython = Join-Path $venvDirectory "Scripts\python.exe"
$requirementsPath = Join-Path $PSScriptRoot "requirements-gemma4.txt"

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    if (Test-Path -LiteralPath $venvDirectory) {
        throw "The existing .venv has no Scripts\python.exe. Inspect it before recreating the environment."
    }
    & $PythonExe -m venv $venvDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "Creating .venv failed (exit $LASTEXITCODE). Use -PythonExe with the path to Python 3.12."
    }
}

& $venvPython -m pip install --no-cache-dir --disable-pip-version-check "torch==2.11.0+cu128" --index-url "https://download.pytorch.org/whl/cu128"
if ($LASTEXITCODE -ne 0) {
    throw "Installing PyTorch CUDA 12.8 failed (exit $LASTEXITCODE)."
}

& $venvPython -m pip install --no-cache-dir --disable-pip-version-check -r $requirementsPath
if ($LASTEXITCODE -ne 0) {
    throw "Installing Gemma 4 dependencies failed (exit $LASTEXITCODE)."
}

& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "Dependency verification failed (exit $LASTEXITCODE)."
}

& $venvPython -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected')"
if ($LASTEXITCODE -ne 0) {
    throw "Importing PyTorch failed (exit $LASTEXITCODE)."
}

Write-Host "Environment ready: $venvPython"
Write-Host "See GEMMA4.md for data preparation, a short training check, and experiment commands."
