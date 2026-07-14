param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$racineProjet = Resolve-Path (Join-Path $PSScriptRoot "..")
$dossierVenv = Join-Path $racineProjet ".venv"
$pythonVenv = Join-Path $dossierVenv "Scripts\python.exe"
$requirements = Join-Path $racineProjet "requirements.txt"

Write-Host ""
Write-Host "FedCompress - preparation de l'environnement Python"
Write-Host "Racine projet : $racineProjet"
Write-Host "Venv          : $dossierVenv"
Write-Host ""

if (-not (Test-Path $pythonVenv)) {
    Write-Host "Creation du venv local..."
    & $Python -m venv $dossierVenv
}
else {
    Write-Host "Venv deja present, mise a jour des dependances..."
}

Write-Host "Mise a jour de pip..."
& $pythonVenv -m pip install --upgrade pip

Write-Host "Installation des dependances du projet..."
& $pythonVenv -m pip install -r $requirements

Write-Host ""
Write-Host "Verification rapide..."
& $pythonVenv -c "import torch, torchvision, transformers; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('transformers:', transformers.__version__); print('cuda disponible:', torch.cuda.is_available())"

Write-Host ""
Write-Host "Environnement pret."
Write-Host "Commande type : .\.venv\Scripts\python.exe scripts\inspect_dataset.py --save-grid"
