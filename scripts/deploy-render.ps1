param(
    [string]$ServiceName = "marine-oil-spill-api"
)

$ErrorActionPreference = "Stop"

Write-Host "Preparing Render deployment..."

if (-not (Test-Path "render.yaml")) {
    throw "render.yaml not found in repository root"
}

$branch = (git branch --show-current).Trim()
Write-Host "Current branch: $branch"
Write-Host "Pushing latest code to origin/$branch ..."
$filesToStage = @(
    "README.md",
    "requirements.txt",
    "api.py",
    "Dockerfile",
    ".dockerignore",
    "render.yaml",
    "scripts/deploy-local.ps1",
    "scripts/test-predict.ps1",
    "scripts/deploy-render.ps1"
)

$existingFiles = $filesToStage | Where-Object { Test-Path $_ }
if ($existingFiles.Count -eq 0) {
    throw "No deployment files found to stage."
}

git add -- $existingFiles
try {
    git commit -m "chore: prepare cloud deployment" | Out-Null
} catch {
    Write-Host "No new commit created (possibly no staged changes)."
}
git push origin $branch

Write-Host "Next step in browser:"
Write-Host "1) Open https://dashboard.render.com/blueprint/new"
Write-Host "2) Connect this GitHub repo"
Write-Host "3) Select service: $ServiceName"
Write-Host "4) Click Apply"
