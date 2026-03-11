param(
    [string]$Host = "0.0.0.0",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

Write-Host "[1/4] Installing dependencies..."
python -m pip install -r requirements.txt

Write-Host "[2/4] Stopping existing uvicorn process on port $Port if present..."
$existing = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty OwningProcess
if ($existing) {
    Stop-Process -Id $existing -Force
    Start-Sleep -Seconds 1
}

Write-Host "[3/4] Starting API server..."
Start-Process -FilePath "python" -ArgumentList "-m uvicorn api:app --host $Host --port $Port" -WorkingDirectory (Get-Location)

Write-Host "[4/4] Waiting for health endpoint..."
$healthUrl = "http://127.0.0.1:$Port/health"
$ok = $false
for ($i = 0; $i -lt 20; $i++) {
    try {
        $resp = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        if ($resp.status -eq "ok") {
            $ok = $true
            break
        }
    } catch {
        Start-Sleep -Milliseconds 500
    }
}

if (-not $ok) {
    throw "API did not become healthy. Check terminal processes."
}

Write-Host "Deployment successful."
Write-Host "Health URL: $healthUrl"
Write-Host "Docs URL:   http://127.0.0.1:$Port/docs"
