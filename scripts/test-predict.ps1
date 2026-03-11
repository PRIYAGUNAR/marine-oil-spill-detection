param(
    [string]$ImagePath = "test_inputs/lena.jpg",
    [int]$Port = 8000,
    [double]$Threshold = 0.5
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ImagePath)) {
    throw "Image not found at $ImagePath"
}

$url = "http://127.0.0.1:$Port/predict?threshold=$Threshold&return_mask=true"
$json = curl.exe -s -X POST $url -F "file=@$ImagePath"

if (-not $json) {
    throw "No response from API"
}

$result = $json | ConvertFrom-Json

Write-Host "Prediction complete"
Write-Host "Oil ratio: $($result.oil_ratio)"
Write-Host "Avg probability: $($result.avg_probability)"
Write-Host "Max probability: $($result.max_probability)"

if ($result.mask_png_base64) {
    $bytes = [Convert]::FromBase64String($result.mask_png_base64)
    $outPath = "test_inputs/pred_mask.png"
    [System.IO.File]::WriteAllBytes($outPath, $bytes)
    Write-Host "Mask saved to: $outPath"
}
