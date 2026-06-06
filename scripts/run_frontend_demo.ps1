$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location (Join-Path $projectRoot "frontend")

$env:VITE_API_BASE_URL = "http://localhost:8000"
$env:VITE_DEMO_INDEX_NAME = "chromatest2"

if (-not (Test-Path "node_modules")) {
    npm install
}

npm run dev
