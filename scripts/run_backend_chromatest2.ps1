$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $projectRoot

$venvActivate = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
}

python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('fastapi') and importlib.util.find_spec('uvicorn') else 1)" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing backend demo dependencies from requirements-local.txt"
    python -m pip install -r requirements-local.txt
}

$env:CHROMA_PERSIST_DIR = "data/chromatest2"
$env:EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
$env:CHROMA_COLLECTION_NAME = "news_articles"

Write-Host "Using Chroma path: $env:CHROMA_PERSIST_DIR"
Write-Host "Using embedding model: $env:EMBEDDING_MODEL_NAME"

python -c "import sys; import traceback;`ntry:`n    from sentence_transformers import SentenceTransformer`n    print('Embedding import OK')`nexcept Exception as exc:`n    print(f'Embedding import failed: {exc}')`n    sys.exit(1)"
if ($LASTEXITCODE -ne 0) {
    throw "Embedding model import failed. If Windows Application Control blocks .venv DLL/PYD files, run Get-ChildItem .\.venv -Recurse -File | Unblock-File or run backend with a trusted/global Python environment."
}

python scripts/test_chroma_client.py --chroma-dir data/chromatest2
if ($LASTEXITCODE -ne 0) {
    throw "Chroma healthcheck failed for data/chromatest2"
}

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
