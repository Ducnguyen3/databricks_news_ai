param(
    [ValidateSet("full", "incremental")]
    [string]$Mode = "incremental",
    [int]$Limit = 0,
    [switch]$RunDatabricksJobs,
    [switch]$SkipTests,
    [switch]$RunChunkAudit,
    [switch]$RunSmokeTest
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Command
    )
    Write-Host ""
    Write-Host "==== $Name ===="
    & $Command[0] $Command[1..($Command.Length - 1)]
}

if ($RunDatabricksJobs) {
    Invoke-Step "Crawl raw documents" @("python", "-m", "app.jobs.crawl_news_job")
    Invoke-Step "Parse, canonicalize, deduplicate" @("python", "-m", "app.jobs.parse_and_canonicalize_job")
    Invoke-Step "Build Gold articles_clean" @("python", "-m", "app.jobs.build_articles_clean_job")
}
else {
    Write-Host "Skipping Databricks jobs. Run them on Databricks or pass -RunDatabricksJobs if this machine is configured."
}

$indexArgs = @("python", "-m", "app.local_ai.index_sync", "--rebuild_mode", $Mode)
if ($Limit -gt 0) {
    $indexArgs += @("--limit", [string]$Limit)
}
Invoke-Step "Sync Gold to local Chroma" $indexArgs

if ($RunChunkAudit) {
    Invoke-Step "Run chunk quality audit" @("python", "-m", "app.local_ai.chunk_quality_audit")
}

if (-not $SkipTests) {
    Invoke-Step "Run unit tests" @("python", "-m", "unittest", "discover", "-s", "tests")
}

if ($RunSmokeTest) {
    Invoke-Step "Run RAG structured smoke test" @("python", "-m", "app.local_ai.rag_smoke_test")
}
