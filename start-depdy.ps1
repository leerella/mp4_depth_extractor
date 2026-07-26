param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$WebRoot = Join-Path $Root "web"
$Runtime = Join-Path $Root "worker\runtime"
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Checkpoint = Join-Path $Root "worker\vendor\video-depth-anything\checkpoints\video_depth_anything_vits.pth"
$WorkerPidFile = Join-Path $Runtime "worker.pid"
$WebPidFile = Join-Path $Runtime "web.pid"

function Get-ListeningProcessId([int]$Port) {
    $Listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($Listener) {
        return $Listener.OwningProcess
    }
    return $null
}

function Wait-ForUrl([string]$Url, [int]$TimeoutSeconds = 60) {
    $Deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            $Response = Invoke-WebRequest -UseBasicParsing $Url -TimeoutSec 2
            if ($Response.StatusCode -ge 200 -and $Response.StatusCode -lt 400) {
                return
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    } until ((Get-Date) -ge $Deadline)
    throw "Service did not respond before the timeout: $Url"
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Depth model checkpoint not found: $Checkpoint"
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -or
    -not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    throw "ffmpeg and ffprobe must be available on PATH."
}

New-Item -ItemType Directory -Force -Path $Runtime | Out-Null

$WorkerProcessId = Get-ListeningProcessId 8000
if (-not $WorkerProcessId) {
    $Worker = Start-Process `
        -FilePath $Python `
        -ArgumentList @("-m", "uvicorn", "worker.app:app", "--host", "127.0.0.1", "--port", "8000") `
        -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $Runtime "worker.stdout.log") `
        -RedirectStandardError (Join-Path $Runtime "worker.stderr.log") `
        -PassThru
    Set-Content -LiteralPath $WorkerPidFile -Value $Worker.Id
}
Wait-ForUrl "http://127.0.0.1:8000/health"

$WebProcessId = Get-ListeningProcessId 3002
if (-not $WebProcessId) {
    Push-Location $WebRoot
    try {
        & (Get-Command npm.cmd).Source run build
        if ($LASTEXITCODE -ne 0) {
            throw "Next.js production build failed."
        }
    } finally {
        Pop-Location
    }

    $Node = (Get-Command node.exe).Source
    $Next = Join-Path $WebRoot "node_modules\next\dist\bin\next"
    $Web = Start-Process `
        -FilePath $Node `
        -ArgumentList @($Next, "start", "-H", "127.0.0.1", "-p", "3002") `
        -WorkingDirectory $WebRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $Runtime "web.stdout.log") `
        -RedirectStandardError (Join-Path $Runtime "web.stderr.log") `
        -PassThru
    Set-Content -LiteralPath $WebPidFile -Value $Web.Id
}
Wait-ForUrl "http://127.0.0.1:3002/"

Write-Host "DEPDY is running: http://localhost:3002"
if (-not $NoBrowser) {
    Start-Process "http://localhost:3002"
}
