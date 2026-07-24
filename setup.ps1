$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Checkpoint = Join-Path $Root "worker\vendor\video-depth-anything\checkpoints\video_depth_anything_vits.pth"
$CheckpointUrl = "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth?download=true"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is not on PATH."
}
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "The Python launcher (py) is not on PATH. Install Python 3.11."
}
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "Node.js is not on PATH."
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -or
    -not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    throw "ffmpeg and ffprobe must be on PATH. Example: winget install Gyan.FFmpeg"
}

Write-Host "1/5 Initializing submodules..."
Push-Location $Root
try {
    git submodule update --init --recursive
} finally {
    Pop-Location
}

Write-Host "2/5 Python virtual environment..."
if (-not (Test-Path -LiteralPath $Python)) {
    py -3.11 -m venv (Join-Path $Root ".venv")
}
& $Python -m pip install --quiet -r (Join-Path $Root "worker\requirements.txt")

Write-Host "3/5 Depth model checkpoint..."
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Checkpoint) | Out-Null
    Invoke-WebRequest -Uri $CheckpointUrl -OutFile $Checkpoint
} else {
    Write-Host "  already present, skipping"
}

Write-Host "4/5 Web dependencies..."
Push-Location (Join-Path $Root "web")
try {
    npm ci
} finally {
    Pop-Location
}

Write-Host "5/5 Done"
Write-Host ""
Write-Host "Run .\start-depdy.ps1 next. (Person Matte / Line Art models auto-download on first use.)"
