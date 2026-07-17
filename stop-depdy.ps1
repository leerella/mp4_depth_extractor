$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runtime = Join-Path $Root "worker\runtime"
$PidFiles = @(
    (Join-Path $Runtime "web.pid"),
    (Join-Path $Runtime "worker.pid")
)

foreach ($PidFile in $PidFiles) {
    if (-not (Test-Path -LiteralPath $PidFile)) {
        continue
    }

    $ProcessId = [int](Get-Content -LiteralPath $PidFile -Raw)
    $Process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($Process) {
        Stop-Process -Id $ProcessId
        $Process.WaitForExit()
    }
    Remove-Item -LiteralPath $PidFile -Force
}

Write-Host "DEPDY processes stopped."
