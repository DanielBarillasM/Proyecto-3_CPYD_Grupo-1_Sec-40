param(
    [string]$Executable = ".\hough.exe",
    [int]$Runs = 10,
    [int]$Warmup = 2
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

New-Item -ItemType Directory -Force -Path "results" | Out-Null
nvidia-smi | Out-File -FilePath "results\gpu_info.txt" -Encoding utf8

& $Executable `
    --input "data\runway.pgm" `
    --mode all `
    --runs $Runs `
    --warmup $Warmup `
    --sigma 2 `
    --max-lines 20 `
    --output "results\runway_lines.png" `
    --csv "results\timings.csv"

if ($LASTEXITCODE -ne 0) {
    throw "La ejecución CUDA terminó con código $LASTEXITCODE."
}

& (Join-Path $PSScriptRoot "summarize_timings.ps1") `
    -CsvPath "results\timings.csv" `
    -OutputPath "results\summary.md"

Write-Host "Corrida completada. Revise results\timings.csv y results\runway_lines.png."
