param(
    [string]$Executable = ".\hough.exe",
    [int]$Runs = 10,
    [int]$Warmup = 2,
    [double]$Sigma = 2.4,
    [int]$MaxLines = 12,
    [int]$NmsRadius = 4,
    [int]$NmsTheta = 3,
    [int]$PixelThreshold = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

New-Item -ItemType Directory -Force -Path "results" | Out-Null
nvidia-smi |
    ForEach-Object { $_.TrimEnd() } |
    Out-File -FilePath "results\gpu_info.txt" -Encoding utf8

$metadata = @(
    "timestamp=$(Get-Date -Format o)"
    "computer=$env:COMPUTERNAME"
    "runs=$Runs"
    "warmup=$Warmup"
    "sigma=$Sigma"
    "max_lines=$MaxLines"
    "nms_radius=$NmsRadius"
    "nms_theta=$NmsTheta"
    "pixel_threshold=$PixelThreshold"
)
$metadata | Out-File -FilePath "results\run_metadata.txt" -Encoding utf8
$nvccCommand = Get-Command nvcc.exe -ErrorAction SilentlyContinue
if ($null -ne $nvccCommand) {
    $nvccPath = $nvccCommand.Source
} else {
    $nvccPath = Get-ChildItem `
        -LiteralPath "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" `
        -Filter "nvcc.exe" -Recurse -ErrorAction Stop |
        Sort-Object FullName -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}
& $nvccPath --version |
    Out-File -FilePath "results\run_metadata.txt" -Append -Encoding utf8

& $Executable `
    --input "data\runway.pgm" `
    --mode all `
    --runs $Runs `
    --warmup $Warmup `
    --sigma $Sigma `
    --max-lines $MaxLines `
    --nms-radius $NmsRadius `
    --nms-theta $NmsTheta `
    --pixel-threshold $PixelThreshold `
    --output "results\runway_lines.png" `
    --csv "results\timings.csv" `
    --lines-csv "results\detected_lines.csv"

if ($LASTEXITCODE -ne 0) {
    throw "La ejecución CUDA terminó con código $LASTEXITCODE."
}

& (Join-Path $PSScriptRoot "summarize_timings.ps1") `
    -CsvPath "results\timings.csv" `
    -OutputPath "results\summary.md"

Write-Host "Corrida completada. Revise timings.csv, detected_lines.csv y runway_lines.png."
