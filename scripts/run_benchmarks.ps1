# Automatiza una corrida reproducible en Windows.
# Los parámetros permiten repetir el protocolo sin editar el script.
param(
    # Ejecutable CUDA previamente compilado.
    [string]$Executable = ".\hough.exe",
    # Número de kernels que sí se guardan en timings.csv.
    [int]$Runs = 10,
    # Corridas previas que estabilizan contexto, cachés y frecuencias de GPU.
    [int]$Warmup = 2,
    # Multiplicador de desviación estándar para seleccionar picos de Hough.
    [double]$Sigma = 2.4,
    # Límite superior de rectas dibujadas después de NMS.
    [int]$MaxLines = 12,
    # Vecindad radial usada por la supresión de máximos locales.
    [int]$NmsRadius = 4,
    # Vecindad angular usada por la supresión de máximos locales.
    [int]$NmsTheta = 3,
    # Solo votan píxeles cuya intensidad sea mayor que este valor.
    [int]$PixelThreshold = 0
)

# Convertir cualquier error de cmdlet en una excepción que detenga la corrida.
$ErrorActionPreference = "Stop"
# PSScriptRoot es scripts/; su padre es la raíz del proyecto.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

# Crear el destino y registrar el estado del controlador/GPU antes de medir.
New-Item -ItemType Directory -Force -Path "results" | Out-Null
nvidia-smi |
    ForEach-Object { $_.TrimEnd() } |
    Out-File -FilePath "results\gpu_info.txt" -Encoding utf8

# Guardar todos los parámetros que afectan la reproducibilidad de la corrida.
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
# Buscar nvcc primero en PATH; si no está, localizar la versión CUDA instalada.
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
# Anexar versión exacta del Toolkit al archivo de metadatos.
& $nvccPath --version |
    Out-File -FilePath "results\run_metadata.txt" -Append -Encoding utf8

# Ejecutar las tres variantes con una sola entrada y configuración común.
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

# Los programas nativos comunican su estado mediante LASTEXITCODE.
if ($LASTEXITCODE -ne 0) {
    throw "La ejecución CUDA terminó con código $LASTEXITCODE."
}

# Derivar media, mediana, desviación, mínimo y máximo desde el CSV crudo.
& (Join-Path $PSScriptRoot "summarize_timings.ps1") `
    -CsvPath "results\timings.csv" `
    -OutputPath "results\summary.md"

# Indicar explícitamente los tres artefactos que debe inspeccionar el usuario.
Write-Host "Corrida completada. Revise timings.csv, detected_lines.csv y runway_lines.png."
