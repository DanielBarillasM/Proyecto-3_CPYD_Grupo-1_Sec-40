# Convierte el CSV de observaciones individuales en una tabla Markdown.
param(
    # CSV esperado: columnas mode, run y kernel_ms.
    [string]$CsvPath = "results\timings.csv",
    # Documento que recibirá la tabla estadística.
    [string]$OutputPath = "results\summary.md"
)

# Fallar inmediatamente si una operación de lectura/escritura no funciona.
$ErrorActionPreference = "Stop"
# Import-Csv transforma cada fila en un objeto con propiedades por columna.
$rows = Import-Csv -LiteralPath $CsvPath
if (-not $rows) {
    throw "El CSV no contiene mediciones."
}

# Encabezado y separador de la tabla Markdown que se construirá por variante.
$lines = @(
    "# Resumen de tiempos",
    "",
    "| Variante | N | Media (ms) | Mediana (ms) | Desv. estandar | Minimo | Maximo |",
    "|---|---:|---:|---:|---:|---:|---:|"
)

# Group-Object separa global, constant y shared sin asumir un orden fijo.
foreach ($group in ($rows | Group-Object mode)) {
    # Parsear con cultura invariante para que el punto siempre sea decimal.
    $values = @($group.Group | ForEach-Object {
        [double]::Parse($_.kernel_ms, [System.Globalization.CultureInfo]::InvariantCulture)
    } | Sort-Object)

    # La lista ordenada permite obtener extremos y mediana directamente.
    $count = $values.Count
    $mean = ($values | Measure-Object -Average).Average
    # Para N par se promedian los dos valores centrales; para N impar se usa uno.
    if ($count % 2 -eq 0) {
        $median = ($values[$count / 2 - 1] + $values[$count / 2]) / 2.0
    } else {
        $median = $values[[math]::Floor($count / 2)]
    }
    # Se usa desviación poblacional: el divisor es N, igual que en hough.cu.
    $variance = (($values | ForEach-Object { ($_ - $mean) * ($_ - $mean) }) |
        Measure-Object -Sum).Sum / $count
    $standardDeviation = [math]::Sqrt($variance)
    $minimum = $values[0]
    $maximum = $values[-1]

    # F6 mantiene seis decimales y hace comparables ejecuciones sucesivas.
    $lines += "| {0} | {1} | {2:F6} | {3:F6} | {4:F6} | {5:F6} | {6:F6} |" -f `
        $group.Name, $count, $mean, $median, $standardDeviation, $minimum, $maximum
}

# Sobrescribir el resumen para que nunca mezcle corridas distintas.
$lines | Set-Content -LiteralPath $OutputPath -Encoding utf8
Write-Host "Resumen estadistico: $OutputPath"
