param(
    [string]$CsvPath = "results\timings.csv",
    [string]$OutputPath = "results\summary.md"
)

$ErrorActionPreference = "Stop"
$rows = Import-Csv -LiteralPath $CsvPath
if (-not $rows) {
    throw "El CSV no contiene mediciones."
}

$lines = @(
    "# Resumen de tiempos",
    "",
    "| Variante | N | Media (ms) | Mediana (ms) | Desv. estandar | Minimo | Maximo |",
    "|---|---:|---:|---:|---:|---:|---:|"
)

foreach ($group in ($rows | Group-Object mode)) {
    $values = @($group.Group | ForEach-Object {
        [double]::Parse($_.kernel_ms, [System.Globalization.CultureInfo]::InvariantCulture)
    } | Sort-Object)

    $count = $values.Count
    $mean = ($values | Measure-Object -Average).Average
    if ($count % 2 -eq 0) {
        $median = ($values[$count / 2 - 1] + $values[$count / 2]) / 2.0
    } else {
        $median = $values[[math]::Floor($count / 2)]
    }
    $variance = (($values | ForEach-Object { ($_ - $mean) * ($_ - $mean) }) |
        Measure-Object -Sum).Sum / $count
    $standardDeviation = [math]::Sqrt($variance)
    $minimum = $values[0]
    $maximum = $values[-1]

    $lines += "| {0} | {1} | {2:F6} | {3:F6} | {4:F6} | {5:F6} | {6:F6} |" -f `
        $group.Name, $count, $mean, $median, $standardDeviation, $minimum, $maximum
}

$lines | Set-Content -LiteralPath $OutputPath -Encoding utf8
Write-Host "Resumen estadistico: $OutputPath"
