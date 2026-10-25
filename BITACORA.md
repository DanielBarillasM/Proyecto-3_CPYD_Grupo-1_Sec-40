# Bitácora de pruebas

## Identificación

- Curso: Computación Paralela y Distribuida
- Proyecto: Transformada Lineal de Hough con CUDA
- Grupo: 1
- Sección: 40
- Integrantes: Pablo Daniel Barillas Moreno y Milton Polanco

## Preparación - 25 de octubre de 2026

- Se revisó el enunciado completo y la rúbrica.
- Se inspeccionó la imagen `data/runway.pgm`: PGM P5, 800 x 600, 480,000 píxeles.
- Se detectaron 138,406 píxeles con intensidad mayor que cero.
- Se calculó una referencia independiente de 12,456,540 votos totales.
- Equipo detectado: NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB, capacidad de cómputo 8.9.
- Inicialmente solo estaba disponible el controlador NVIDIA. Se instalaron Visual Studio Build Tools 2022 17.14.42 con MSVC 19.44 y CUDA Toolkit 13.4.
- Se implementaron las variantes Global, Constante y Compartida.
- Se agregó validación exacta CPU-GPU, CUDA Events, salida CSV y salida PNG.
- Se compiló `tests/test_image_io.cpp` con un compilador C++17 temporal: prueba aprobada para PGM `800 x 600` y PNG `800 x 600`.
- El PNG de prueba fue abierto e inspeccionado visualmente sin corrupción.
- Se analizó la sintaxis C++ del archivo CUDA después de normalizar únicamente los calificadores/extensiones CUDA en memoria: cero nodos de error.
- Se probó `scripts/summarize_timings.ps1` con el fixture de seis mediciones; medias, medianas y desviaciones coincidieron con los valores esperados.
- La compilación Release con `nvcc 13.4` terminó correctamente.
- Las tres variantes coincidieron exactamente con CPU para los 9,000 bins y 12,456,540 votos.
- `compute-sanitizer --tool memcheck` terminó con 0 errores.
- `compute-sanitizer --tool racecheck` terminó con 0 peligros, 0 errores y 0 advertencias.

## Protocolo oficial

1. Conectar el equipo a corriente y seleccionar el perfil de máximo rendimiento.
2. Cerrar aplicaciones que utilicen intensivamente la GPU.
3. Registrar `nvidia-smi` en `results/gpu_info.txt`.
4. Ejecutar dos calentamientos no medidos por kernel.
5. Ejecutar diez mediciones por kernel con la misma imagen y geometría.
6. Confirmar que las tres variantes impriman `Validacion CPU-GPU: OK`.
7. Ejecutar `compute-sanitizer` y registrar el resultado.
8. Conservar `results/timings.csv` y `results/runway_lines.png`.

Comando PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_benchmarks.ps1 -Executable ".\hough.exe" -Runs 10 -Warmup 2
```

## Mediciones

La tabla se completa con `results/timings.csv`. No deben inventarse datos antes de ejecutar el programa.

| Corrida | Global (ms) | Global + Constante (ms) | Global + Constante + Compartida (ms) |
|---:|---:|---:|---:|
| 1 | 1.091584 | 1.079296 | 0.799744 |
| 2 | 1.088512 | 1.078272 | 0.800768 |
| 3 | 1.078272 | 1.081344 | 0.799744 |
| 4 | 1.093632 | 1.083392 | 0.800768 |
| 5 | 1.091584 | 1.080320 | 0.799744 |
| 6 | 1.086464 | 1.077248 | 0.800768 |
| 7 | 1.079296 | 1.072128 | 0.800768 |
| 8 | 1.090560 | 1.072128 | 0.799744 |
| 9 | 1.088512 | 1.076224 | 0.799744 |
| 10 | 1.087488 | 1.082368 | 0.799744 |

## Resumen estadístico

| Variante | Media | Mediana | Desviación estándar | Mínimo | Máximo |
|---|---:|---:|---:|---:|---:|
| Global | 1.087590 | 1.088512 | 0.004856 | 1.078272 | 1.093632 |
| Constante | 1.078272 | 1.078784 | 0.003720 | 1.072128 | 1.083392 |
| Compartida | 0.800154 | 0.799744 | 0.000502 | 0.799744 | 0.800768 |

La memoria Constante obtuvo un speedup de `1.0086x` y redujo el tiempo medio aproximadamente `0.86%`. La variante con memoria Compartida obtuvo un speedup de `1.3592x` y redujo el tiempo medio aproximadamente `26.43%` frente a Global.

## Lista de comprobación

- [x] CUDA Toolkit y compilador host instalados.
- [x] Compilación sin errores.
- [x] Tres validaciones CPU-GPU correctas.
- [x] Treinta mediciones registradas.
- [x] PNG abierto e inspeccionado visualmente.
- [x] `compute-sanitizer` sin errores ni carreras.
- [x] Resultados incorporados al informe.
- [x] Integrantes, fecha y datos académicos completados; agregar carnés si la guía UVG los exige.
