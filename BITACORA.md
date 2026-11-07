# Bitácora de pruebas

## Identificación

- Curso: Computación Paralela y Distribuida
- Proyecto: Transformada Lineal de Hough con CUDA
- Grupo: 1
- Sección: 40
- Integrantes: Pablo Daniel Barillas Moreno (carné 22193) y Milton Giovanni Polanco Serrano (carné 23471)

## Preparación - 25 de octubre de 2026

- Se revisó el enunciado completo y la rúbrica.
- Se inspeccionó `data/runway.pgm`: PGM P5, 800 x 600 y 480,000 píxeles.
- Se detectaron 138,406 píxeles con intensidad mayor que cero.
- Se calculó una referencia independiente de 12,456,540 votos.
- Equipo: NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB y capacidad de cómputo 8.9.
- Se instalaron Visual Studio Build Tools 2022 17.14.42 con MSVC 19.44 y CUDA Toolkit 13.4.
- Se implementaron las variantes Global, Constante y Compartida.
- Se agregó validación exacta CPU-GPU, medición con CUDA Events, salida PNG, CSV de tiempos y CSV de líneas.
- La compilación Release con `nvcc 13.4` terminó correctamente.
- Las tres variantes coincidieron exactamente con CPU en los 9,000 bins.
- Una imagen P2 de 7 x 5 validó un último bloque incompleto.
- `compute-sanitizer --tool memcheck` terminó con 0 errores.
- `compute-sanitizer --tool racecheck` terminó con 0 peligros, 0 errores y 0 advertencias.

## Protocolo oficial

1. Conectar el equipo a corriente y seleccionar el perfil de máximo rendimiento.
2. Cerrar aplicaciones que utilicen intensivamente la GPU.
3. Registrar `nvidia-smi` en `results/gpu_info.txt`.
4. Ejecutar dos calentamientos no medidos por kernel.
5. Ejecutar diez mediciones por kernel con la misma imagen y geometría.
6. Confirmar que las tres variantes impriman `Validacion CPU-GPU: OK`.
7. Ejecutar Compute Sanitizer y conservar sus registros.
8. Conservar los CSV, metadatos y PNG oficiales.

Comando PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_benchmarks.ps1 -Executable ".\hough.exe" -Runs 10 -Warmup 2
```

## Mediciones oficiales

| Corrida | Global (ms) | Global + Constante (ms) | Global + Constante + Compartida (ms) |
|---:|---:|---:|---:|
| 1 | 1.078272 | 1.073152 | 0.798720 |
| 2 | 1.084416 | 1.074176 | 0.798720 |
| 3 | 1.084416 | 1.078272 | 0.797696 |
| 4 | 1.082368 | 1.081344 | 0.797696 |
| 5 | 1.084320 | 1.071104 | 0.797696 |
| 6 | 1.087488 | 1.077248 | 0.798720 |
| 7 | 1.077248 | 1.069056 | 0.798720 |
| 8 | 1.077248 | 1.073152 | 0.798720 |
| 9 | 1.084416 | 1.076224 | 0.798720 |
| 10 | 1.085440 | 1.075232 | 0.797696 |

## Resumen estadístico

| Variante | Media | Mediana | Desviación estándar | Mínimo | Máximo |
|---|---:|---:|---:|---:|---:|
| Global | 1.082563 | 1.084368 | 0.003477 | 1.077248 | 1.087488 |
| Constante | 1.074896 | 1.074704 | 0.003398 | 1.069056 | 1.081344 |
| Compartida | 0.798310 | 0.798720 | 0.000502 | 0.797696 | 0.798720 |

Constante obtuvo un speedup de `1.0071x` y redujo el tiempo medio `0.71%`. Compartida obtuvo `1.3561x` y redujo el tiempo medio `26.26%` frente a Global.

La salida visual usa `sigma=2.4`, NMS de 4 bins en `r` y 3 bins en `theta`. Se seleccionaron seis líneas y se registraron en `results/detected_lines.csv`.

## Regresión ampliada - 4 de noviembre de 2026

Se amplió la cobertura sin modificar la lógica de producción:

- PGM P2 y P5, escritura PNG y entradas inválidas;
- imagen negra, píxel único, línea horizontal y línea vertical;
- líneas cruzadas y frontera del umbral de intensidad;
- último bloque CUDA incompleto;
- ayuda CLI y argumentos inválidos;
- ejecución de Global, Constante y Compartida con validación exacta contra CPU.

La compilación final con MSVC 19.44 y CUDA 13.4.59 terminó correctamente. CTest reportó **7 de 7 pruebas aprobadas**. La suite se repitió tres veces con el mismo resultado.

## Cierre documental - 6 de noviembre de 2026

- Se preparó el informe final en LaTeX y PDF.
- Se preparó una presentación ejecutiva HTML que funciona sin conexión.
- Se agregó una guía de exposición y una revisión final contra la rúbrica.
- Se conservaron intactos los resultados oficiales.
- Se agregaron los nombres completos y carnés en la portada.

## Cierre de entrega - 7 de noviembre de 2026

- Se confirmó que `HEAD` y `origin/main` coincidían en el commit documental `cb28172` antes de preparar los paquetes.
- Se mantuvo `a5955e4` como el último commit con cambios técnicos y de pruebas; los commits posteriores solo modificaron documentación y recursos de presentación.
- Se generaron por separado `01_Codigo_CUDA_Grupo_1.zip` y `02_Mediciones_y_Resultados_Grupo_1.zip` mediante listas explícitas de archivos rastreados.
- Se excluyeron `.git/`, `build/`, objetos, ejecutables, ZIP anteriores y resultados locales ignorados.
- Se preparó `03_Informe_Final_Proyecto_3_CUDA_Grupo_1.pdf`, una copia verificada del informe de 18 páginas.
- Se agregó un guion extendido de defensa en LaTeX y PDF, junto con auditoría, manifiesto de trazabilidad y hashes SHA-256.
- Se leyó por completo el PDF de instrucciones y se comprobó la correspondencia entre cada requisito y su evidencia.
- Se ejecutó nuevamente la suite compilada: CTest aprobó **7 de 7 pruebas**.
- Se realizó una revisión visual de las 18 páginas del informe y de las 11 diapositivas de la presentación.

Los paquetes generados el 7 de noviembre representan el commit `cb28172`. Esta actualización documental permanece local y debe revisarse antes de regenerar los ZIP definitivos.