# Proyecto 3 - Transformada Lineal de Hough con CUDA

Implementación de la Transformada Lineal de Hough para comparar tres jerarquías de memoria de CUDA:

1. acumulador y tablas trigonométricas en memoria global;
2. acumulador global y tablas trigonométricas en memoria constante;
3. acumulador local por bloque en memoria compartida, tablas en memoria constante y fusión final en memoria global.

El programa valida cada resultado GPU contra una referencia CPU, mide únicamente la llamada al kernel con CUDA Events, guarda todas las corridas en CSV y genera un PNG con las líneas detectadas.

## Requisitos

- GPU NVIDIA compatible con CUDA.
- CUDA Toolkit con `nvcc` y `compute-sanitizer`.
- Compilador host soportado por la versión instalada del Toolkit.
- GNU Make o CMake 3.24+.

El entorno verificado usa una NVIDIA GeForce RTX 4050 Laptop, capacidad de cómputo 8.9, CUDA Toolkit 13.4 y MSVC 19.44. La corrida oficial produjo validación exacta CPU-GPU y 30 mediciones.

## Compilación

Con Make:

```bash
make
```

Compilación directa equivalente:

```bash
nvcc -O3 -std=c++17 -lineinfo --fmad=false -I. src/hough.cu common/image_io.cpp -o hough
```

Con CMake:

```bash
cmake -S . -B build
cmake --build build --config Release
```

`--fmad=false` mantiene comparables las operaciones de punto flotante de CPU y GPU al discretizar `r`. Se aplica por igual a los tres kernels.

## Ejecución completa

Linux o un entorno CUDA compatible:

```bash
./hough \
  --input data/runway.pgm \
  --mode all \
  --warmup 2 \
  --runs 10 \
  --sigma 2.4 \
  --max-lines 12 \
  --nms-radius 4 \
  --nms-theta 3 \
  --pixel-threshold 0 \
  --output results/runway_lines.png \
  --csv results/timings.csv \
  --lines-csv results/detected_lines.csv
```

En PowerShell:

```powershell
.\hough.exe `
  --input data\runway.pgm `
  --mode all `
  --warmup 2 `
  --runs 10 `
  --sigma 2.4 `
  --max-lines 12 `
  --nms-radius 4 `
  --nms-theta 3 `
  --pixel-threshold 0 `
  --output results\runway_lines.png `
  --csv results\timings.csv `
  --lines-csv results\detected_lines.csv
```

También se puede ejecutar una sola variante con `--mode global`, `--mode constant` o `--mode shared`.

La corrida automatizada y el resumen estadístico se ejecutan así cuando la política de PowerShell bloquea scripts locales:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_benchmarks.ps1
```

## Qué mide el programa

- Las transferencias host-device, el `cudaMemset`, la referencia CPU y la escritura del PNG están fuera del intervalo medido.
- Cada medición comienza con `cudaEventRecord(start)`, lanza un kernel y termina con `cudaEventRecord(stop)` y `cudaEventSynchronize(stop)`.
- Se ejecutan dos calentamientos por defecto y diez mediciones por variante.
- Se imprimen media, mediana, desviación estándar, mínimo y máximo.
- `results/timings.csv` conserva cada observación individual.
- `results/detected_lines.csv` registra `r`, `theta` y votos de cada línea dibujada.
- `results/run_metadata.txt` identifica la computadora, parámetros y versión de CUDA Toolkit.

## Criterios de corrección

Antes de aceptar tiempos, el programa exige que los 9,000 bins del acumulador GPU coincidan exactamente con CPU. Un solo bin distinto hace que la ejecución termine con error.

Para la imagen incluida se esperan:

- resolución de `800 x 600`;
- 90 bins angulares y 100 bins radiales;
- 12,456,540 votos totales con la condición `pixel > 0`.

La visualización usa el umbral `media + sigma * desviación estándar`, supresión de máximos locales configurable y un máximo configurable de líneas. La vecindad de `theta` es circular y respeta la equivalencia `(r, theta) = (-r, theta + 180 grados)`. La corrida oficial usa `sigma=2.4`, radio 4 en `r`, radio 3 en `theta` y produce seis líneas. Las líneas se dibujan en rojo sobre una copia RGB de la entrada.

## Resultados de la corrida oficial

| Variante | Media (ms) | Desv. estándar | Speedup |
|---|---:|---:|---:|
| Global | 1.082563 | 0.003477 | 1.000x |
| Constante | 1.074896 | 0.003398 | 1.007x |
| Compartida | 0.798310 | 0.000502 | 1.356x |

Las tres variantes coincidieron exactamente con CPU. `memcheck` reportó cero errores y `racecheck` reportó cero peligros, errores o advertencias. La evidencia completa está en `results/`.

## Transformación de coordenadas

Cada hilo procesa el píxel lineal `global_id`:

```cpp
int x_coordinate = global_id % width - width / 2;
int y_coordinate = height / 2 - global_id / width;
```

El módulo obtiene la columna y la división entera obtiene la fila. Se resta el centro para trasladar el origen desde la esquina superior izquierda al centro de la imagen. La coordenada vertical se invierte porque las filas de una imagen aumentan hacia abajo, mientras que el eje cartesiano `y` aumenta hacia arriba.

## Memoria compartida y barreras

El acumulador por bloque contiene `90 * 100 = 9,000` enteros, es decir, 36,000 bytes. Todos los hilos inicializan el arreglo por pasos de `blockDim.x`, esperan en la primera barrera, votan mediante `atomicAdd`, esperan en la segunda barrera y finalmente fusionan el acumulador local con el global.

Ningún hilo retorna antes de las barreras; los hilos fuera de la imagen omiten la votación pero siguen participando en la sincronización.

## Validación de memoria

Después de compilar:

```bash
make sanitize
```

La ejecución debe terminar sin accesos inválidos reportados por `compute-sanitizer`.
El objetivo `sanitize` también ejecuta `racecheck` sobre el kernel compartido.

El lector PGM y escritor PNG cuentan con una prueba independiente:

```bash
make test-image-io
```

La prueba GPU opcional usa una imagen P2 de 7 x 5 píxeles, por lo que cubre un último bloque incompleto, y ejecuta las tres variantes con `pixel-threshold=32`:

```bash
cmake -S . -B build -DHOUGH_ENABLE_GPU_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## Estructura

```text
common/                 Lector PGM y escritor PNG autocontenidos
data/runway.pgm         Imagen de bordes proporcionada
docs/                   Borrador del informe y material conceptual
results/                CSV, PNG y datos generados
scripts/                Automatización de la corrida oficial
src/hough.cu            CPU, kernels CUDA, validación y visualización
tests/data/              Entrada sintética para el último bloque incompleto
BITACORA.md              Registro de pruebas y resultados
Makefile                 Compilación solicitada en el enunciado
CMakeLists.txt           Alternativa para Windows/Linux
```

## Referencias técnicas

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUDA Runtime API - Event Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)
- R. O. Duda y P. E. Hart, “Use of the Hough Transformation to Detect Lines and Curves in Pictures”, *Communications of the ACM*, 1972.
