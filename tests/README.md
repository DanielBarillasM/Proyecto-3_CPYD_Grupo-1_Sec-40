# Suite de pruebas

Esta carpeta contiene pruebas unitarias e integrales para comprobar entrada/salida, argumentos de línea de comandos y las tres variantes CUDA. Ninguna prueba modifica los resultados oficiales: todos los temporales se escriben dentro de `build/`, carpeta ignorada por Git.

## Matriz de casos sintéticos

Cada píxel activo emite un voto por cada uno de los 90 ángulos. Por ello, el número total esperado puede calcularse antes de ejecutar la GPU.

| Entrada | Propósito | Píxeles activos | Votos esperados | Comprobación adicional |
|---|---|---:|---:|---|
| `black_8x8.pgm` | Caso vacío | 0 | 0 | No debe producir picos |
| `single_pixel_9x9.pgm` | Votación mínima | 1 | 90 | Tres acumuladores iguales a CPU |
| `horizontal_17x17.pgm` | Recta horizontal centrada | 17 | 1,530 | Pico principal en 90 grados |
| `vertical_17x17.pgm` | Recta vertical centrada | 17 | 1,530 | Pico principal en 0 grados |
| `crossing_lines_17x17.pgm` | Dos orientaciones simultáneas | 33 | 2,970 | Deben aparecer 0 y 90 grados |
| `threshold_3x1.pgm` | Límite del umbral | 1 con umbral 32 | 90 | 31 y 32 no votan; 33 sí |
| `synthetic_7x5.pgm` | Último bloque incompleto | 15 | 1,350 | 35 píxeles no son múltiplo de 256 |

Para cada caso, `run_gpu_cases.cmake` exige:

- código de salida cero;
- cantidad exacta de votos de la referencia CPU;
- tres mensajes `Validacion CPU-GPU: OK`;
- un tiempo para Global, Constante y Compartida;
- creación del PNG, CSV de tiempos y CSV de líneas;
- orientación esperada en los casos geométricos.

## Prueba de imágenes

`test_image_io.cpp` no requiere GPU y cubre:

- lectura de la imagen P5 oficial;
- dimensiones y cantidad de píxeles;
- conversión de gris a RGB y firma PNG;
- P2 con comentarios y escalado desde `max_value=15`;
- rechazo de rutas inexistentes;
- rechazo de valores P2 fuera de rango;
- rechazo de un raster P5 truncado.

## Pruebas CLI

CTest también comprueba:

- `--help` termina correctamente y muestra el uso;
- un modo desconocido se rechaza;
- `--runs 0` se rechaza;
- una imagen inexistente se rechaza.

## Ejecución con CMake y CTest

Configurar todas las pruebas, incluidas las que necesitan GPU:

```bash
cmake -S . -B build/tests -DBUILD_TESTING=ON -DHOUGH_ENABLE_GPU_TESTS=ON
cmake --build build/tests --config Release
ctest --test-dir build/tests -C Release --output-on-failure
```

Solo pruebas CPU/CLI:

```bash
cmake -S . -B build/tests -DBUILD_TESTING=ON -DHOUGH_ENABLE_GPU_TESTS=OFF
cmake --build build/tests --config Release
ctest --test-dir build/tests -C Release --output-on-failure
```

## Ejecución con Make

```bash
make test-image-io
make test-gpu-cases
```

O ejecutar ambas:

```bash
make test-all
```

## Ejecución directa de la suite GPU

Después de compilar `hough`:

```bash
cmake \
  -DHOUGH_EXECUTABLE="$PWD/hough" \
  -DSOURCE_DIR="$PWD" \
  -DOUTPUT_DIR="$PWD/build/test-output/gpu-cases" \
  -P tests/run_gpu_cases.cmake
```

En Windows cambia `hough` por la ruta absoluta de `hough.exe`. Las salidas quedan en `build/test-output/` y pueden borrarse sin afectar el proyecto.
