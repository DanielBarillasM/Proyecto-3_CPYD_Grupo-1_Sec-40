# Se requiere CMake 3.24 para el soporte moderno de CUDA_ARCHITECTURES.
cmake_minimum_required(VERSION 3.24)

# Si el usuario no fija una arquitectura, compilar para la GPU disponible.
# En servidores sin GPU se puede sustituir con -DCMAKE_CUDA_ARCHITECTURES=XX.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# El ejecutable mezcla código C++17 de host y kernels CUDA C++17.
project(hough_cuda LANGUAGES CXX CUDA)

# Programa principal: algoritmo, kernels y módulo autocontenido de imágenes.
add_executable(hough
  src/hough.cu
  common/image_io.cpp
  common/image_io.h
)

# Permitir includes relativos a la raíz, por ejemplo common/image_io.h.
target_include_directories(hough PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Exigir el mismo estándar tanto al compilador host como a nvcc.
target_compile_features(hough PRIVATE cxx_std_17 cuda_std_17)
# Optimizar kernels, conservar números de línea y evitar diferencias CPU/GPU
# causadas por la contracción automática de multiplicación y suma (FMA).
target_compile_options(hough PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;--fmad=false>
)

# Todos los kernels viven en una sola unidad CUDA; no hace falta device linking.
set_target_properties(hough PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

# Habilitar CTest y crear pruebas cuando BUILD_TESTING esté activo.
include(CTest)
if(BUILD_TESTING)
  # Esta prueba no necesita GPU: valida lectura PGM y escritura PNG.
  add_executable(test_image_io
    tests/test_image_io.cpp
    common/image_io.cpp
    common/image_io.h
  )
  # La prueba comparte la misma raíz de includes que el programa principal.
  target_include_directories(test_image_io PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_features(test_image_io PRIVATE cxx_std_17)
  # Registrar el binario y enviar sus temporales a la carpeta de compilación.
  add_test(
    NAME image_io
    COMMAND test_image_io ${CMAKE_CURRENT_BINARY_DIR}/test-output/image_io
  )
  # Ejecutar desde la raíz para resolver data/runway.pgm correctamente.
  set_tests_properties(image_io PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    LABELS "cpu;io"
  )

  # Estas pruebas terminan antes de inicializar la GPU y validan la interfaz CLI.
  add_test(NAME cli_help COMMAND hough --help)
  set_tests_properties(cli_help PROPERTIES
    PASS_REGULAR_EXPRESSION "Uso:"
    LABELS "cli"
  )

  add_test(NAME cli_rejects_invalid_mode COMMAND hough --mode invalid)
  add_test(NAME cli_rejects_zero_runs COMMAND hough --runs 0)
  add_test(
    NAME cli_rejects_missing_input
    COMMAND hough --input tests/data/does_not_exist.pgm --runs 1 --warmup 0
  )
  set_tests_properties(
    cli_rejects_invalid_mode
    cli_rejects_zero_runs
    cli_rejects_missing_input
    PROPERTIES
      WILL_FAIL TRUE
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      LABELS "cli;negative"
  )

  # Las pruebas GPU se desactivan por defecto para permitir CI sin CUDA físico.
  option(HOUGH_ENABLE_GPU_TESTS "Ejecutar pruebas que requieren una GPU CUDA" OFF)
  if(HOUGH_ENABLE_GPU_TESTS)
    # La imagen 7x5 produce un último bloque incompleto y comprueba que los
    # hilos sobrantes no abandonen el kernel compartido antes de las barreras.
    add_test(
      NAME hough_gpu_incomplete_block
      COMMAND hough
        --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/synthetic_7x5.pgm
        --mode all
        --runs 1
        --warmup 0
        --pixel-threshold 32
        --output ${CMAKE_CURRENT_BINARY_DIR}/synthetic_7x5.png
        --csv ${CMAKE_CURRENT_BINARY_DIR}/synthetic_7x5_timings.csv
        --lines-csv ${CMAKE_CURRENT_BINARY_DIR}/synthetic_7x5_lines.csv
    )
    set_tests_properties(hough_gpu_incomplete_block PROPERTIES
      LABELS "gpu;integration"
      TIMEOUT 60
    )

    # Suite con votos conocidos, orientaciones conocidas y umbral de intensidad.
    add_test(
      NAME hough_gpu_synthetic_suite
      COMMAND ${CMAKE_COMMAND}
        -DHOUGH_EXECUTABLE=$<TARGET_FILE:hough>
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/test-output/gpu-cases
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_gpu_cases.cmake
    )
    set_tests_properties(hough_gpu_synthetic_suite PROPERTIES
      LABELS "gpu;integration;synthetic"
      TIMEOUT 120
    )
  endif()
endif()
