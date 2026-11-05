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
  # Registrar el binario como una prueba de CTest.
  add_test(NAME image_io COMMAND test_image_io)
  # Ejecutar desde la raíz para resolver data/runway.pgm correctamente.
  set_tests_properties(image_io PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
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
  endif()
endif()
