cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(hough_cuda LANGUAGES CXX CUDA)

add_executable(hough
  src/hough.cu
  common/image_io.cpp
  common/image_io.h
)

target_include_directories(hough PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(hough PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(hough PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;--fmad=false>
)

set_target_properties(hough PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

include(CTest)
if(BUILD_TESTING)
  add_executable(test_image_io
    tests/test_image_io.cpp
    common/image_io.cpp
    common/image_io.h
  )
  target_include_directories(test_image_io PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_features(test_image_io PRIVATE cxx_std_17)
  add_test(NAME image_io COMMAND test_image_io)
  set_tests_properties(image_io PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  option(HOUGH_ENABLE_GPU_TESTS "Ejecutar pruebas que requieren una GPU CUDA" OFF)
  if(HOUGH_ENABLE_GPU_TESTS)
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
