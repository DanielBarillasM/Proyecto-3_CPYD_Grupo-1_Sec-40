# Suite de integración ejecutada por CTest o mediante cmake -P.
# Cada caso invoca el ejecutable real y aprovecha su validación CPU-GPU interna.

cmake_minimum_required(VERSION 3.24)

foreach(required_variable HOUGH_EXECUTABLE SOURCE_DIR OUTPUT_DIR)
  if(NOT DEFINED ${required_variable})
    message(FATAL_ERROR "Falta -D${required_variable}=...")
  endif()
endforeach()

if(NOT EXISTS "${HOUGH_EXECUTABLE}")
  message(FATAL_ERROR "No existe el ejecutable: ${HOUGH_EXECUTABLE}")
endif()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")

# Calcula los votos teóricos directamente desde un PGM P2 de prueba.
function(assert_p2_fixture input_path pixel_threshold expected_votes)
  file(READ "${input_path}" pgm_text)
  string(REGEX REPLACE "#[^\r\n]*" "" pgm_text "${pgm_text}")
  string(STRIP "${pgm_text}" pgm_text)
  string(REGEX REPLACE "[ \t\r\n]+" ";" pgm_tokens "${pgm_text}")

  list(GET pgm_tokens 0 magic)
  list(GET pgm_tokens 1 width)
  list(GET pgm_tokens 2 height)
  if(NOT magic STREQUAL "P2")
    message(FATAL_ERROR "La fixture debe ser P2: ${input_path}")
  endif()

  list(SUBLIST pgm_tokens 4 -1 pixels)
  list(LENGTH pixels actual_pixel_count)
  math(EXPR expected_pixel_count "${width} * ${height}")
  if(NOT actual_pixel_count EQUAL expected_pixel_count)
    message(FATAL_ERROR
      "${input_path}: contiene ${actual_pixel_count} muestras; "
      "se esperaban ${expected_pixel_count}.")
  endif()

  set(active_pixels 0)
  foreach(pixel IN LISTS pixels)
    if(pixel GREATER pixel_threshold)
      math(EXPR active_pixels "${active_pixels} + 1")
    endif()
  endforeach()
  math(EXPR calculated_votes "${active_pixels} * 90")
  if(NOT calculated_votes EQUAL expected_votes)
    message(FATAL_ERROR
      "${input_path}: votos teóricos=${calculated_votes}, "
      "esperados=${expected_votes}.")
  endif()
endfunction()

# Ejecuta las tres variantes y comprueba votos, validaciones y artefactos.
function(run_hough_case name input_file pixel_threshold expected_votes)
  set(input_path "${SOURCE_DIR}/tests/data/${input_file}")
  set(case_dir "${OUTPUT_DIR}/${name}")
  set(png_path "${case_dir}/lines.png")
  set(timings_path "${case_dir}/timings.csv")
  set(lines_path "${case_dir}/detected_lines.csv")
  set(stdout_path "${case_dir}/stdout.log")
  set(stderr_path "${case_dir}/stderr.log")
  file(MAKE_DIRECTORY "${case_dir}")
  assert_p2_fixture("${input_path}" "${pixel_threshold}" "${expected_votes}")

  execute_process(
    COMMAND "${HOUGH_EXECUTABLE}"
      --input "${input_path}"
      --mode all
      --runs 1
      --warmup 0
      --sigma 2.0
      --max-lines 20
      --nms-radius 2
      --nms-theta 2
      --pixel-threshold "${pixel_threshold}"
      --output "${png_path}"
      --csv "${timings_path}"
      --lines-csv "${lines_path}"
    RESULT_VARIABLE exit_code
    OUTPUT_FILE "${stdout_path}"
    ERROR_FILE "${stderr_path}"
    TIMEOUT 60
  )

  # Leer después de que el proceso cierre los archivos evita pérdidas de salida
  # observadas ocasionalmente al capturar OUTPUT_VARIABLE bajo CTest en Windows.
  file(READ "${stdout_path}" standard_output)
  file(READ "${stderr_path}" standard_error)

  if(NOT exit_code EQUAL 0)
    message(FATAL_ERROR
      "Caso ${name} terminó con ${exit_code}.\n"
      "STDOUT:\n${standard_output}\nSTDERR:\n${standard_error}")
  endif()

  foreach(artifact "${png_path}" "${timings_path}" "${lines_path}")
    if(NOT EXISTS "${artifact}")
      message(FATAL_ERROR "Caso ${name}: falta el artefacto ${artifact}")
    endif()
  endforeach()

  file(STRINGS "${timings_path}" timing_rows)
  list(LENGTH timing_rows timing_row_count)
  if(NOT timing_row_count EQUAL 4)
    message(FATAL_ERROR
      "Caso ${name}: timings.csv debe tener cabecera y 3 observaciones.")
  endif()

  # Confirmar que el ejecutable completó exactamente las tres estrategias.
  list(REMOVE_AT timing_rows 0)
  set(measured_modes "")
  foreach(row IN LISTS timing_rows)
    string(REPLACE "," ";" fields "${row}")
    list(GET fields 0 measured_mode)
    list(APPEND measured_modes "${measured_mode}")
  endforeach()
  foreach(expected_mode global constant shared)
    if(NOT expected_mode IN_LIST measured_modes)
      message(FATAL_ERROR
        "Caso ${name}: falta el modo ${expected_mode}; encontrados=${measured_modes}")
    endif()
  endforeach()

  set(${name}_LINES "${lines_path}" PARENT_SCOPE)
  message(STATUS
    "${name}: ${expected_votes} votos teóricos; Global/Constante/Compartida OK")
endfunction()

# Comprueba la orientación del pico de mayor rango en un CSV de líneas.
function(assert_first_theta name lines_path expected_theta)
  file(STRINGS "${lines_path}" rows)
  list(LENGTH rows row_count)
  if(row_count LESS 2)
    message(FATAL_ERROR "Caso ${name}: no produjo ningún pico de Hough.")
  endif()
  list(GET rows 1 first_data_row)
  string(REPLACE "," ";" fields "${first_data_row}")
  list(GET fields 4 actual_theta)
  if(NOT actual_theta STREQUAL "${expected_theta}")
    message(FATAL_ERROR
      "Caso ${name}: theta principal=${actual_theta}, esperado=${expected_theta}.")
  endif()
  message(STATUS "${name}: theta principal ${actual_theta} grados")
endfunction()

# Comprueba que un CSV contenga picos para todas las orientaciones indicadas.
function(assert_contains_thetas name lines_path)
  file(STRINGS "${lines_path}" rows)
  set(found_thetas "")
  foreach(row IN LISTS rows)
    if(row MATCHES "^rank,")
      continue()
    endif()
    string(REPLACE "," ";" fields "${row}")
    list(GET fields 4 theta)
    list(APPEND found_thetas "${theta}")
  endforeach()
  foreach(expected_theta IN LISTS ARGN)
    if(NOT expected_theta IN_LIST found_thetas)
      message(FATAL_ERROR
        "Caso ${name}: no apareció theta=${expected_theta}; encontrados=${found_thetas}")
    endif()
  endforeach()
  message(STATUS "${name}: contiene las orientaciones requeridas ${ARGN}")
endfunction()

run_hough_case(black black_8x8.pgm 0 0)
run_hough_case(single_pixel single_pixel_9x9.pgm 0 90)
run_hough_case(horizontal horizontal_17x17.pgm 0 1530)
run_hough_case(vertical vertical_17x17.pgm 0 1530)
run_hough_case(crossing crossing_lines_17x17.pgm 0 2970)
run_hough_case(threshold threshold_3x1.pgm 32 90)
run_hough_case(incomplete_block synthetic_7x5.pgm 32 1350)

assert_first_theta(horizontal "${horizontal_LINES}" 90)
assert_first_theta(vertical "${vertical_LINES}" 0)
assert_contains_thetas(crossing "${crossing_LINES}" 0 90)

message(STATUS "Suite GPU sintética completada correctamente.")
