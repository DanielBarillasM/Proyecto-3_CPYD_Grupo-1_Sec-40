# Compiladores configurables desde la línea de comandos o el entorno.
NVCC ?= nvcc
CXX ?= c++
# CMake también se usa como ejecutor portable de la suite de integración.
CMAKE ?= cmake
# Nombre del ejecutable; permite usar TARGET=build/prueba-local/hough.
TARGET ?= hough
# --fmad=false conserva comparabilidad exacta con la referencia CPU.
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo --fmad=false
# La raíz se agrega al include path para common/image_io.h.
CPPFLAGS ?= -I.

# Archivos que obligan a recompilar el programa cuando cambian.
SOURCES := src/hough.cu common/image_io.cpp
HEADERS := common/image_io.h

# Estos nombres representan acciones y no archivos generados.
.PHONY: all run test-image-io test-gpu-cases test-all sanitize clean

# Objetivo predeterminado: construir el ejecutable CUDA.
all: $(TARGET)

# nvcc compila el .cu y delega el .cpp al compilador C++ del sistema.
$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(SOURCES) -o $(TARGET)

# Corrida oficial: tres variantes, dos calentamientos y diez mediciones.
run: $(TARGET)
	./$(TARGET) --input data/runway.pgm --mode all --runs 10 --warmup 2 --sigma 2.4 --max-lines 12 --nms-radius 4 --nms-theta 3 --output results/runway_lines.png --csv results/timings.csv --lines-csv results/detected_lines.csv

# Prueba aislada del lector PGM y escritor PNG; no requiere GPU.
test-image-io:
	$(CMAKE) -E make_directory build/test-output/image_io
	$(CXX) -O2 -std=c++17 -I. tests/test_image_io.cpp common/image_io.cpp -o build/test_image_io
	./build/test_image_io build/test-output/image_io

# Casos sintéticos GPU: votos conocidos, orientaciones y bloque incompleto.
test-gpu-cases: $(TARGET)
	$(CMAKE) -DHOUGH_EXECUTABLE=$(abspath $(TARGET)) -DSOURCE_DIR=$(CURDIR) -DOUTPUT_DIR=$(CURDIR)/build/test-output/gpu-cases -P tests/run_gpu_cases.cmake

# Ejecutar todas las pruebas disponibles en una máquina con GPU CUDA.
test-all: test-image-io test-gpu-cases

# memcheck valida accesos; racecheck busca carreras en la variante compartida.
sanitize: $(TARGET)
	compute-sanitizer --tool memcheck ./$(TARGET) --input data/runway.pgm --mode all --runs 1 --warmup 0 --sigma 2.4 --nms-radius 4 --nms-theta 3 --output results/runway_lines_sanitized.png --csv results/timings_sanitized.csv --lines-csv results/detected_lines_sanitized.csv
	compute-sanitizer --tool racecheck ./$(TARGET) --input data/runway.pgm --mode shared --runs 1 --warmup 0 --sigma 2.4 --nms-radius 4 --nms-theta 3 --output results/runway_lines_racecheck.png --csv results/timings_racecheck.csv --lines-csv results/detected_lines_racecheck.csv

# Eliminar únicamente binarios creados por este Makefile.
clean:
	$(RM) $(TARGET) $(TARGET).exe $(TARGET).exp $(TARGET).lib test_image_io test_image_io.exe build/test_image_io build/test_image_io.exe
