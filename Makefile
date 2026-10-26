NVCC ?= nvcc
CXX ?= c++
TARGET ?= hough
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo --fmad=false
CPPFLAGS ?= -I.

SOURCES := src/hough.cu common/image_io.cpp
HEADERS := common/image_io.h

.PHONY: all run test-image-io sanitize clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(SOURCES) -o $(TARGET)

run: $(TARGET)
	./$(TARGET) --input data/runway.pgm --mode all --runs 10 --warmup 2 --sigma 2.4 --max-lines 12 --nms-radius 4 --nms-theta 3 --output results/runway_lines.png --csv results/timings.csv --lines-csv results/detected_lines.csv

test-image-io:
	$(CXX) -O2 -std=c++17 -I. tests/test_image_io.cpp common/image_io.cpp -o test_image_io
	./test_image_io

sanitize: $(TARGET)
	compute-sanitizer --tool memcheck ./$(TARGET) --input data/runway.pgm --mode all --runs 1 --warmup 0 --sigma 2.4 --nms-radius 4 --nms-theta 3 --output results/runway_lines_sanitized.png --csv results/timings_sanitized.csv --lines-csv results/detected_lines_sanitized.csv
	compute-sanitizer --tool racecheck ./$(TARGET) --input data/runway.pgm --mode shared --runs 1 --warmup 0 --sigma 2.4 --nms-radius 4 --nms-theta 3 --output results/runway_lines_racecheck.png --csv results/timings_racecheck.csv --lines-csv results/detected_lines_racecheck.csv

clean:
	$(RM) $(TARGET) $(TARGET).exe $(TARGET).exp $(TARGET).lib test_image_io test_image_io.exe
