#include "common/image_io.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kDegreeIncrement = 2;
constexpr int kDegreeBins = 180 / kDegreeIncrement;
constexpr int kRadiusBins = 100;
constexpr int kAccumulatorSize = kDegreeBins * kRadiusBins;
constexpr int kDefaultBlockSize = 256;
constexpr double kPi = 3.14159265358979323846;

__constant__ float c_cos[kDegreeBins];
__constant__ float c_sin[kDegreeBins];

enum class KernelMode { Global, Constant, Shared };

struct Options {
  std::string input_path = "data/runway.pgm";
  std::string output_path = "results/runway_lines.png";
  std::string csv_path = "results/timings.csv";
  std::string lines_csv_path = "results/detected_lines.csv";
  std::string mode = "all";
  int runs = 10;
  int warmup_runs = 2;
  int max_lines = 20;
  int nms_radius = 2;
  int nms_theta = 2;
  int pixel_threshold = 0;
  double sigma = 2.0;
  bool help = false;
};

struct TimingResult {
  KernelMode mode{};
  std::vector<float> milliseconds;
  std::vector<int> accumulator;
};

struct TimingStats {
  double mean = 0.0;
  double median = 0.0;
  double standard_deviation = 0.0;
  double minimum = 0.0;
  double maximum = 0.0;
};

struct HoughPeak {
  int radius_bin = 0;
  int theta_bin = 0;
  int votes = 0;
};

void check_cuda(cudaError_t result, const char *expression, const char *file,
                int line) {
  if (result == cudaSuccess) {
    return;
  }
  std::ostringstream message;
  message << "CUDA error en " << file << ':' << line << " al ejecutar "
          << expression << ": " << cudaGetErrorString(result);
  throw std::runtime_error(message.str());
}

#define CUDA_CHECK(expression)                                                 \
  check_cuda((expression), #expression, __FILE__, __LINE__)

template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { allocate(count); }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  DeviceBuffer(DeviceBuffer &&other) noexcept
      : pointer_(other.pointer_), count_(other.count_) {
    other.pointer_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      release();
      pointer_ = other.pointer_;
      count_ = other.count_;
      other.pointer_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  ~DeviceBuffer() { release(); }

  void allocate(std::size_t count) {
    release();
    if (count == 0) {
      return;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&pointer_),
                          count * sizeof(T)));
    count_ = count;
  }

  T *get() const { return pointer_; }
  std::size_t size() const { return count_; }
  std::size_t bytes() const { return count_ * sizeof(T); }

private:
  void release() noexcept {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
      pointer_ = nullptr;
      count_ = 0;
    }
  }

  T *pointer_ = nullptr;
  std::size_t count_ = 0;
};

class CudaEvent {
public:
  CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }
  CudaEvent(const CudaEvent &) = delete;
  CudaEvent &operator=(const CudaEvent &) = delete;
  ~CudaEvent() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }
  cudaEvent_t get() const { return event_; }

private:
  cudaEvent_t event_ = nullptr;
};

__device__ __forceinline__ int radius_index(float radius, float radius_max,
                                             float radius_scale) {
  return static_cast<int>((radius + radius_max) / radius_scale);
}

__global__ void hough_global_kernel(const std::uint8_t *image, int width,
                                    int height, int *accumulator,
                                    float radius_max, float radius_scale,
                                    const float *cos_table,
                                    const float *sin_table,
                                    std::uint8_t pixel_threshold) {
  // Cada hilo obtiene un índice lineal y procesa como máximo un píxel.
  const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixel_count = width * height;
  if (global_id >= pixel_count) {
    return;
  }

  if (image[global_id] <= pixel_threshold) {
    return;
  }

  const int x_center = width / 2;
  const int y_center = height / 2;
  // El módulo da la columna y la división entera da la fila. Se invierte Y
  // porque en la imagen crece hacia abajo y en el plano cartesiano hacia arriba.
  const int x_coordinate = global_id % width - x_center;
  const int y_coordinate = y_center - global_id / width;

  for (int theta_index = 0; theta_index < kDegreeBins; ++theta_index) {
    const float radius = x_coordinate * cos_table[theta_index] +
                         y_coordinate * sin_table[theta_index];
    const int r_index = radius_index(radius, radius_max, radius_scale);
    if (r_index >= 0 && r_index < kRadiusBins) {
      atomicAdd(&accumulator[r_index * kDegreeBins + theta_index], 1);
    }
  }
}

__global__ void hough_constant_kernel(const std::uint8_t *image, int width,
                                      int height, int *accumulator,
                                      float radius_max,
                                      float radius_scale,
                                      std::uint8_t pixel_threshold) {
  // Se mantiene un hilo por píxel; solo cambia dónde se guardan seno y coseno.
  const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixel_count = width * height;
  if (global_id >= pixel_count) {
    return;
  }

  if (image[global_id] <= pixel_threshold) {
    return;
  }

  const int x_center = width / 2;
  const int y_center = height / 2;
  const int x_coordinate = global_id % width - x_center;
  const int y_coordinate = y_center - global_id / width;

  for (int theta_index = 0; theta_index < kDegreeBins; ++theta_index) {
    // En cada iteración el warp consulta el mismo ángulo, lo que permite usar
    // el broadcast de la memoria constante.
    const float radius = x_coordinate * c_cos[theta_index] +
                         y_coordinate * c_sin[theta_index];
    const int r_index = radius_index(radius, radius_max, radius_scale);
    if (r_index >= 0 && r_index < kRadiusBins) {
      atomicAdd(&accumulator[r_index * kDegreeBins + theta_index], 1);
    }
  }
}

__global__ void hough_shared_kernel(const std::uint8_t *image, int width,
                                    int height, int *accumulator,
                                    float radius_max, float radius_scale,
                                    std::uint8_t pixel_threshold) {
  // Cada bloque usa su propio acumulador y al final lo suma al global.
  __shared__ int local_accumulator[kAccumulatorSize];

  const int local_id = threadIdx.x;
  // Entre todos los hilos del bloque se limpian los 9,000 bins.
  for (int index = local_id; index < kAccumulatorSize;
       index += blockDim.x) {
    local_accumulator[index] = 0;
  }
  // Esperar hasta que todo el acumulador esté en cero.
  __syncthreads();

  const int global_id = blockIdx.x * blockDim.x + local_id;
  const int pixel_count = width * height;
  // Los hilos sobrantes no votan, pero deben llegar a la segunda barrera.
  if (global_id < pixel_count && image[global_id] > pixel_threshold) {
    const int x_center = width / 2;
    const int y_center = height / 2;
    const int x_coordinate = global_id % width - x_center;
    const int y_coordinate = y_center - global_id / width;

    for (int theta_index = 0; theta_index < kDegreeBins; ++theta_index) {
      const float radius = x_coordinate * c_cos[theta_index] +
                           y_coordinate * c_sin[theta_index];
      const int r_index = radius_index(radius, radius_max, radius_scale);
      if (r_index >= 0 && r_index < kRadiusBins) {
        atomicAdd(&local_accumulator[r_index * kDegreeBins + theta_index],
                  1);
      }
    }
  }

  // Esperar a que todos terminen de votar antes de copiar al acumulador global.
  __syncthreads();
  // Se omiten los bins vacíos. atomicAdd sigue siendo necesario porque varios
  // bloques pueden sumar al mismo bin global al mismo tiempo.
  for (int index = local_id; index < kAccumulatorSize;
       index += blockDim.x) {
    const int block_votes = local_accumulator[index];
    if (block_votes != 0) {
      atomicAdd(&accumulator[index], block_votes);
    }
  }
}

std::string mode_name(KernelMode mode) {
  switch (mode) {
  case KernelMode::Global:
    return "global";
  case KernelMode::Constant:
    return "constant";
  case KernelMode::Shared:
    return "shared";
  }
  return "unknown";
}

int parse_integer(const std::string &text, const std::string &option) {
  try {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size()) {
      throw std::invalid_argument("trailing data");
    }
    return value;
  } catch (...) {
    throw std::runtime_error("Valor invalido para " + option + ": " + text);
  }
}

double parse_double(const std::string &text, const std::string &option) {
  try {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value)) {
      throw std::invalid_argument("invalid double");
    }
    return value;
  } catch (...) {
    throw std::runtime_error("Valor invalido para " + option + ": " + text);
  }
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto require_value = [&]() -> std::string {
      if (index + 1 >= argc) {
        throw std::runtime_error("Falta un valor para " + argument);
      }
      return argv[++index];
    };

    if (argument == "--input" || argument == "-i") {
      options.input_path = require_value();
    } else if (argument == "--output" || argument == "-o") {
      options.output_path = require_value();
    } else if (argument == "--csv") {
      options.csv_path = require_value();
    } else if (argument == "--lines-csv") {
      options.lines_csv_path = require_value();
    } else if (argument == "--mode" || argument == "-m") {
      options.mode = require_value();
    } else if (argument == "--runs") {
      options.runs = parse_integer(require_value(), argument);
    } else if (argument == "--warmup") {
      options.warmup_runs = parse_integer(require_value(), argument);
    } else if (argument == "--max-lines") {
      options.max_lines = parse_integer(require_value(), argument);
    } else if (argument == "--nms-radius") {
      options.nms_radius = parse_integer(require_value(), argument);
    } else if (argument == "--nms-theta") {
      options.nms_theta = parse_integer(require_value(), argument);
    } else if (argument == "--pixel-threshold") {
      options.pixel_threshold = parse_integer(require_value(), argument);
    } else if (argument == "--sigma") {
      options.sigma = parse_double(require_value(), argument);
    } else if (argument == "--help" || argument == "-h") {
      options.help = true;
    } else {
      throw std::runtime_error("Opcion desconocida: " + argument);
    }
  }

  if (options.mode != "all" && options.mode != "global" &&
      options.mode != "constant" && options.mode != "shared") {
    throw std::runtime_error(
        "--mode debe ser global, constant, shared o all.");
  }
  if (options.runs < 1 || options.warmup_runs < 0 || options.max_lines < 1 ||
      options.sigma < 0.0 || options.nms_radius < 0 ||
      options.nms_radius >= kRadiusBins || options.nms_theta < 0 ||
      options.nms_theta >= kDegreeBins ||
      options.pixel_threshold < 0 || options.pixel_threshold > 255) {
    throw std::runtime_error(
        "Argumentos invalidos: runs/max-lines deben ser positivos; "
        "warmup/sigma/nms no negativos; nms-radius < 100, nms-theta < 90 y "
        "pixel-threshold entre 0 y 255.");
  }
  return options;
}

void print_usage(const char *program) {
  std::cout
      << "Uso: " << program << " [opciones]\n\n"
      << "  -i, --input RUTA       Imagen PGM de entrada\n"
      << "  -o, --output RUTA      PNG con las lineas detectadas\n"
      << "      --csv RUTA         CSV de tiempos por corrida\n"
      << "      --lines-csv RUTA   CSV de lineas detectadas\n"
      << "  -m, --mode MODO        global|constant|shared|all\n"
      << "      --runs N           Corridas medidas (default: 10)\n"
      << "      --warmup N         Corridas de calentamiento (default: 2)\n"
      << "      --sigma X          Umbral media + X*desv. (default: 2)\n"
      << "      --max-lines N      Maximo de lineas dibujadas (default: 20)\n"
      << "      --nms-radius N     Radio NMS en bins de r (default: 2)\n"
      << "      --nms-theta N      Radio NMS en bins theta (default: 2)\n"
      << "      --pixel-threshold N Ignorar pixeles <= N (default: 0)\n"
      << "  -h, --help             Mostrar esta ayuda\n";
}

std::vector<KernelMode> selected_modes(const std::string &mode) {
  if (mode == "global") {
    return {KernelMode::Global};
  }
  if (mode == "constant") {
    return {KernelMode::Constant};
  }
  if (mode == "shared") {
    return {KernelMode::Shared};
  }
  return {KernelMode::Global, KernelMode::Constant, KernelMode::Shared};
}

void launch_kernel(KernelMode mode, int grid_size, int block_size,
                   const std::uint8_t *device_image, int width, int height,
                   int *device_accumulator, float radius_max,
                   float radius_scale, const float *device_cos,
                   const float *device_sin,
                   std::uint8_t pixel_threshold) {
  switch (mode) {
  case KernelMode::Global:
    hough_global_kernel<<<grid_size, block_size>>>(
        device_image, width, height, device_accumulator, radius_max,
        radius_scale, device_cos, device_sin, pixel_threshold);
    break;
  case KernelMode::Constant:
    hough_constant_kernel<<<grid_size, block_size>>>(
        device_image, width, height, device_accumulator, radius_max,
        radius_scale, pixel_threshold);
    break;
  case KernelMode::Shared:
    hough_shared_kernel<<<grid_size, block_size>>>(
        device_image, width, height, device_accumulator, radius_max,
        radius_scale, pixel_threshold);
    break;
  }
  CUDA_CHECK(cudaGetLastError());
}

std::vector<int> cpu_hough(const GrayImage &image,
                           const std::vector<float> &cos_table,
                           const std::vector<float> &sin_table,
                           float radius_max, float radius_scale,
                           std::uint8_t pixel_threshold) {
  std::vector<int> accumulator(kAccumulatorSize, 0);
  const int x_center = image.width / 2;
  const int y_center = image.height / 2;

  for (int row = 0; row < image.height; ++row) {
    for (int column = 0; column < image.width; ++column) {
      const int pixel_index = row * image.width + column;
      if (image.pixels[static_cast<std::size_t>(pixel_index)] <=
          pixel_threshold) {
        continue;
      }
      const int x_coordinate = column - x_center;
      const int y_coordinate = y_center - row;
      for (int theta_index = 0; theta_index < kDegreeBins; ++theta_index) {
        const float radius = x_coordinate * cos_table[theta_index] +
                             y_coordinate * sin_table[theta_index];
        const int r_index =
            static_cast<int>((radius + radius_max) / radius_scale);
        if (r_index >= 0 && r_index < kRadiusBins) {
          ++accumulator[r_index * kDegreeBins + theta_index];
        }
      }
    }
  }
  return accumulator;
}

TimingResult run_kernel_mode(KernelMode mode, const Options &options,
                             const GrayImage &image,
                             const DeviceBuffer<std::uint8_t> &device_image,
                             DeviceBuffer<int> &device_accumulator,
                             const DeviceBuffer<float> &device_cos,
                             const DeviceBuffer<float> &device_sin,
                             float radius_max, float radius_scale) {
  const int pixel_count = image.width * image.height;
  const int grid_size =
      (pixel_count + kDefaultBlockSize - 1) / kDefaultBlockSize;

  for (int run = 0; run < options.warmup_runs; ++run) {
    // Estas corridas calientan la GPU; no se guardan en el CSV.
    CUDA_CHECK(cudaMemset(device_accumulator.get(), 0,
                          device_accumulator.bytes()));
    launch_kernel(mode, grid_size, kDefaultBlockSize, device_image.get(),
                  image.width, image.height, device_accumulator.get(),
                  radius_max, radius_scale, device_cos.get(), device_sin.get(),
                  static_cast<std::uint8_t>(options.pixel_threshold));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  CudaEvent start_event;
  CudaEvent stop_event;
  TimingResult result;
  result.mode = mode;
  result.milliseconds.reserve(static_cast<std::size_t>(options.runs));

  for (int run = 0; run < options.runs; ++run) {
    // cudaMemset queda fuera de la medición; los eventos rodean solo el kernel.
    CUDA_CHECK(cudaMemset(device_accumulator.get(), 0,
                          device_accumulator.bytes()));
    CUDA_CHECK(cudaEventRecord(start_event.get()));
    launch_kernel(mode, grid_size, kDefaultBlockSize, device_image.get(),
                  image.width, image.height, device_accumulator.get(),
                  radius_max, radius_scale, device_cos.get(), device_sin.get(),
                  static_cast<std::uint8_t>(options.pixel_threshold));
    CUDA_CHECK(cudaEventRecord(stop_event.get()));
    CUDA_CHECK(cudaEventSynchronize(stop_event.get()));
    float milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event.get(),
                                    stop_event.get()));
    result.milliseconds.push_back(milliseconds);
  }

  result.accumulator.resize(kAccumulatorSize);
  CUDA_CHECK(cudaMemcpy(result.accumulator.data(), device_accumulator.get(),
                        device_accumulator.bytes(), cudaMemcpyDeviceToHost));
  return result;
}

TimingStats calculate_stats(const std::vector<float> &values) {
  TimingStats stats;
  if (values.empty()) {
    return stats;
  }

  const auto [minimum, maximum] =
      std::minmax_element(values.begin(), values.end());
  stats.minimum = *minimum;
  stats.maximum = *maximum;
  stats.mean = std::accumulate(values.begin(), values.end(), 0.0) /
               static_cast<double>(values.size());

  std::vector<float> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  const std::size_t middle = sorted.size() / 2U;
  stats.median = sorted.size() % 2U == 0U
                     ? (static_cast<double>(sorted[middle - 1U]) +
                        static_cast<double>(sorted[middle])) /
                           2.0
                     : sorted[middle];

  double squared_sum = 0.0;
  for (const float value : values) {
    const double difference = value - stats.mean;
    squared_sum += difference * difference;
  }
  stats.standard_deviation =
      std::sqrt(squared_sum / static_cast<double>(values.size()));
  return stats;
}

void validate_accumulator(KernelMode mode, const std::vector<int> &reference,
                          const std::vector<int> &candidate) {
  if (reference.size() != candidate.size()) {
    throw std::runtime_error("Tamano de acumulador incorrecto en modo " +
                             mode_name(mode));
  }

  std::size_t mismatch_count = 0;
  std::size_t first_mismatch = 0;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    if (reference[index] != candidate[index]) {
      if (mismatch_count == 0) {
        first_mismatch = index;
      }
      ++mismatch_count;
    }
  }

  if (mismatch_count != 0) {
    std::ostringstream error;
    error << "Validacion CPU-GPU fallo en " << mode_name(mode) << ": "
          << mismatch_count << " bins distintos; primer bin "
          << first_mismatch << " CPU=" << reference[first_mismatch]
          << " GPU=" << candidate[first_mismatch];
    throw std::runtime_error(error.str());
  }
}

std::vector<HoughPeak> find_peaks(const std::vector<int> &accumulator,
                                  double sigma, int max_lines,
                                  int radius_neighborhood,
                                  int theta_neighborhood,
                                  double &threshold) {
  const double mean =
      std::accumulate(accumulator.begin(), accumulator.end(), 0.0) /
      static_cast<double>(accumulator.size());
  double squared_sum = 0.0;
  for (const int value : accumulator) {
    const double difference = value - mean;
    squared_sum += difference * difference;
  }
  const double standard_deviation =
      std::sqrt(squared_sum / static_cast<double>(accumulator.size()));
  threshold = mean + sigma * standard_deviation;

  std::vector<HoughPeak> peaks;

  for (int radius_bin = 0; radius_bin < kRadiusBins; ++radius_bin) {
    for (int theta_bin = 0; theta_bin < kDegreeBins; ++theta_bin) {
      const int index = radius_bin * kDegreeBins + theta_bin;
      const int value = accumulator[static_cast<std::size_t>(index)];
      if (static_cast<double>(value) <= threshold) {
        continue;
      }

      bool local_maximum = true;
      for (int radius_delta = -radius_neighborhood;
           radius_delta <= radius_neighborhood && local_maximum;
           ++radius_delta) {
        int neighbor_radius = radius_bin + radius_delta;
        if (neighbor_radius < 0 || neighbor_radius >= kRadiusBins) {
          continue;
        }
        for (int theta_delta = -theta_neighborhood;
             theta_delta <= theta_neighborhood; ++theta_delta) {
          int neighbor_theta = theta_bin + theta_delta;
          int mapped_radius = neighbor_radius;
          // Al cruzar 0/180 grados, la misma recta cambia el signo de r.
          if (neighbor_theta < 0) {
            neighbor_theta += kDegreeBins;
            mapped_radius = kRadiusBins - 1 - mapped_radius;
          } else if (neighbor_theta >= kDegreeBins) {
            neighbor_theta -= kDegreeBins;
            mapped_radius = kRadiusBins - 1 - mapped_radius;
          }
          if (mapped_radius == radius_bin &&
              neighbor_theta == theta_bin) {
            continue;
          }
          const int neighbor_index =
              mapped_radius * kDegreeBins + neighbor_theta;
          const int neighbor_value =
              accumulator[static_cast<std::size_t>(neighbor_index)];
          if (neighbor_value > value ||
              (neighbor_value == value && neighbor_index < index)) {
            local_maximum = false;
            break;
          }
        }
      }

      if (local_maximum) {
        peaks.push_back({radius_bin, theta_bin, value});
      }
    }
  }

  std::sort(peaks.begin(), peaks.end(),
            [](const HoughPeak &left, const HoughPeak &right) {
              if (left.votes != right.votes) {
                return left.votes > right.votes;
              }
              if (left.radius_bin != right.radius_bin) {
                return left.radius_bin < right.radius_bin;
              }
              return left.theta_bin < right.theta_bin;
            });

  std::vector<HoughPeak> selected;
  selected.reserve(static_cast<std::size_t>(max_lines));
  for (const HoughPeak &candidate : peaks) {
    bool duplicate = false;
    for (const HoughPeak &chosen : selected) {
      const int theta_distance =
          std::abs(candidate.theta_bin - chosen.theta_bin);
      const bool direct_neighbor =
          theta_distance <= theta_neighborhood &&
          std::abs(candidate.radius_bin - chosen.radius_bin) <=
              radius_neighborhood;
      const bool wrapped_neighbor =
          kDegreeBins - theta_distance <= theta_neighborhood &&
          std::abs(candidate.radius_bin -
                   (kRadiusBins - 1 - chosen.radius_bin)) <=
              radius_neighborhood;
      if (direct_neighbor || wrapped_neighbor) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) {
      selected.push_back(candidate);
      if (selected.size() == static_cast<std::size_t>(max_lines)) {
        break;
      }
    }
  }
  return selected;
}

void set_red_pixel(std::vector<std::uint8_t> &rgb, int width, int height,
                   int x, int y) {
  for (int y_offset = -1; y_offset <= 1; ++y_offset) {
    for (int x_offset = -1; x_offset <= 1; ++x_offset) {
      const int draw_x = x + x_offset;
      const int draw_y = y + y_offset;
      if (draw_x < 0 || draw_x >= width || draw_y < 0 || draw_y >= height) {
        continue;
      }
      const std::size_t index =
          (static_cast<std::size_t>(draw_y) * static_cast<std::size_t>(width) +
           static_cast<std::size_t>(draw_x)) *
          3U;
      rgb[index] = 255U;
      rgb[index + 1U] = 32U;
      rgb[index + 2U] = 32U;
    }
  }
}

void draw_segment(std::vector<std::uint8_t> &rgb, int width, int height,
                  int x0, int y0, int x1, int y1) {
  const int delta_x = std::abs(x1 - x0);
  const int step_x = x0 < x1 ? 1 : -1;
  const int delta_y = -std::abs(y1 - y0);
  const int step_y = y0 < y1 ? 1 : -1;
  int error = delta_x + delta_y;

  while (true) {
    set_red_pixel(rgb, width, height, x0, y0);
    if (x0 == x1 && y0 == y1) {
      break;
    }
    const int doubled_error = 2 * error;
    if (doubled_error >= delta_y) {
      error += delta_y;
      x0 += step_x;
    }
    if (doubled_error <= delta_x) {
      error += delta_x;
      y0 += step_y;
    }
  }
}

void draw_hough_line(std::vector<std::uint8_t> &rgb, int width, int height,
                     double radius, double theta) {
  struct Point {
    double x = 0.0;
    double y = 0.0;
  };

  const double cosine = std::cos(theta);
  const double sine = std::sin(theta);
  const int x_center = width / 2;
  const int y_center = height / 2;
  const double x_min = -static_cast<double>(x_center);
  const double x_max = static_cast<double>(width - 1 - x_center);
  const double y_min = static_cast<double>(y_center - (height - 1));
  const double y_max = static_cast<double>(y_center);
  constexpr double kEpsilon = 1.0e-9;
  constexpr double kBoundaryTolerance = 1.0e-6;

  std::vector<Point> intersections;
  auto add_point = [&](double x, double y) {
    if (x < x_min - kBoundaryTolerance || x > x_max + kBoundaryTolerance ||
        y < y_min - kBoundaryTolerance || y > y_max + kBoundaryTolerance) {
      return;
    }
    for (const Point &point : intersections) {
      const double dx = point.x - x;
      const double dy = point.y - y;
      if (dx * dx + dy * dy < 1.0e-8) {
        return;
      }
    }
    intersections.push_back({x, y});
  };

  if (std::abs(sine) > kEpsilon) {
    add_point(x_min, (radius - x_min * cosine) / sine);
    add_point(x_max, (radius - x_max * cosine) / sine);
  }
  if (std::abs(cosine) > kEpsilon) {
    add_point((radius - y_min * sine) / cosine, y_min);
    add_point((radius - y_max * sine) / cosine, y_max);
  }
  if (intersections.size() < 2U) {
    return;
  }

  std::size_t first = 0;
  std::size_t second = 1;
  double greatest_distance = -1.0;
  for (std::size_t i = 0; i < intersections.size(); ++i) {
    for (std::size_t j = i + 1U; j < intersections.size(); ++j) {
      const double dx = intersections[i].x - intersections[j].x;
      const double dy = intersections[i].y - intersections[j].y;
      const double distance = dx * dx + dy * dy;
      if (distance > greatest_distance) {
        greatest_distance = distance;
        first = i;
        second = j;
      }
    }
  }

  const int x0 = static_cast<int>(
      std::lround(intersections[first].x + static_cast<double>(x_center)));
  const int y0 = static_cast<int>(
      std::lround(static_cast<double>(y_center) - intersections[first].y));
  const int x1 = static_cast<int>(
      std::lround(intersections[second].x + static_cast<double>(x_center)));
  const int y1 = static_cast<int>(
      std::lround(static_cast<double>(y_center) - intersections[second].y));
  draw_segment(rgb, width, height, x0, y0, x1, y1);
}

void write_visualization(const GrayImage &image,
                         const std::vector<int> &accumulator,
                         const Options &options, float radius_max,
                         float radius_scale) {
  double threshold = 0.0;
  const std::vector<HoughPeak> peaks =
      find_peaks(accumulator, options.sigma, options.max_lines,
                 options.nms_radius, options.nms_theta, threshold);

  std::vector<std::uint8_t> rgb(image.pixels.size() * 3U);
  for (std::size_t index = 0; index < image.pixels.size(); ++index) {
    const std::uint8_t gray = image.pixels[index];
    rgb[index * 3U] = gray;
    rgb[index * 3U + 1U] = gray;
    rgb[index * 3U + 2U] = gray;
  }

  for (const HoughPeak &peak : peaks) {
    const double radius = -static_cast<double>(radius_max) +
                          (static_cast<double>(peak.radius_bin) + 0.5) *
                              static_cast<double>(radius_scale);
    const double theta = static_cast<double>(peak.theta_bin) *
                         static_cast<double>(kDegreeIncrement) * kPi / 180.0;
    draw_hough_line(rgb, image.width, image.height, radius, theta);
  }

  std::string error;
  if (!save_png_rgb(options.output_path, image.width, image.height, rgb,
                    error)) {
    throw std::runtime_error(error);
  }

  const std::filesystem::path lines_path(options.lines_csv_path);
  if (lines_path.has_parent_path()) {
    std::filesystem::create_directories(lines_path.parent_path());
  }
  std::ofstream lines_output(options.lines_csv_path);
  if (!lines_output) {
    throw std::runtime_error("No se pudo crear el CSV de lineas: " +
                             options.lines_csv_path);
  }
  lines_output << "rank,radius_bin,theta_bin,radius_pixels,theta_degrees,votes\n";
  lines_output << std::fixed << std::setprecision(3);
  for (std::size_t index = 0; index < peaks.size(); ++index) {
    const HoughPeak &peak = peaks[index];
    const double radius = -static_cast<double>(radius_max) +
                          (static_cast<double>(peak.radius_bin) + 0.5) *
                              static_cast<double>(radius_scale);
    lines_output << index + 1U << ',' << peak.radius_bin << ','
                 << peak.theta_bin << ',' << radius << ','
                 << peak.theta_bin * kDegreeIncrement << ',' << peak.votes
                 << '\n';
  }

  std::cout << "Lineas: umbral=" << std::fixed << std::setprecision(2)
            << threshold << ", maximos locales dibujados=" << peaks.size()
            << "\nPNG: " << options.output_path
            << "\nCSV de lineas: " << options.lines_csv_path << '\n';
}

void write_csv(const std::string &path,
               const std::vector<TimingResult> &results) {
  const std::filesystem::path csv_path(path);
  if (csv_path.has_parent_path()) {
    std::filesystem::create_directories(csv_path.parent_path());
  }
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("No se pudo crear el CSV: " + path);
  }
  output << "mode,run,kernel_ms\n";
  output << std::fixed << std::setprecision(6);
  for (const TimingResult &result : results) {
    for (std::size_t run = 0; run < result.milliseconds.size(); ++run) {
      output << mode_name(result.mode) << ',' << run + 1U << ','
             << result.milliseconds[run] << '\n';
    }
  }
}

int run_project(const Options &options) {
  GrayImage image;
  std::string image_error;
  if (!load_pgm(options.input_path, image, image_error)) {
    throw std::runtime_error(image_error);
  }
  const std::size_t pixel_count = image.pixels.size();
  if (pixel_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("La imagen excede el rango de indices int.");
  }

  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  const std::size_t required_shared_memory =
      static_cast<std::size_t>(kAccumulatorSize) * sizeof(int);
  if (required_shared_memory > properties.sharedMemPerBlock) {
    std::ostringstream error;
    error << "El kernel compartido requiere " << required_shared_memory
          << " bytes por bloque, pero el dispositivo permite "
          << properties.sharedMemPerBlock << '.';
    throw std::runtime_error(error.str());
  }

  std::cout << "GPU: " << properties.name << " (CC " << properties.major
            << '.' << properties.minor << ")\n"
            << "Imagen: " << image.width << 'x' << image.height << ", "
            << pixel_count << " pixeles\n"
            << "Memoria compartida requerida por bloque: "
            << required_shared_memory << " bytes\n";

  std::vector<float> cos_table(kDegreeBins);
  std::vector<float> sin_table(kDegreeBins);
  for (int index = 0; index < kDegreeBins; ++index) {
    const double radians = static_cast<double>(index * kDegreeIncrement) * kPi /
                           180.0;
    cos_table[static_cast<std::size_t>(index)] =
        static_cast<float>(std::cos(radians));
    sin_table[static_cast<std::size_t>(index)] =
        static_cast<float>(std::sin(radians));
  }

  const float radius_max =
      std::sqrt(static_cast<float>(image.width * image.width +
                                   image.height * image.height)) /
      2.0F;
  const float radius_scale = 2.0F * radius_max / kRadiusBins;
  std::cout << "Calculando referencia CPU...\n";
  const std::vector<int> reference =
      cpu_hough(image, cos_table, sin_table, radius_max, radius_scale,
                static_cast<std::uint8_t>(options.pixel_threshold));
  const long long reference_votes =
      std::accumulate(reference.begin(), reference.end(), 0LL);
  std::cout << "Votos de referencia: " << reference_votes << '\n';

  DeviceBuffer<std::uint8_t> device_image(pixel_count);
  DeviceBuffer<int> device_accumulator(kAccumulatorSize);
  DeviceBuffer<float> device_cos(kDegreeBins);
  DeviceBuffer<float> device_sin(kDegreeBins);

  CUDA_CHECK(cudaMemcpy(device_image.get(), image.pixels.data(),
                        device_image.bytes(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_cos.get(), cos_table.data(), device_cos.bytes(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_sin.get(), sin_table.data(), device_sin.bytes(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyToSymbol(c_cos, cos_table.data(),
                                cos_table.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_sin, sin_table.data(),
                                sin_table.size() * sizeof(float)));

  std::vector<TimingResult> results;
  std::vector<int> visualization_accumulator;
  for (const KernelMode mode : selected_modes(options.mode)) {
    std::cout << "Ejecutando modo " << mode_name(mode) << "...\n";
    TimingResult result = run_kernel_mode(
        mode, options, image, device_image, device_accumulator, device_cos,
        device_sin, radius_max, radius_scale);
    validate_accumulator(mode, reference, result.accumulator);
    const TimingStats stats = calculate_stats(result.milliseconds);
    std::cout << std::fixed << std::setprecision(4)
              << "  Validacion CPU-GPU: OK\n"
              << "  ms: media=" << stats.mean << ", mediana=" << stats.median
              << ", desv=" << stats.standard_deviation
              << ", min=" << stats.minimum << ", max=" << stats.maximum
              << '\n';
    visualization_accumulator = result.accumulator;
    results.push_back(std::move(result));
  }

  write_csv(options.csv_path, results);
  write_visualization(image, visualization_accumulator, options, radius_max,
                      radius_scale);
  std::cout << "CSV: " << options.csv_path << "\nProyecto completado.\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.help) {
      print_usage(argv[0]);
      return 0;
    }

    const int status = run_project(options);
    CUDA_CHECK(cudaDeviceReset());
    return status;
  } catch (const std::exception &exception) {
    std::cerr << "ERROR: " << exception.what() << '\n';
    const cudaError_t reset_result = cudaDeviceReset();
    if (reset_result != cudaSuccess && reset_result != cudaErrorNoDevice &&
        reset_result != cudaErrorInsufficientDriver) {
      std::cerr << "Aviso: cudaDeviceReset fallo: "
                << cudaGetErrorString(reset_result) << '\n';
    }
    return 1;
  }
}
