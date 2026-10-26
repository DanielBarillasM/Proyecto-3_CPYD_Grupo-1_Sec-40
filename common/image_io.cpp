#include "image_io.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

bool next_pgm_token(std::istream &input, std::string &token) {
  token.clear();
  char ch = 0;

  while (input.get(ch)) {
    if (std::isspace(static_cast<unsigned char>(ch))) {
      continue;
    }
    if (ch == '#') {
      input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }
    token.push_back(ch);
    break;
  }

  if (token.empty()) {
    return false;
  }

  while (input.get(ch)) {
    if (ch == '#') {
      input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      break;
    }
    if (std::isspace(static_cast<unsigned char>(ch))) {
      if (ch == '\r' && input.peek() == '\n') {
        input.get();
      }
      break;
    }
    token.push_back(ch);
  }
  return true;
}

bool parse_positive_int(const std::string &text, int &value) {
  try {
    std::size_t consumed = 0;
    const long parsed = std::stol(text, &consumed, 10);
    if (consumed != text.size() || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max()) {
      return false;
    }
    value = static_cast<int>(parsed);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_nonnegative_int(const std::string &text, int &value) {
  try {
    std::size_t consumed = 0;
    const long parsed = std::stol(text, &consumed, 10);
    if (consumed != text.size() || parsed < 0 ||
        parsed > std::numeric_limits<int>::max()) {
      return false;
    }
    value = static_cast<int>(parsed);
    return true;
  } catch (...) {
    return false;
  }
}

void append_be32(std::vector<std::uint8_t> &output, std::uint32_t value) {
  output.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xffU));
  output.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xffU));
  output.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xffU));
  output.push_back(static_cast<std::uint8_t>(value & 0xffU));
}

void write_be32(std::ostream &output, std::uint32_t value) {
  const std::array<char, 4> bytes{
      static_cast<char>((value >> 24U) & 0xffU),
      static_cast<char>((value >> 16U) & 0xffU),
      static_cast<char>((value >> 8U) & 0xffU),
      static_cast<char>(value & 0xffU)};
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::uint32_t crc32(const std::uint8_t *data, std::size_t size) {
  std::uint32_t crc = 0xffffffffU;
  for (std::size_t i = 0; i < size; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      const std::uint32_t mask = 0U - (crc & 1U);
      crc = (crc >> 1U) ^ (0xedb88320U & mask);
    }
  }
  return crc ^ 0xffffffffU;
}

std::uint32_t adler32(const std::vector<std::uint8_t> &data) {
  constexpr std::uint32_t kMod = 65521U;
  std::uint32_t a = 1U;
  std::uint32_t b = 0U;
  for (const std::uint8_t byte : data) {
    a = (a + byte) % kMod;
    b = (b + a) % kMod;
  }
  return (b << 16U) | a;
}

void write_png_chunk(std::ostream &output, const char type[4],
                     const std::vector<std::uint8_t> &data) {
  write_be32(output, static_cast<std::uint32_t>(data.size()));
  output.write(type, 4);

  if (!data.empty()) {
    output.write(reinterpret_cast<const char *>(data.data()),
                 static_cast<std::streamsize>(data.size()));
  }

  std::vector<std::uint8_t> crc_input;
  crc_input.reserve(4U + data.size());
  crc_input.insert(crc_input.end(), type, type + 4);
  crc_input.insert(crc_input.end(), data.begin(), data.end());
  write_be32(output, crc32(crc_input.data(), crc_input.size()));
}

std::vector<std::uint8_t>
make_uncompressed_zlib_stream(const std::vector<std::uint8_t> &raw) {
  std::vector<std::uint8_t> output;
  output.reserve(raw.size() + raw.size() / 65535U * 5U + 16U);

  // CMF/FLG for DEFLATE with a 32 KiB window and fastest compression level.
  output.push_back(0x78U);
  output.push_back(0x01U);

  std::size_t offset = 0;
  while (offset < raw.size()) {
    const std::size_t remaining = raw.size() - offset;
    const std::uint16_t length = static_cast<std::uint16_t>(
        std::min<std::size_t>(remaining, 65535U));
    const bool final_block = offset + length == raw.size();

    // Stored DEFLATE block: BFINAL plus BTYPE=00, padded to a byte boundary.
    output.push_back(final_block ? 0x01U : 0x00U);
    output.push_back(static_cast<std::uint8_t>(length & 0xffU));
    output.push_back(static_cast<std::uint8_t>((length >> 8U) & 0xffU));
    const std::uint16_t inverse = static_cast<std::uint16_t>(~length);
    output.push_back(static_cast<std::uint8_t>(inverse & 0xffU));
    output.push_back(static_cast<std::uint8_t>((inverse >> 8U) & 0xffU));
    output.insert(output.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset),
                  raw.begin() + static_cast<std::ptrdiff_t>(offset + length));
    offset += length;
  }

  append_be32(output, adler32(raw));
  return output;
}

} // namespace

bool load_pgm(const std::string &path, GrayImage &image, std::string &error) {
  try {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      error = "No se pudo abrir la imagen PGM: " + path;
      return false;
    }

    std::string magic;
    std::string width_token;
    std::string height_token;
    std::string max_token;
    if (!next_pgm_token(input, magic) || !next_pgm_token(input, width_token) ||
        !next_pgm_token(input, height_token) ||
        !next_pgm_token(input, max_token)) {
      error = "Encabezado PGM incompleto.";
      return false;
    }

    int width = 0;
    int height = 0;
    int max_value = 0;
    if ((magic != "P5" && magic != "P2") ||
        !parse_positive_int(width_token, width) ||
        !parse_positive_int(height_token, height) ||
        !parse_positive_int(max_token, max_value) || max_value > 255) {
      error = "PGM no compatible; se requiere P2/P5 de 8 bits.";
      return false;
    }

    const std::size_t pixel_count =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (pixel_count == 0 || pixel_count >
                                std::numeric_limits<std::size_t>::max() /
                                    sizeof(std::uint8_t)) {
      error = "Dimensiones PGM invalidas.";
      return false;
    }

    GrayImage parsed;
    parsed.width = width;
    parsed.height = height;
    parsed.max_value = max_value;
    parsed.pixels.resize(pixel_count);

    if (magic == "P5") {
      input.read(reinterpret_cast<char *>(parsed.pixels.data()),
                 static_cast<std::streamsize>(parsed.pixels.size()));
      if (input.gcount() != static_cast<std::streamsize>(parsed.pixels.size())) {
        error = "El raster PGM esta truncado.";
        return false;
      }
      if (max_value != 255) {
        for (std::uint8_t &pixel : parsed.pixels) {
          const int scaled =
              (static_cast<int>(pixel) * 255 + max_value / 2) / max_value;
          pixel = static_cast<std::uint8_t>(scaled);
        }
      }
    } else {
      std::string value_token;
      for (std::size_t i = 0; i < pixel_count; ++i) {
        int value = 0;
        if (!next_pgm_token(input, value_token) ||
            !parse_nonnegative_int(value_token, value) || value > max_value) {
          error = "Valor de pixel P2 invalido o raster truncado.";
          return false;
        }
        const int scaled = (value * 255 + max_value / 2) / max_value;
        parsed.pixels[i] = static_cast<std::uint8_t>(scaled);
      }
    }

    parsed.max_value = 255;

    image = std::move(parsed);
    error.clear();
    return true;
  } catch (const std::exception &ex) {
    error = ex.what();
    return false;
  }
}

bool save_png_rgb(const std::string &path, int width, int height,
                  const std::vector<std::uint8_t> &rgb,
                  std::string &error) {
  try {
    if (width <= 0 || height <= 0 ||
        rgb.size() != static_cast<std::size_t>(width) *
                          static_cast<std::size_t>(height) * 3U) {
      error = "Dimensiones o buffer RGB invalidos.";
      return false;
    }

    const std::filesystem::path output_path(path);
    if (output_path.has_parent_path()) {
      std::error_code directory_error;
      std::filesystem::create_directories(output_path.parent_path(),
                                          directory_error);
      if (directory_error) {
        error = "No se pudo crear el directorio de salida: " +
                directory_error.message();
        return false;
      }
    }

    std::ofstream output(path, std::ios::binary);
    if (!output) {
      error = "No se pudo crear el PNG: " + path;
      return false;
    }

    const std::array<std::uint8_t, 8> signature{0x89U, 'P', 'N', 'G',
                                                 0x0dU, 0x0aU, 0x1aU,
                                                 0x0aU};
    output.write(reinterpret_cast<const char *>(signature.data()),
                 static_cast<std::streamsize>(signature.size()));

    std::vector<std::uint8_t> ihdr;
    ihdr.reserve(13);
    append_be32(ihdr, static_cast<std::uint32_t>(width));
    append_be32(ihdr, static_cast<std::uint32_t>(height));
    ihdr.push_back(8U); // bit depth
    ihdr.push_back(2U); // truecolor RGB
    ihdr.push_back(0U); // compression method
    ihdr.push_back(0U); // filter method
    ihdr.push_back(0U); // no interlace
    write_png_chunk(output, "IHDR", ihdr);

    const std::size_t stride = static_cast<std::size_t>(width) * 3U;
    std::vector<std::uint8_t> scanlines;
    scanlines.reserve((stride + 1U) * static_cast<std::size_t>(height));
    for (int row = 0; row < height; ++row) {
      scanlines.push_back(0U); // PNG filter: None
      const auto begin = rgb.begin() + static_cast<std::ptrdiff_t>(row) *
                                           static_cast<std::ptrdiff_t>(stride);
      scanlines.insert(scanlines.end(), begin,
                       begin + static_cast<std::ptrdiff_t>(stride));
    }

    const std::vector<std::uint8_t> compressed =
        make_uncompressed_zlib_stream(scanlines);
    write_png_chunk(output, "IDAT", compressed);
    write_png_chunk(output, "IEND", {});

    if (!output) {
      error = "Fallo al escribir el archivo PNG.";
      return false;
    }
    error.clear();
    return true;
  } catch (const std::exception &ex) {
    error = ex.what();
    return false;
  }
}
