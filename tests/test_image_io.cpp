#include "common/image_io.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
  GrayImage image;
  std::string error;
  if (!load_pgm("data/runway.pgm", image, error)) {
    std::cerr << error << '\n';
    return 1;
  }
  if (image.width != 800 || image.height != 600 ||
      image.pixels.size() != 480000U) {
    std::cerr << "Dimensiones PGM inesperadas.\n";
    return 2;
  }

  std::vector<std::uint8_t> rgb(image.pixels.size() * 3U);
  for (std::size_t index = 0; index < image.pixels.size(); ++index) {
    rgb[index * 3U] = image.pixels[index];
    rgb[index * 3U + 1U] = image.pixels[index];
    rgb[index * 3U + 2U] = image.pixels[index];
  }

  if (!save_png_rgb("results/image_io_test.png", image.width, image.height,
                    rgb, error)) {
    std::cerr << error << '\n';
    return 3;
  }
  std::cout << "PGM/PNG OK: " << image.width << 'x' << image.height << '\n';
  return 0;
}
