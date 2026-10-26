#include "common/image_io.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
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

  const std::filesystem::path p2_path = "results/image_io_comments_test.pgm";
  {
    std::ofstream p2(p2_path);
    p2 << "P2\n# comentario de prueba\n3 2\n15\n0 15 7\n1 2 3\n";
  }
  GrayImage p2_image;
  if (!load_pgm(p2_path.string(), p2_image, error)) {
    std::cerr << "No se pudo leer P2 con comentario: " << error << '\n';
    return 4;
  }
  const std::vector<std::uint8_t> expected = {0U, 255U, 119U,
                                               17U, 34U, 51U};
  if (p2_image.width != 3 || p2_image.height != 2 ||
      p2_image.pixels != expected) {
    std::cerr << "Escalado o comentarios P2 incorrectos.\n";
    return 5;
  }

  GrayImage missing;
  if (load_pgm("results/archivo_que_no_existe.pgm", missing, error)) {
    std::cerr << "Una ruta inexistente no debe aceptarse.\n";
    return 6;
  }
  std::filesystem::remove(p2_path);

  std::cout << "PGM/PNG OK: " << image.width << 'x' << image.height << '\n';
  return 0;
}
