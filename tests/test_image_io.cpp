// Prueba de integración del módulo de imágenes.
// No usa CUDA y puede ejecutarse en cualquier máquina con C++17.
#include "common/image_io.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  // Todos los temporales se concentran en build/ para no ensuciar Git.
  const std::filesystem::path output_directory =
      argc > 1 ? std::filesystem::path(argv[1])
               : std::filesystem::path("build/test-output/image_io");
  std::filesystem::create_directories(output_directory);

  // Caso real: cargar la imagen que usa el benchmark oficial.
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

  // Convertir escala de grises a RGB repitiendo el canal en R, G y B.
  std::vector<std::uint8_t> rgb(image.pixels.size() * 3U);
  for (std::size_t index = 0; index < image.pixels.size(); ++index) {
    rgb[index * 3U] = image.pixels[index];
    rgb[index * 3U + 1U] = image.pixels[index];
    rgb[index * 3U + 2U] = image.pixels[index];
  }

  // El PNG debe poder escribirse usando exactamente el buffer construido.
  const std::filesystem::path png_path = output_directory / "image_io_test.png";
  if (!save_png_rgb(png_path.string(), image.width, image.height,
                    rgb, error)) {
    std::cerr << error << '\n';
    return 3;
  }

  // Verificar también la firma física del archivo, no solo el valor de retorno.
  {
    // El bloque cierra el archivo antes de intentar eliminarlo en Windows.
    std::ifstream png_input(png_path, std::ios::binary);
    std::array<unsigned char, 8> png_signature{};
    png_input.read(reinterpret_cast<char *>(png_signature.data()),
                   static_cast<std::streamsize>(png_signature.size()));
    const std::array<unsigned char, 8> expected_signature{
        0x89U, 'P', 'N', 'G', 0x0dU, 0x0aU, 0x1aU, 0x0aU};
    if (!png_input || png_signature != expected_signature) {
      std::cerr << "La firma del PNG generado es incorrecta.\n";
      return 4;
    }
  }

  // Caso sintético P2: comprueba comentarios, espacios y escalado desde 0..15.
  const std::filesystem::path p2_path = output_directory / "comments_test.pgm";
  {
    // El bloque limita la vida del stream y fuerza su cierre antes de leer.
    std::ofstream p2(p2_path);
    p2 << "P2\n# comentario de prueba\n3 2\n15\n0 15 7\n1 2 3\n";
  }
  GrayImage p2_image;
  if (!load_pgm(p2_path.string(), p2_image, error)) {
    std::cerr << "No se pudo leer P2 con comentario: " << error << '\n';
    return 5;
  }
  // Valores esperados tras normalizar cada muestra al intervalo 0..255.
  const std::vector<std::uint8_t> expected = {0U, 255U, 119U,
                                               17U, 34U, 51U};
  if (p2_image.width != 3 || p2_image.height != 2 ||
      p2_image.pixels != expected) {
    std::cerr << "Escalado o comentarios P2 incorrectos.\n";
    return 6;
  }

  // Caso negativo: una ruta inexistente debe fallar y no producir una imagen.
  GrayImage missing;
  if (load_pgm((output_directory / "missing.pgm").string(), missing, error)) {
    std::cerr << "Una ruta inexistente no debe aceptarse.\n";
    return 7;
  }

  // Un valor P2 mayor que max_value viola el formato y debe rechazarse.
  const std::filesystem::path range_path = output_directory / "bad_range.pgm";
  {
    std::ofstream invalid_range(range_path);
    invalid_range << "P2\n2 1\n15\n0 16\n";
  }
  GrayImage invalid_range_image;
  if (load_pgm(range_path.string(), invalid_range_image, error)) {
    std::cerr << "Un P2 con valores fuera de rango no debe aceptarse.\n";
    return 8;
  }

  // Un P5 de 2x2 necesita cuatro bytes; aquí se escriben solo tres.
  const std::filesystem::path truncated_path =
      output_directory / "truncated.pgm";
  {
    std::ofstream truncated(truncated_path, std::ios::binary);
    truncated << "P5\n2 2\n255\n";
    const std::array<char, 3> pixels{0, 1, 2};
    truncated.write(pixels.data(),
                    static_cast<std::streamsize>(pixels.size()));
  }
  GrayImage truncated_image;
  if (load_pgm(truncated_path.string(), truncated_image, error)) {
    std::cerr << "Un raster P5 truncado no debe aceptarse.\n";
    return 9;
  }

  // Limpiar únicamente los archivos temporales creados por esta prueba.
  std::filesystem::remove(p2_path);
  std::filesystem::remove(range_path);
  std::filesystem::remove(truncated_path);
  std::filesystem::remove(png_path);

  // Cero comunica éxito a CTest, Make y cualquier entorno de integración.
  std::cout << "PGM/PNG OK: " << image.width << 'x' << image.height << '\n';
  return 0;
}
