#pragma once

// Interfaz mínima de entrada/salida usada por el proyecto.
// Se mantiene separada de CUDA para poder probarla con un compilador C++ normal.

#include <cstdint>
#include <string>
#include <vector>

/**
 * Imagen en escala de grises almacenada por filas.
 * Cada píxel se normaliza al intervalo [0, 255], aunque el PGM original use
 * un valor máximo diferente.
 */
struct GrayImage {
  int width = 0;                   // Número de columnas.
  int height = 0;                  // Número de filas.
  int max_value = 255;             // Máximo normalizado después de leer.
  std::vector<std::uint8_t> pixels; // Raster: índice = fila * width + columna.
};

/**
 * Lee un PGM ASCII (P2) o binario (P5) de hasta 8 bits.
 * @param path Ruta del archivo de entrada.
 * @param image Destino que solo se reemplaza si la lectura termina bien.
 * @param error Mensaje de diagnóstico; queda vacío cuando retorna true.
 */
bool load_pgm(const std::string &path, GrayImage &image, std::string &error);

/**
 * Escribe un PNG RGB de 8 bits sin depender de bibliotecas externas.
 * @param rgb Buffer intercalado R,G,B con width * height * 3 bytes.
 */
bool save_png_rgb(const std::string &path, int width, int height,
                  const std::vector<std::uint8_t> &rgb,
                  std::string &error);
