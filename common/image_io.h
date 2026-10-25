#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct GrayImage {
  int width = 0;
  int height = 0;
  int max_value = 255;
  std::vector<std::uint8_t> pixels;
};

bool load_pgm(const std::string &path, GrayImage &image, std::string &error);

bool save_png_rgb(const std::string &path, int width, int height,
                  const std::vector<std::uint8_t> &rgb,
                  std::string &error);
