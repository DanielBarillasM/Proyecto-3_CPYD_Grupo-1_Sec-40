#!/usr/bin/env bash
# Ejecuta la corrida oficial en Linux/WSL desde cualquier directorio.
# -e: detenerse ante error; -u: variables indefinidas; pipefail: errores en pipes.
set -euo pipefail

# BASH_SOURCE apunta al script; subir un nivel obtiene la raíz del repositorio.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

# Conservar la configuración de GPU junto con los resultados reproducibles.
mkdir -p results
nvidia-smi > results/gpu_info.txt

# Ejecutar las tres variantes con idénticos parámetros experimentales.
./hough \
  --input data/runway.pgm \
  --mode all \
  --runs 10 \
  --warmup 2 \
  --sigma 2.4 \
  --max-lines 12 \
  --nms-radius 4 \
  --nms-theta 3 \
  --pixel-threshold 0 \
  --output results/runway_lines.png \
  --csv results/timings.csv \
  --lines-csv results/detected_lines.csv

# Este mensaje solo se imprime si todas las instrucciones anteriores funcionaron.
echo "Corrida completada. Revise timings.csv, detected_lines.csv y runway_lines.png."
