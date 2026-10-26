#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

mkdir -p results
nvidia-smi > results/gpu_info.txt

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

echo "Corrida completada. Revise timings.csv, detected_lines.csv y runway_lines.png."
