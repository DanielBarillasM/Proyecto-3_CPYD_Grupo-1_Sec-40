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
  --sigma 2 \
  --max-lines 20 \
  --output results/runway_lines.png \
  --csv results/timings.csv

echo "Corrida completada. Revise results/timings.csv y results/runway_lines.png."
