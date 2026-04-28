#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_PATH="${1:-${ROOT_DIR}/configs/quantization_new_quantize_merge.yml}"
OUTPUT_DIR="${2:-/ltenas8/data/PMW_Analysis}"
PYTHON_BIN="${PYTHON_BIN:-/home/streltso/miniconda/envs/pmw-analysis-env/bin/python}"

export PYTHONPATH="${ROOT_DIR}/src"

"${PYTHON_BIN}" -m pmw_analysis.preprocessing.script_new_quantize_merge \
  --config "${CONFIG_PATH}" \
  --dir "${OUTPUT_DIR}" \
  --step statistics

"${PYTHON_BIN}" -m pmw_analysis.preprocessing.script_new_quantize_merge \
  --config "${CONFIG_PATH}" \
  --dir "${OUTPUT_DIR}" \
  --step quantize

"${PYTHON_BIN}" -m pmw_analysis.preprocessing.script_new_quantize_merge \
  --config "${CONFIG_PATH}" \
  --dir "${OUTPUT_DIR}" \
  --step merge
