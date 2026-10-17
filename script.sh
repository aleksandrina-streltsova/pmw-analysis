#!/usr/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}/src"

/home/streltso/miniconda/envs/pmw-analysis-env/bin/python3.12 -m pmw_analysis.preprocessing.script --step factor --config "${ROOT_DIR}/configs/quantization.yml"
/home/streltso/miniconda/envs/pmw-analysis-env/bin/python3.12 -m pmw_analysis.preprocessing.script --step quantize --config "${ROOT_DIR}/configs/quantization.yml"
/home/streltso/miniconda/envs/pmw-analysis-env/bin/python3.12 -m pmw_analysis.preprocessing.script --step merge --config "${ROOT_DIR}/configs/quantization.yml"
