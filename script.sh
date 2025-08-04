#!/usr/bin/bash
/home/streltso/miniconda/envs/pmw-analysis-env/bin/python3.12 /home/streltso/git/pmw-analysis/src/pmw_analysis/quantization/script.py --step factor --config configs/quantization.yml
/home/streltso/miniconda/envs/pmw-analysis-env/bin/python3.12 /home/streltso/git/pmw-analysis/src/pmw_analysis/quantization/script.py --step quantize --config configs/quantization.yml
/home/streltso/miniconda/envs/pmw-analysis-env/bin/python3.12 /home/streltso/git/pmw-analysis/src/pmw_analysis/quantization/script.py --step merge --config configs/quantization.yml
