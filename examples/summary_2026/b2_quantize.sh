#!/usr/bin/env bash
# Run the statistics, quantize, and merge steps of the new quantization pipeline with configs/quantization_v5.yml,
# each under /usr/bin/time -v (wall time, peak memory). Usage: b2_quantize.sh <output dir> [extra options]
set -euo pipefail

REPO=/home/streltso/git/pmw-analysis
PYTHON=/home/streltso/miniconda/envs/pmw-analysis-env/bin/python
OUT_DIR=$1
shift

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"  # the script writes its logs to ./logs
for step in statistics quantize merge; do
    echo "$(date '+%F %T') start $step"
    PYTHONPATH=$REPO/src /usr/bin/time -v "$PYTHON" -u "$REPO/src/pmw_analysis/preprocessing/script_new_quantize_merge.py" \
        --config "$REPO/configs/quantization_v5.yml" --dir "$OUT_DIR" --step "$step" "$@"
    echo "$(date '+%F %T') done $step"
done
echo "$(date '+%F %T') FINISHED"
