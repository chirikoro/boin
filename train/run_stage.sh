#!/bin/bash
# Applio を CLI で動かす薄いラッパー（Mac, MPS）
# 使い方:
#   run_stage.sh preprocess <model> <dataset_dir>
#   run_stage.sh extract    <model>
#   run_stage.sh train      <model> <total_epoch> [batch_size] [save_every]
#   run_stage.sh blend      <new_model> <pth1> <pth2> <ratio 0.0-1.0>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ORIG="$PWD"
cd "$HERE/Applio"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
PY=.venv/bin/python
SR=40000
STAGE="$1"; shift
case "$STAGE" in
  preprocess)
    MODEL="$1"; DATASET="$(cd "$ORIG" && cd "$2" && pwd)"
    $PY core.py preprocess --model-name "$MODEL" --dataset-path "$DATASET" --sample-rate $SR \
      --cpu-cores 10 --cut-preprocess Automatic --chunk-len 3.0 --overlap-len 0.3 --normalization-mode none ;;
  extract)
    MODEL="$1"
    $PY core.py extract --model-name "$MODEL" --f0-method rmvpe --sample-rate $SR --gpu 0 --cpu-cores 4 \
      --embedder-model contentvec --include-mutes 2 ;;
  train)
    MODEL="$1"; EPOCHS="$2"; BS="${3:-8}"; SAVE="${4:-25}"
    $PY core.py train --model-name "$MODEL" --sample-rate $SR --gpu 0 --total-epoch "$EPOCHS" \
      --batch-size "$BS" --save-every-epoch "$SAVE" --save-only-latest --vocoder HiFi-GAN --pretrained --index-algorithm Auto ;;
  blend)
    NEW="$1"; RATIO="$4"
    P1="$(cd "$ORIG" && cd "$(dirname "$2")" && pwd)/$(basename "$2")"
    P2="$(cd "$ORIG" && cd "$(dirname "$3")" && pwd)/$(basename "$3")"
    $PY core.py model-blender --model-name "$NEW" --pth-path-1 "$P1" --pth-path-2 "$P2" --ratio "$RATIO" ;;
  *) echo "unknown stage: $STAGE"; exit 2 ;;
esac
