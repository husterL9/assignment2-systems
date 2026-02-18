#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${OUT_DIR:-nsys_reports}
BATCH_SIZE=${BATCH_SIZE:-4}
WARMUP=${WARMUP:-5}
STEPS=${STEPS:-10}
DEVICE=${DEVICE:-cuda}
BACKWARD=${BACKWARD:-0}
OPTIMIZER=${OPTIMIZER:-0}
TIME_FORWARD_ONLY=${TIME_FORWARD_ONLY:-1}
TRACE=${TRACE:-cuda,nvtx,osrt}
PYTORCH=${PYTORCH:-}
# trace_label=${TRACE//,/+}
pytorch_label=${PYTORCH//,/+}
ts=$(date +%Y%m%d_%H%M%S)
fo_suffix=""
if [[ "$TIME_FORWARD_ONLY" == "1" ]]; then
  fo_suffix="_time-forward-only"
fi
RESULT_FILE=${RESULT_FILE:-$OUT_DIR/benchmark_times${fo_suffix}_pytorch${pytorch_label}_${ts}.csv}

mkdir -p "$OUT_DIR"

sizes=(small medium large xl 2_7b)
contexts=(128 256 512 1024)

for size in "${sizes[@]}"; do
  case "$size" in
    small)
      d_model=768
      d_ff=3072
      num_layers=12
      num_heads=12
      label=small
      ;;
    medium)
      d_model=1024
      d_ff=4096
      num_layers=24
      num_heads=16
      label=medium
      ;;
    large)
      d_model=1280
      d_ff=5120
      num_layers=36
      num_heads=20
      label=large
      ;;
    xl)
      d_model=1600
      d_ff=6400
      num_layers=48
      num_heads=25
      label=xl
      ;;
    2_7b)
      d_model=2560
      d_ff=10240
      num_layers=32
      num_heads=32
      label=2.7b
      ;;
    *)
      echo "Unknown size: $size" >&2
      exit 1
      ;;
  esac

  for ctx in "${contexts[@]}"; do
    out_base="$OUT_DIR/${label}_ctx${ctx}_backward${BACKWARD}_optimizer${OPTIMIZER}${fo_suffix}_pytorch${pytorch_label}_${ts}"
    echo "==> size=$label ctx=$ctx"

    py_args=()
    if [[ "$BACKWARD" == "1" ]]; then
      py_args+=(--backward)
    fi
    if [[ "$OPTIMIZER" == "1" ]]; then
      py_args+=(--optimizer)
    fi
    if [[ "$TIME_FORWARD_ONLY" == "1" ]]; then
      py_args+=(--time-forward-only)
    fi

    if ! nsys profile \
      --force-overwrite true \
      --trace="$TRACE" \
      --pytorch="$PYTORCH" \
      -o "$out_base" \
      uv run python -m cs336_systems.benchmark \
        --context-length "$ctx" \
        --d-model "$d_model" \
        --d-ff "$d_ff" \
        --num-layers "$num_layers" \
        --num-heads "$num_heads" \
        --batch-size "$BATCH_SIZE" \
        --warmup "$WARMUP" \
        --steps "$STEPS" \
        --device "$DEVICE" \
        --model-size "$label" \
        --result-file "$RESULT_FILE" \
        --nvtx \
        "${py_args[@]}"; then
      echo "FAILED: size=$label ctx=$ctx (see error above)" >&2
    fi
  done
  echo
 done
