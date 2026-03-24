#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SELF_PLAY_OUTPUT_ROOT_DEFAULT="$REPO_ROOT/AlphaZero-based-autonomous-driving/outputs/progressive_self_play"
TRAINING_OUTPUT_ROOT_DEFAULT="$REPO_ROOT/AlphaZero-based-autonomous-driving/outputs/progressive_training"

find_latest_batch() {
  find "$SELF_PLAY_OUTPUT_ROOT_DEFAULT" -mindepth 1 -maxdepth 1 -type d -name 'iter_*' 2>/dev/null \
    | sort \
    | tail -n 1
}

find_latest_model() {
  find "$TRAINING_OUTPUT_ROOT_DEFAULT" -maxdepth 1 -type f -name 'model_iter_*.pth' 2>/dev/null \
    | sort \
    | tail -n 1
}

infer_iteration_from_batch() {
  local batch_dir="$1"
  local batch_name
  batch_name="$(basename "$batch_dir")"
  if [[ "$batch_name" =~ iter_([0-9]+)$ ]]; then
    printf '%d\n' "$((10#${BASH_REMATCH[1]}))"
    return
  fi
  printf '1\n'
}

infer_model_for_iteration() {
  local iteration="$1"
  if (( iteration <= 0 )); then
    return 0
  fi
  printf '%s/model_iter_%04d.pth\n' "$TRAINING_OUTPUT_ROOT_DEFAULT" "$((iteration - 1))"
}

BATCH_DIR="${BATCH_DIR:-}"
if [[ -z "$BATCH_DIR" ]]; then
  BATCH_DIR="$(find_latest_batch || true)"
fi

if [[ -z "$BATCH_DIR" ]]; then
  cat <<EOF
BATCH_DIR is not set and no local self-play batch was found under:
  $SELF_PLAY_OUTPUT_ROOT_DEFAULT

Set it explicitly, for example:
  BATCH_DIR=/path/to/iter_0001 bash scripts/run_progressive_train.sh
EOF
  exit 1
fi

if [[ ! -d "$BATCH_DIR" ]]; then
  echo "BATCH_DIR does not exist: $BATCH_DIR" >&2
  exit 1
fi

ITERATION="${ITERATION:-$(infer_iteration_from_batch "$BATCH_DIR")}"
MODEL_IN="${MODEL_IN:-}"
if [[ -z "$MODEL_IN" ]]; then
  MODEL_IN="$(infer_model_for_iteration "$ITERATION")"
fi
if [[ ! -f "$MODEL_IN" ]]; then
  MODEL_IN="$(find_latest_model || true)"
fi

if [[ -z "$MODEL_IN" || ! -f "$MODEL_IN" ]]; then
  cat <<EOF
MODEL_IN is not set and no suitable local checkpoint was found.

Tried:
  $(infer_model_for_iteration "$ITERATION")
  latest under $TRAINING_OUTPUT_ROOT_DEFAULT

Set it explicitly, for example:
  MODEL_IN=/path/to/model_iter_0000.pth bash scripts/run_progressive_train.sh
EOF
  exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$TRAINING_OUTPUT_ROOT_DEFAULT}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"

cmd=(
  uv run python
  AlphaZero-based-autonomous-driving/AlphaZero/scripts/progressive_train.py
  --iteration "$ITERATION"
  --model-in "$MODEL_IN"
  --batch-dir "$BATCH_DIR"
  --device "$DEVICE"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --output-root "$OUTPUT_ROOT"
)

cmd+=("$@")

printf '[train] iteration=%s model_in=%s batch_dir=%s device=%s epochs=%s batch_size=%s\n' \
  "$ITERATION" "$MODEL_IN" "$BATCH_DIR" "$DEVICE" "$EPOCHS" "$BATCH_SIZE"

exec "${cmd[@]}"
