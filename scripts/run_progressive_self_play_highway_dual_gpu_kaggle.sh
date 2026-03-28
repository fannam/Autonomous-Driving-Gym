#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Kaggle notebooks often export MPLBACKEND=module://matplotlib_inline.backend_inline,
# which breaks subprocess imports of matplotlib. Force a headless-safe backend.
if [[ "${MPLBACKEND:-}" == "module://matplotlib_inline.backend_inline" || -z "${MPLBACKEND:-}" ]]; then
  export MPLBACKEND="Agg"
fi

export ALPHAZERO_SCENARIO="highway"
export ALPHAZERO_CONFIG_PATH="$REPO_ROOT/configs/highway.yaml"

TRAINING_OUTPUT_ROOT_DEFAULT="$REPO_ROOT/AlphaZero-based-autonomous-driving/outputs/progressive_training/highway"
SELF_PLAY_OUTPUT_ROOT_DEFAULT="$REPO_ROOT/AlphaZero-based-autonomous-driving/outputs/progressive_self_play/highway"
BOOTSTRAP_MODEL_PATH_DEFAULT="$TRAINING_OUTPUT_ROOT_DEFAULT/model_iter_0000.pth"

find_latest_model() {
  find "$TRAINING_OUTPUT_ROOT_DEFAULT" -maxdepth 1 -type f -name 'model_iter_*.pth' 2>/dev/null \
    | sort \
    | tail -n 1
}

infer_next_iteration_from_model() {
  local model_path="$1"
  local model_name
  model_name="$(basename "$model_path")"
  if [[ "$model_name" =~ model_iter_([0-9]+)\.pth$ ]]; then
    printf '%d\n' "$((10#${BASH_REMATCH[1]} + 1))"
    return
  fi
  printf '1\n'
}

SOURCE_MODEL="${SOURCE_MODEL:-}"
if [[ -z "$SOURCE_MODEL" ]]; then
  SOURCE_MODEL="$(find_latest_model || true)"
fi

if [[ -n "$SOURCE_MODEL" && ! -f "$SOURCE_MODEL" ]]; then
  echo "SOURCE_MODEL does not exist: $SOURCE_MODEL" >&2
  exit 1
fi

if [[ -n "$SOURCE_MODEL" ]]; then
  ITERATION_DEFAULT="$(infer_next_iteration_from_model "$SOURCE_MODEL")"
else
  ITERATION_DEFAULT="1"
fi
ITERATION="${ITERATION:-$ITERATION_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SELF_PLAY_OUTPUT_ROOT_DEFAULT}"
BOOTSTRAP_MODEL_PATH="${BOOTSTRAP_MODEL_PATH:-$BOOTSTRAP_MODEL_PATH_DEFAULT}"
WORKERS="${WORKERS:-4}"
TOTAL_EPISODES="${TOTAL_EPISODES:-8}"
EPISODES_PER_WORKER="${EPISODES_PER_WORKER:-}"
DEVICE="${DEVICE:-cuda}"
NUM_GPUS="${NUM_GPUS:-2}"
GPU_INDICES="${GPU_INDICES:-0,1}"
N_SIMULATIONS="${N_SIMULATIONS:-400}"
MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-}"
RESULT_TIMEOUT="${RESULT_TIMEOUT:-}"

cmd=(
  uv run python
  AlphaZero-based-autonomous-driving/AlphaZero/scripts/progressive_self_play.py
  --iteration "$ITERATION"
  --output-root "$OUTPUT_ROOT"
  --bootstrap-model-path "$BOOTSTRAP_MODEL_PATH"
  --workers "$WORKERS"
  --device "$DEVICE"
  --n-simulations "$N_SIMULATIONS"
  --allow-non-cpu-device
)

if [[ -n "$SOURCE_MODEL" ]]; then
  cmd+=(--source-model "$SOURCE_MODEL")
fi

if [[ -n "$TOTAL_EPISODES" && -n "$EPISODES_PER_WORKER" ]]; then
  echo "Set only one of TOTAL_EPISODES or EPISODES_PER_WORKER." >&2
  exit 1
fi

episode_mode=""
if [[ -n "$TOTAL_EPISODES" ]]; then
  cmd+=(--total-episodes "$TOTAL_EPISODES")
  episode_mode="total_episodes=$TOTAL_EPISODES"
else
  EPISODES_PER_WORKER="${EPISODES_PER_WORKER:-200}"
  cmd+=(--episodes-per-worker "$EPISODES_PER_WORKER")
  episode_mode="episodes_per_worker=$EPISODES_PER_WORKER"
fi

if [[ -n "$MAX_STEPS_PER_EPISODE" ]]; then
  cmd+=(--max-steps-per-episode "$MAX_STEPS_PER_EPISODE")
fi

if [[ -n "$RESULT_TIMEOUT" ]]; then
  cmd+=(--result-timeout "$RESULT_TIMEOUT")
fi

device_lower="${DEVICE,,}"
if [[ "$device_lower" == "cuda" || "$device_lower" == "auto" ]]; then
  if [[ -n "$NUM_GPUS" ]]; then
    cmd+=(--num-gpus "$NUM_GPUS")
  fi

  if [[ -n "$GPU_INDICES" ]]; then
    cmd+=(--gpu-indices "$GPU_INDICES")
  fi
fi

cmd+=("$@")

if [[ -n "$SOURCE_MODEL" ]]; then
  source_model_display="$SOURCE_MODEL"
else
  source_model_display="bootstrap:$BOOTSTRAP_MODEL_PATH"
fi

printf '[self-play-highway-dual-gpu-kaggle] scenario=%s config=%s iteration=%s source_model=%s workers=%s %s device=%s n_simulations=%s num_gpus=%s gpu_indices=%s result_timeout=%s\n' \
  "$ALPHAZERO_SCENARIO" \
  "$ALPHAZERO_CONFIG_PATH" \
  "$ITERATION" \
  "$source_model_display" \
  "$WORKERS" \
  "$episode_mode" \
  "$DEVICE" \
  "$N_SIMULATIONS" \
  "${NUM_GPUS:-disabled}" \
  "${GPU_INDICES:-disabled}" \
  "${RESULT_TIMEOUT:-disabled}"

exec "${cmd[@]}"
