#!/usr/bin/env bash
# Resume PPO-traditional training from an existing checkpoint.
#
# Không tái sinh config YAML — mặc định dùng snapshot config đã được tạo ở lần
# train đầu (OUTPUT_DIR/highway_ppo_traditional_local.yaml) để giữ nguyên
# hyperparameters. Checkpoint (.pt) sẽ được load qua --resume-from, metrics
# được append (không truncate).
#
# Ví dụ: resume thêm 30 updates nữa từ checkpoint sẵn có
#   bash scripts/run_ppo_traditional_resume_local.sh
# Ví dụ: resume 50 updates, đổi OUTPUT_DIR
#   UPDATES=50 OUTPUT_DIR=/path/to/other_run bash scripts/run_ppo_traditional_resume_local.sh
set -euo pipefail

EXPECTED_TRAIN_RELATIVE="source/algorithm/PPO-based/PPO-traditional/PPOTraditional/scripts/train.py"

has_repo_layout() {
  local root="$1"
  [[ -f "$root/$EXPECTED_TRAIN_RELATIVE" ]]
}

resolve_invocation_dir() {
  local script_ref="${BASH_SOURCE[0]-}"
  if [[ -n "$script_ref" && "$script_ref" != "bash" ]]; then
    cd "$(dirname "$script_ref")" && pwd
    return
  fi

  local argv0="${0-}"
  if [[ -n "$argv0" && "$argv0" != "bash" && "$argv0" != "-bash" ]]; then
    cd "$(dirname "$argv0")" && pwd
    return
  fi

  pwd
}

resolve_repo_root() {
  local candidate="$1"
  while [[ "$candidate" != "/" ]]; do
    if has_repo_layout "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
    candidate="$(dirname "$candidate")"
  done
  return 1
}

INVOCATION_DIR="$(resolve_invocation_dir)"
if ! REPO_ROOT="$(resolve_repo_root "$INVOCATION_DIR")"; then
  echo "Could not resolve repository root from $INVOCATION_DIR" >&2
  exit 1
fi

cd "$REPO_ROOT"

if [[ "${MPLBACKEND:-}" == "module://matplotlib_inline.backend_inline" || -z "${MPLBACKEND:-}" ]]; then
  export MPLBACKEND="Agg"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

DRY_RUN="${DRY_RUN:-0}"
SHOW_GPU_INFO="${SHOW_GPU_INFO:-1}"
QUIET="${QUIET:-0}"
LOG_FILE="${LOG_FILE:-}"

PPO_ROOT="$REPO_ROOT/source/algorithm/PPO-based/PPO-traditional"
TRAIN_SCRIPT="$PPO_ROOT/PPOTraditional/scripts/train.py"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/ppo_traditional_highway_local}"
CONFIG_PATH="${CONFIG_PATH:-$OUTPUT_DIR/highway_ppo_traditional_local.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$OUTPUT_DIR/ppo_traditional_highway_local.pt}"
RESUME_FROM="${RESUME_FROM:-$CHECKPOINT_PATH}"

UPDATES="${UPDATES:-30}"
N_ENVS="${N_ENVS:-16}"
STEPS_PER_ENV="${STEPS_PER_ENV:-}"
DEVICE="${DEVICE:-auto}"
SEED_START="${SEED_START:-100}"
GPU_INDICES="${GPU_INDICES:-}"

export PPO_TRADITIONAL_SCENARIO="${PPO_TRADITIONAL_SCENARIO:-highway_ppo_traditional}"
export PPO_TRADITIONAL_CONFIG_PATH="$CONFIG_PATH"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/source:$REPO_ROOT/source/highway-env:$PPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "$GPU_INDICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_INDICES"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[resume] config_path không tồn tại: $CONFIG_PATH" >&2
  echo "[resume] chỉ rõ CONFIG_PATH hoặc chạy run_ppo_traditional_train_local.sh trước." >&2
  exit 1
fi

if [[ ! -f "$RESUME_FROM" ]]; then
  echo "[resume] resume_from không tồn tại: $RESUME_FROM" >&2
  exit 1
fi

cmd=(
  "$PYTHON_BIN"
  "$TRAIN_SCRIPT"
  --config-path "$CONFIG_PATH"
  --updates "$UPDATES"
  --n-envs "$N_ENVS"
  --device "$DEVICE"
  --seed-start "$SEED_START"
  --save-path "$CHECKPOINT_PATH"
  --resume-from "$RESUME_FROM"
)

if [[ -n "$STEPS_PER_ENV" ]]; then
  cmd+=(--steps-per-env "$STEPS_PER_ENV")
fi

if [[ "$QUIET" == "1" || "${QUIET,,}" == "true" ]]; then
  cmd+=(--quiet)
fi

cmd+=("$@")

if [[ "$SHOW_GPU_INFO" == "1" && -x "$(command -v nvidia-smi || true)" ]]; then
  echo "[gpu] visible devices: ${CUDA_VISIBLE_DEVICES:-all}" >&2
  nvidia-smi -L >&2 || true
fi

printf '[run-ppo-traditional-resume-local] scenario=%s repo_root=%s config=%s output_dir=%s resume_from=%s updates=%s n_envs=%s device=%s checkpoint=%s\n' \
  "$PPO_TRADITIONAL_SCENARIO" \
  "$REPO_ROOT" \
  "$CONFIG_PATH" \
  "$OUTPUT_DIR" \
  "$RESUME_FROM" \
  "$UPDATES" \
  "$N_ENVS" \
  "$DEVICE" \
  "$CHECKPOINT_PATH"

if [[ -n "$STEPS_PER_ENV" ]]; then
  printf 'steps_per_env=%s\n' "$STEPS_PER_ENV"
fi

if [[ "$DRY_RUN" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  printf '[dry-run] '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

if [[ -n "$LOG_FILE" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
else
  "${cmd[@]}"
fi
