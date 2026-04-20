#!/usr/bin/env bash
set -euo pipefail

EXPECTED_EVAL_RELATIVE="source/algorithm/PPO-based/PPO-traditional/PPOTraditional/scripts/evaluate.py"

has_repo_layout() {
  local root="$1"
  [[ -f "$root/$EXPECTED_EVAL_RELATIVE" ]]
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

resolve_checkpoint_path() {
  if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
    if [[ -f "$CHECKPOINT_PATH" ]]; then
      printf '%s\n' "$(cd "$(dirname "$CHECKPOINT_PATH")" && pwd)/$(basename "$CHECKPOINT_PATH")"
      return 0
    fi
    echo "Checkpoint file does not exist: $CHECKPOINT_PATH" >&2
    exit 1
  fi

  local candidates=(
    "$REPO_ROOT/outputs/ppo_traditional_highway_local/ppo_traditional_highway_local.pt"
    "$REPO_ROOT/outputs/ppo_traditional_highway/ppo_traditional_highway.pt"
    "$REPO_ROOT/ppo_traditional_highway_local.pt"
    "$REPO_ROOT/ppo_traditional_highway.pt"
    "$PWD/ppo_traditional_highway_local.pt"
    "$PWD/ppo_traditional_highway.pt"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  local discovered
  discovered="$(find "$REPO_ROOT" -maxdepth 5 -type f \( -name 'ppo_traditional_highway_local.pt' -o -name 'ppo_traditional_highway.pt' \) | head -n 1 || true)"
  if [[ -n "$discovered" ]]; then
    printf '%s\n' "$discovered"
    return 0
  fi

  echo "Could not find a PPOTraditional checkpoint. Set CHECKPOINT_PATH=/abs/path/to/ppo_traditional_highway_local.pt" >&2
  exit 1
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

INSTALL_DEPS="${INSTALL_DEPS:-auto}"
INSTALL_LOCAL_EDITABLE="${INSTALL_LOCAL_EDITABLE:-0}"
UPGRADE_PIP="${UPGRADE_PIP:-0}"
DRY_RUN="${DRY_RUN:-0}"
SHOW_GPU_INFO="${SHOW_GPU_INFO:-1}"
QUIET="${QUIET:-0}"
LOG_FILE="${LOG_FILE:-}"

PPO_ROOT="$REPO_ROOT/source/algorithm/PPO-based/PPO-traditional"
EVAL_SCRIPT="$PPO_ROOT/PPOTraditional/scripts/evaluate.py"
BASE_CONFIG_PATH="${BASE_CONFIG_PATH:-$PPO_ROOT/configs/highway_ppo_traditional.yaml}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/ppo_traditional_infer_local}"
GENERATED_CONFIG_PATH="${GENERATED_CONFIG_PATH:-$OUTPUT_DIR/highway_ppo_traditional_eval_from_checkpoint.json}"

POLICY_SOURCE="${POLICY_SOURCE:-best}"
EPISODES="${EPISODES:-5}"
SEED_START="${SEED_START:-10}"
RENDER_MODE="${RENDER_MODE:-human}"
DEVICE="${DEVICE:-auto}"
GPU_INDICES="${GPU_INDICES:-}"

export PPO_TRADITIONAL_SCENARIO="${PPO_TRADITIONAL_SCENARIO:-highway_ppo_traditional}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/source:$REPO_ROOT/source/highway-env:$PPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "$GPU_INDICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_INDICES"
fi

ensure_python_module() {
  local module_name="$1"
  local package_spec="$2"
  if ! "$PYTHON_BIN" -c "import ${module_name}" >/dev/null 2>&1; then
    echo "[install] missing module=${module_name}, installing ${package_spec}" >&2
    "$PYTHON_BIN" -m pip install -q "$package_spec"
  fi
}

install_local_editable() {
  local package_path="$1"
  echo "[install] editable ${package_path}" >&2
  "$PYTHON_BIN" -m pip install -q --no-deps -e "$package_path"
}

is_truthy() {
  local raw="${1:-}"
  case "${raw,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ "${INSTALL_DEPS,,}" == "auto" || "${INSTALL_DEPS,,}" == "1" ]] || is_truthy "$INSTALL_DEPS"; then
  if [[ "$UPGRADE_PIP" == "1" ]]; then
    echo "[install] upgrading pip/setuptools/wheel" >&2
    "$PYTHON_BIN" -m pip install -q --upgrade pip setuptools wheel
  fi

  ensure_python_module gymnasium "gymnasium"
  ensure_python_module yaml "PyYAML"
  ensure_python_module numpy "numpy"
  ensure_python_module torch "torch"
fi

if is_truthy "$INSTALL_LOCAL_EDITABLE"; then
  install_local_editable "$REPO_ROOT/source/highway-env"
  install_local_editable "$PPO_ROOT"
fi

mkdir -p "$OUTPUT_DIR"

RESOLVED_CHECKPOINT_PATH="$(resolve_checkpoint_path)"

if [[ "$DRY_RUN" != "1" && "${DRY_RUN,,}" != "true" ]]; then
  CHECKPOINT_PATH="$RESOLVED_CHECKPOINT_PATH" \
  BASE_CONFIG_PATH="$BASE_CONFIG_PATH" \
  GENERATED_CONFIG_PATH="$GENERATED_CONFIG_PATH" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

import torch


checkpoint_path = Path(os.environ["CHECKPOINT_PATH"]).expanduser().resolve()
generated_config_path = Path(os.environ["GENERATED_CONFIG_PATH"]).expanduser().resolve()
base_config_path = Path(os.environ["BASE_CONFIG_PATH"]).expanduser().resolve()

checkpoint = torch.load(checkpoint_path, map_location="cpu")
config_snapshot = checkpoint.get("config_snapshot")

if isinstance(config_snapshot, dict):
    generated_config_path.write_text(
        json.dumps(config_snapshot, indent=2),
        encoding="utf-8",
    )
else:
    generated_config_path.write_text(
        base_config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
PY
fi

export PPO_TRADITIONAL_CONFIG_PATH="$GENERATED_CONFIG_PATH"

cmd=(
  "$PYTHON_BIN"
  "$EVAL_SCRIPT"
  --config-path "$GENERATED_CONFIG_PATH"
  --checkpoint-path "$RESOLVED_CHECKPOINT_PATH"
  --policy-source "$POLICY_SOURCE"
  --episodes "$EPISODES"
  --seed-start "$SEED_START"
  --render-mode "$RENDER_MODE"
  --device "$DEVICE"
)

if [[ "$QUIET" == "1" || "${QUIET,,}" == "true" ]]; then
  cmd+=(--quiet)
fi

cmd+=("$@")

if [[ "$SHOW_GPU_INFO" == "1" && -x "$(command -v nvidia-smi || true)" ]]; then
  echo "[gpu] visible devices: ${CUDA_VISIBLE_DEVICES:-all}" >&2
  nvidia-smi -L >&2 || true
fi

printf '[run-ppo-traditional-infer-local] repo_root=%s checkpoint=%s config=%s policy_source=%s episodes=%s seed_start=%s render_mode=%s device=%s\n' \
  "$REPO_ROOT" \
  "$RESOLVED_CHECKPOINT_PATH" \
  "$GENERATED_CONFIG_PATH" \
  "$POLICY_SOURCE" \
  "$EPISODES" \
  "$SEED_START" \
  "$RENDER_MODE" \
  "$DEVICE"

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
  exec "${cmd[@]}"
fi
