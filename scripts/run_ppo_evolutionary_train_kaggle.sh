#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPECTED_TRAIN_RELATIVE="source/algorithm/PPO-based/PPO-evolutionary-algorithm/PPOEvolutionary/scripts/train.py"
REPO_URL="${REPO_URL:-https://github.com/fannam/Autonomous-Driving-Gym.git}"
AUTO_CLONE_REPO="${AUTO_CLONE_REPO:-1}"
FORCE_CLONE="${FORCE_CLONE:-0}"
UPDATE_CLONE="${UPDATE_CLONE:-0}"
CLONE_ROOT="${CLONE_ROOT:-/kaggle/working}"
REPO_DIR_NAME="${REPO_DIR_NAME:-Autonomous-Driving-Gym}"
CLONE_REF="${CLONE_REF:-}"
GIT_DEPTH="${GIT_DEPTH:-1}"

has_repo_layout() {
  local root="$1"
  [[ -f "$root/$EXPECTED_TRAIN_RELATIVE" ]]
}

ensure_git_checkout() {
  local target_root="$1"

  if ! command -v git >/dev/null 2>&1; then
    echo "git is required to clone the repository." >&2
    exit 1
  fi

  mkdir -p "$(dirname "$target_root")"

  if [[ ! -d "$target_root/.git" ]]; then
    echo "[repo] cloning $REPO_URL into $target_root" >&2
    git clone --depth "$GIT_DEPTH" "$REPO_URL" "$target_root"
  elif [[ "$UPDATE_CLONE" == "1" || "${UPDATE_CLONE,,}" == "true" ]]; then
    echo "[repo] updating existing checkout at $target_root" >&2
    git -C "$target_root" fetch --depth "$GIT_DEPTH" origin
    current_branch="$(git -C "$target_root" rev-parse --abbrev-ref HEAD)"
    if [[ "$current_branch" != "HEAD" ]]; then
      git -C "$target_root" pull --ff-only --depth "$GIT_DEPTH" origin "$current_branch"
    fi
  else
    echo "[repo] reusing existing checkout at $target_root" >&2
  fi

  if [[ -n "$CLONE_REF" ]]; then
    echo "[repo] checking out ref $CLONE_REF" >&2
    git -C "$target_root" fetch --depth "$GIT_DEPTH" origin "$CLONE_REF"
    git -C "$target_root" checkout --detach FETCH_HEAD
  fi
}

CLONE_REPO_ROOT="${CLONE_ROOT%/}/$REPO_DIR_NAME"

if [[ "$AUTO_CLONE_REPO" == "1" || "${AUTO_CLONE_REPO,,}" == "true" ]]; then
  if [[ "$FORCE_CLONE" == "1" || "${FORCE_CLONE,,}" == "true" ]]; then
    ensure_git_checkout "$CLONE_REPO_ROOT"
    REPO_ROOT="$CLONE_REPO_ROOT"
  elif has_repo_layout "$LOCAL_REPO_ROOT"; then
    REPO_ROOT="$LOCAL_REPO_ROOT"
  else
    ensure_git_checkout "$CLONE_REPO_ROOT"
    REPO_ROOT="$CLONE_REPO_ROOT"
  fi
else
  REPO_ROOT="$LOCAL_REPO_ROOT"
fi

if ! has_repo_layout "$REPO_ROOT"; then
  echo "Could not resolve repository layout under $REPO_ROOT" >&2
  exit 1
fi

cd "$REPO_ROOT"

# Kaggle notebooks often export an inline matplotlib backend that breaks in subprocesses.
if [[ "${MPLBACKEND:-}" == "module://matplotlib_inline.backend_inline" || -z "${MPLBACKEND:-}" ]]; then
  export MPLBACKEND="Agg"
fi

# Avoid CPU oversubscription when rollout workers spawn subprocesses.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi
INSTALL_DEPS="${INSTALL_DEPS:-1}"
UPGRADE_PIP="${UPGRADE_PIP:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_EVALUATION_AFTER_TRAIN="${RUN_EVALUATION_AFTER_TRAIN:-0}"
SHOW_GPU_INFO="${SHOW_GPU_INFO:-1}"
QUIET="${QUIET:-0}"
LOG_FILE="${LOG_FILE:-}"

PPO_ROOT="$REPO_ROOT/source/algorithm/PPO-based/PPO-evolutionary-algorithm"
TRAIN_SCRIPT="$PPO_ROOT/PPOEvolutionary/scripts/train.py"
EVAL_SCRIPT="$PPO_ROOT/PPOEvolutionary/scripts/evaluate.py"
BASE_CONFIG_PATH="${BASE_CONFIG_PATH:-$PPO_ROOT/configs/highway_ppo_evolutionary.yaml}"

OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/ppo_evolutionary_highway}"
GENERATED_CONFIG_PATH="${GENERATED_CONFIG_PATH:-$OUTPUT_DIR/highway_ppo_evolutionary_kaggle.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$OUTPUT_DIR/ppo_evolutionary_highway.pt}"
METRICS_PATH="${METRICS_PATH:-$OUTPUT_DIR/ppo_evolutionary_metrics.jsonl}"
EVAL_METRICS_PATH="${EVAL_METRICS_PATH:-$OUTPUT_DIR/ppo_evolutionary_eval_metrics.jsonl}"

GENERATIONS="${GENERATIONS:-20}"
POPULATION_SIZE="${POPULATION_SIZE:-8}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-cuda:0}"
SEED_START="${SEED_START:-21}"
POLICY_SOURCE="${POLICY_SOURCE:-best}"
EVAL_EPISODES="${EVAL_EPISODES:-3}"
RENDER_MODE="${RENDER_MODE:-}"
GPU_INDICES="${GPU_INDICES:-}"

EPISODES_PER_POLICY="${EPISODES_PER_POLICY:-}"
MAX_STEPS="${MAX_STEPS:-}"
DURATION="${DURATION:-}"
VEHICLES_COUNT="${VEHICLES_COUNT:-}"
VEHICLES_DENSITY="${VEHICLES_DENSITY:-}"
LANES_COUNT="${LANES_COUNT:-}"
ROAD_SPEED_LIMIT="${ROAD_SPEED_LIMIT:-}"
POLICY_FREQUENCY="${POLICY_FREQUENCY:-}"
SIMULATION_FREQUENCY="${SIMULATION_FREQUENCY:-}"

export PPO_EVOLUTIONARY_SCENARIO="${PPO_EVOLUTIONARY_SCENARIO:-highway_ppo_evolutionary}"
export PYTHONPATH="$REPO_ROOT/source:$REPO_ROOT/source/highway-env:$PPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

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

if [[ "$INSTALL_DEPS" == "1" ]]; then
  if [[ "$UPGRADE_PIP" == "1" ]]; then
    echo "[install] upgrading pip/setuptools/wheel" >&2
    "$PYTHON_BIN" -m pip install -q --upgrade pip setuptools wheel
  fi

  ensure_python_module gymnasium "gymnasium"
  ensure_python_module yaml "PyYAML"
  ensure_python_module numpy "numpy"
  ensure_python_module torch "torch"

  install_local_editable "$REPO_ROOT/source/highway-env"
  install_local_editable "$PPO_ROOT"
fi

mkdir -p "$OUTPUT_DIR"

BASE_CONFIG_PATH="$BASE_CONFIG_PATH" \
GENERATED_CONFIG_PATH="$GENERATED_CONFIG_PATH" \
CHECKPOINT_PATH="$CHECKPOINT_PATH" \
METRICS_PATH="$METRICS_PATH" \
EVAL_METRICS_PATH="$EVAL_METRICS_PATH" \
WORKERS="$WORKERS" \
EPISODES_PER_POLICY="$EPISODES_PER_POLICY" \
MAX_STEPS="$MAX_STEPS" \
DURATION="$DURATION" \
VEHICLES_COUNT="$VEHICLES_COUNT" \
VEHICLES_DENSITY="$VEHICLES_DENSITY" \
LANES_COUNT="$LANES_COUNT" \
ROAD_SPEED_LIMIT="$ROAD_SPEED_LIMIT" \
POLICY_FREQUENCY="$POLICY_FREQUENCY" \
SIMULATION_FREQUENCY="$SIMULATION_FREQUENCY" \
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path


def parse_optional_int(raw: str):
    value = str(raw).strip()
    if not value:
        return None
    if value.lower() == "null":
        return None
    return int(value)


def parse_optional_float(raw: str):
    value = str(raw).strip()
    if not value:
        return None
    if value.lower() == "null":
        return None
    return float(value)


def maybe_set(container, key, value):
    if value is not None:
        container[key] = value


config = json.loads(Path(os.environ["BASE_CONFIG_PATH"]).read_text(encoding="utf-8"))

train_preset = config["presets"]["train"]
evaluation_preset = config["presets"]["evaluation"]
environment_config = config["environment"]["config"]

train_preset["logging"]["metrics_path"] = os.environ["METRICS_PATH"]
train_preset["model_path"] = os.environ["CHECKPOINT_PATH"]
evaluation_preset["logging"]["metrics_path"] = os.environ["EVAL_METRICS_PATH"]
evaluation_preset["model_path"] = os.environ["CHECKPOINT_PATH"]
train_preset["rollout"]["workers"] = int(os.environ["WORKERS"])
evaluation_preset["rollout"]["workers"] = 1

episodes_per_policy = parse_optional_int(os.environ["EPISODES_PER_POLICY"])
if episodes_per_policy is not None:
    train_preset["rollout"]["episodes_per_policy"] = episodes_per_policy
    evaluation_preset["rollout"]["episodes_per_policy"] = 1

max_steps = parse_optional_int(os.environ["MAX_STEPS"])
train_preset["rollout"]["max_steps"] = max_steps
evaluation_preset["rollout"]["max_steps"] = max_steps

maybe_set(environment_config, "duration", parse_optional_int(os.environ["DURATION"]))
maybe_set(environment_config, "vehicles_count", parse_optional_int(os.environ["VEHICLES_COUNT"]))
maybe_set(
    environment_config,
    "vehicles_density",
    parse_optional_float(os.environ["VEHICLES_DENSITY"]),
)
maybe_set(environment_config, "lanes_count", parse_optional_int(os.environ["LANES_COUNT"]))
maybe_set(
    environment_config,
    "road_speed_limit",
    parse_optional_int(os.environ["ROAD_SPEED_LIMIT"]),
)
maybe_set(
    environment_config,
    "policy_frequency",
    parse_optional_int(os.environ["POLICY_FREQUENCY"]),
)
maybe_set(
    environment_config,
    "simulation_frequency",
    parse_optional_int(os.environ["SIMULATION_FREQUENCY"]),
)

generated_config_path = Path(os.environ["GENERATED_CONFIG_PATH"])
generated_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
PY

export PPO_EVOLUTIONARY_CONFIG_PATH="$GENERATED_CONFIG_PATH"

cmd=(
  "$PYTHON_BIN"
  "$TRAIN_SCRIPT"
  --config-path "$GENERATED_CONFIG_PATH"
  --generations "$GENERATIONS"
  --population-size "$POPULATION_SIZE"
  --workers "$WORKERS"
  --device "$DEVICE"
  --seed-start "$SEED_START"
  --save-path "$CHECKPOINT_PATH"
)

if [[ "$QUIET" == "1" || "${QUIET,,}" == "true" ]]; then
  cmd+=(--quiet)
fi

cmd+=("$@")

if [[ "$SHOW_GPU_INFO" == "1" && -x "$(command -v nvidia-smi || true)" ]]; then
  echo "[gpu] visible devices: ${CUDA_VISIBLE_DEVICES:-all}" >&2
  nvidia-smi -L >&2 || true
fi

printf '[run-ppo-evolutionary-train-kaggle] scenario=%s base_config=%s generated_config=%s output_dir=%s generations=%s population=%s workers=%s device=%s checkpoint=%s metrics=%s note=%s\n' \
  "$PPO_EVOLUTIONARY_SCENARIO" \
  "$BASE_CONFIG_PATH" \
  "$GENERATED_CONFIG_PATH" \
  "$OUTPUT_DIR" \
  "$GENERATIONS" \
  "$POPULATION_SIZE" \
  "$WORKERS" \
  "$DEVICE" \
  "$CHECKPOINT_PATH" \
  "$METRICS_PATH" \
  "current_trainer_uses_single_learner_gpu_only"

printf '[repo] root=%s auto_clone=%s force_clone=%s clone_ref=%s\n' \
  "$REPO_ROOT" \
  "$AUTO_CLONE_REPO" \
  "$FORCE_CLONE" \
  "${CLONE_REF:-default}"

if [[ -n "$EPISODES_PER_POLICY" ]]; then
  printf 'episodes_per_policy=%s\n' "$EPISODES_PER_POLICY"
fi

if [[ -n "$MAX_STEPS" ]]; then
  printf 'max_steps=%s\n' "$MAX_STEPS"
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

if [[ "$RUN_EVALUATION_AFTER_TRAIN" == "1" || "${RUN_EVALUATION_AFTER_TRAIN,,}" == "true" ]]; then
  eval_cmd=(
    "$PYTHON_BIN"
    "$EVAL_SCRIPT"
    --config-path "$GENERATED_CONFIG_PATH"
    --checkpoint-path "$CHECKPOINT_PATH"
    --policy-source "$POLICY_SOURCE"
    --episodes "$EVAL_EPISODES"
    --seed-start "$SEED_START"
    --device "$DEVICE"
  )
  if [[ -n "$RENDER_MODE" ]]; then
    eval_cmd+=(--render-mode "$RENDER_MODE")
  fi
  if [[ "$QUIET" == "1" || "${QUIET,,}" == "true" ]]; then
    eval_cmd+=(--quiet)
  fi

  printf '[post-train-eval] '
  printf '%q ' "${eval_cmd[@]}"
  printf '\n'
  exec "${eval_cmd[@]}"
fi
