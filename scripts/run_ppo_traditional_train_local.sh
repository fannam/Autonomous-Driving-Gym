#!/usr/bin/env bash
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

INSTALL_DEPS="${INSTALL_DEPS:-auto}"
INSTALL_LOCAL_EDITABLE="${INSTALL_LOCAL_EDITABLE:-0}"
UPGRADE_PIP="${UPGRADE_PIP:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_EVALUATION_AFTER_TRAIN="${RUN_EVALUATION_AFTER_TRAIN:-0}"
SHOW_GPU_INFO="${SHOW_GPU_INFO:-1}"
QUIET="${QUIET:-0}"
LOG_FILE="${LOG_FILE:-}"

PPO_ROOT="$REPO_ROOT/source/algorithm/PPO-based/PPO-traditional"
TRAIN_SCRIPT="$PPO_ROOT/PPOTraditional/scripts/train.py"
EVAL_SCRIPT="$PPO_ROOT/PPOTraditional/scripts/evaluate.py"
BASE_CONFIG_PATH="${BASE_CONFIG_PATH:-$PPO_ROOT/configs/highway_ppo_traditional.yaml}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/ppo_traditional_highway_local}"
GENERATED_CONFIG_PATH="${GENERATED_CONFIG_PATH:-$OUTPUT_DIR/highway_ppo_traditional_local.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$OUTPUT_DIR/ppo_traditional_highway_local.pt}"
METRICS_PATH="${METRICS_PATH:-$OUTPUT_DIR/ppo_traditional_metrics.jsonl}"
EVAL_METRICS_PATH="${EVAL_METRICS_PATH:-$OUTPUT_DIR/ppo_traditional_eval_metrics.jsonl}"

UPDATES="${UPDATES:-30}"
N_ENVS="${N_ENVS:-16}"
ENV_CAP="${ENV_CAP:-16}"
CPU_RESERVE="${CPU_RESERVE:-2}"
STEPS_PER_ENV="${STEPS_PER_ENV:-80}"
DEVICE="${DEVICE:-auto}"
SEED_START="${SEED_START:-21}"
POLICY_SOURCE="${POLICY_SOURCE:-best}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
RENDER_MODE="${RENDER_MODE:-}"
GPU_INDICES="${GPU_INDICES:-}"

MAX_STEPS="${MAX_STEPS:-}"
DURATION="${DURATION:-}"
VEHICLES_COUNT="${VEHICLES_COUNT:-}"
VEHICLES_DENSITY="${VEHICLES_DENSITY:-}"
LANES_COUNT="${LANES_COUNT:-}"
ROAD_SPEED_LIMIT="${ROAD_SPEED_LIMIT:-}"
POLICY_FREQUENCY="${POLICY_FREQUENCY:-}"
SIMULATION_FREQUENCY="${SIMULATION_FREQUENCY:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
PPO_EPOCHS="${PPO_EPOCHS:-}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-}"
TARGET_KL="${TARGET_KL:-}"
ENTROPY_COEF="${ENTROPY_COEF:-}"
VALUE_COEF="${VALUE_COEF:-}"
CLIP_EPSILON="${CLIP_EPSILON:-}"
GAE_LAMBDA="${GAE_LAMBDA:-}"
GAMMA="${GAMMA:-}"

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

resolve_cpu_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi

  "$PYTHON_BIN" - <<'PY'
import os
print(max(1, int(os.cpu_count() or 1)))
PY
}

resolve_effective_env_count() {
  if [[ "${N_ENVS,,}" != "auto" ]]; then
    printf '%s\n' "$N_ENVS"
    return
  fi

  local cpu_count
  cpu_count="$(resolve_cpu_count)"
  local reserve="$CPU_RESERVE"
  local env_cap="$ENV_CAP"

  local effective=$(( cpu_count - reserve ))
  if (( effective < 1 )); then
    effective=1
  fi
  if (( env_cap > 0 && effective > env_cap )); then
    effective="$env_cap"
  fi
  printf '%s\n' "$effective"
}

CPU_COUNT="$(resolve_cpu_count)"
EFFECTIVE_N_ENVS="$(resolve_effective_env_count)"

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

BASE_CONFIG_PATH="$BASE_CONFIG_PATH" \
GENERATED_CONFIG_PATH="$GENERATED_CONFIG_PATH" \
CHECKPOINT_PATH="$CHECKPOINT_PATH" \
METRICS_PATH="$METRICS_PATH" \
EVAL_METRICS_PATH="$EVAL_METRICS_PATH" \
N_ENVS="$EFFECTIVE_N_ENVS" \
STEPS_PER_ENV="$STEPS_PER_ENV" \
MAX_STEPS="$MAX_STEPS" \
DURATION="$DURATION" \
VEHICLES_COUNT="$VEHICLES_COUNT" \
VEHICLES_DENSITY="$VEHICLES_DENSITY" \
LANES_COUNT="$LANES_COUNT" \
ROAD_SPEED_LIMIT="$ROAD_SPEED_LIMIT" \
POLICY_FREQUENCY="$POLICY_FREQUENCY" \
SIMULATION_FREQUENCY="$SIMULATION_FREQUENCY" \
LEARNING_RATE="$LEARNING_RATE" \
PPO_EPOCHS="$PPO_EPOCHS" \
MINIBATCH_SIZE="$MINIBATCH_SIZE" \
TARGET_KL="$TARGET_KL" \
ENTROPY_COEF="$ENTROPY_COEF" \
VALUE_COEF="$VALUE_COEF" \
CLIP_EPSILON="$CLIP_EPSILON" \
GAE_LAMBDA="$GAE_LAMBDA" \
GAMMA="$GAMMA" \
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
ppo_config = train_preset["ppo"]
evaluation_ppo_config = evaluation_preset["ppo"]

train_preset["logging"]["metrics_path"] = os.environ["METRICS_PATH"]
train_preset["model_path"] = os.environ["CHECKPOINT_PATH"]
evaluation_preset["logging"]["metrics_path"] = os.environ["EVAL_METRICS_PATH"]
evaluation_preset["model_path"] = os.environ["CHECKPOINT_PATH"]
train_preset["rollout"]["n_envs"] = int(os.environ["N_ENVS"])
evaluation_preset["rollout"]["n_envs"] = 1

steps_per_env = parse_optional_int(os.environ["STEPS_PER_ENV"])
if steps_per_env is not None:
    train_preset["rollout"]["steps_per_env"] = steps_per_env
    evaluation_preset["rollout"]["steps_per_env"] = steps_per_env

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

for ppo_cfg in (ppo_config, evaluation_ppo_config):
    maybe_set(ppo_cfg, "learning_rate", parse_optional_float(os.environ["LEARNING_RATE"]))
    maybe_set(ppo_cfg, "ppo_epochs", parse_optional_int(os.environ["PPO_EPOCHS"]))
    maybe_set(ppo_cfg, "minibatch_size", parse_optional_int(os.environ["MINIBATCH_SIZE"]))
    maybe_set(ppo_cfg, "target_kl", parse_optional_float(os.environ["TARGET_KL"]))
    maybe_set(ppo_cfg, "entropy_coef", parse_optional_float(os.environ["ENTROPY_COEF"]))
    maybe_set(ppo_cfg, "value_coef", parse_optional_float(os.environ["VALUE_COEF"]))
    maybe_set(ppo_cfg, "clip_epsilon", parse_optional_float(os.environ["CLIP_EPSILON"]))
    maybe_set(ppo_cfg, "gae_lambda", parse_optional_float(os.environ["GAE_LAMBDA"]))
    maybe_set(ppo_cfg, "gamma", parse_optional_float(os.environ["GAMMA"]))

generated_config_path = Path(os.environ["GENERATED_CONFIG_PATH"])
generated_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
PY

export PPO_TRADITIONAL_CONFIG_PATH="$GENERATED_CONFIG_PATH"

cmd=(
  "$PYTHON_BIN"
  "$TRAIN_SCRIPT"
  --config-path "$GENERATED_CONFIG_PATH"
  --updates "$UPDATES"
  --n-envs "$EFFECTIVE_N_ENVS"
  --device "$DEVICE"
  --seed-start "$SEED_START"
  --save-path "$CHECKPOINT_PATH"
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

printf '[run-ppo-traditional-train-local] scenario=%s repo_root=%s base_config=%s generated_config=%s output_dir=%s updates=%s n_envs=%s device=%s checkpoint=%s metrics=%s note=%s\n' \
  "$PPO_TRADITIONAL_SCENARIO" \
  "$REPO_ROOT" \
  "$BASE_CONFIG_PATH" \
  "$GENERATED_CONFIG_PATH" \
  "$OUTPUT_DIR" \
  "$UPDATES" \
  "$EFFECTIVE_N_ENVS" \
  "$DEVICE" \
  "$CHECKPOINT_PATH" \
  "$METRICS_PATH" \
  "single-gpu-local-training"

printf '[runtime] cpu_count=%s requested_n_envs=%s effective_n_envs=%s env_cap=%s cpu_reserve=%s install_deps=%s install_local_editable=%s\n' \
  "$CPU_COUNT" \
  "$N_ENVS" \
  "$EFFECTIVE_N_ENVS" \
  "$ENV_CAP" \
  "$CPU_RESERVE" \
  "$INSTALL_DEPS" \
  "$INSTALL_LOCAL_EDITABLE"

if [[ -n "$STEPS_PER_ENV" ]]; then
  printf 'steps_per_env=%s\n' "$STEPS_PER_ENV"
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
