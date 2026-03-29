#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Kaggle notebooks often export an inline matplotlib backend that breaks in subprocesses.
if [[ "${MPLBACKEND:-}" == "module://matplotlib_inline.backend_inline" || -z "${MPLBACKEND:-}" ]]; then
  export MPLBACKEND="Agg"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
UPGRADE_PIP="${UPGRADE_PIP:-0}"
LOG_FILE="${LOG_FILE:-}"

export ALPHAZERO_ADVERSARIAL_SCENARIO="${ALPHAZERO_ADVERSARIAL_SCENARIO:-racetrack_adversarial}"
export ALPHAZERO_ADVERSARIAL_CONFIG_PATH="${ALPHAZERO_ADVERSARIAL_CONFIG_PATH:-$REPO_ROOT/AlphaZero-adversarial-autonomous-driving/configs/racetrack_adversarial.yaml}"
export PYTHONPATH="$REPO_ROOT/highway-env:$REPO_ROOT/AlphaZero-adversarial-autonomous-driving${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/alphazero_adversarial_self_play}"
WORKERS="${WORKERS:-2}"
TOTAL_EPISODES="${TOTAL_EPISODES:-8}"
EPISODES_PER_WORKER="${EPISODES_PER_WORKER:-}"
EPISODES_PER_SHARD="${EPISODES_PER_SHARD:-2}"
DEVICE="${DEVICE:-cuda}"
NUM_GPUS="${NUM_GPUS:-2}"
GPU_INDICES="${GPU_INDICES:-0,1}"
TORCH_THREADS_PER_WORKER="${TORCH_THREADS_PER_WORKER:-1}"
SELF_PLAY_SEED="${SELF_PLAY_SEED:-1000}"
NETWORK_SEED="${NETWORK_SEED:-42}"
MODEL_PATH="${MODEL_PATH:-}"
N_SIMULATIONS="${N_SIMULATIONS:-24}"
C_PUCT="${C_PUCT:-}"
TEMPERATURE="${TEMPERATURE:-}"
TEMPERATURE_DROP_STEP="${TEMPERATURE_DROP_STEP:-}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-}"
ROOT_EXPLORATION_FRACTION="${ROOT_EXPLORATION_FRACTION:-}"
MAX_EXPAND_ACTIONS_PER_AGENT="${MAX_EXPAND_ACTIONS_PER_AGENT:-}"
MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-10}"
RESULT_TIMEOUT="${RESULT_TIMEOUT:-}"
ENV_ID="${ENV_ID:-}"
DURATION="${DURATION:-}"
OTHER_VEHICLES="${OTHER_VEHICLES:-}"
POLICY_FREQUENCY="${POLICY_FREQUENCY:-}"
SIMULATION_FREQUENCY="${SIMULATION_FREQUENCY:-}"

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

  ensure_python_module gymnasium "gymnasium>=1.2.3"
  ensure_python_module yaml "PyYAML>=6.0.2"
  ensure_python_module pygame "pygame>=2.6.0"
  ensure_python_module numpy "numpy>=1.26.0"
  ensure_python_module torch "torch"

  install_local_editable "$REPO_ROOT/highway-env"
  install_local_editable "$REPO_ROOT/AlphaZero-adversarial-autonomous-driving"
fi

cmd=(
  "$PYTHON_BIN"
  AlphaZero-adversarial-autonomous-driving/AlphaZeroAdversarial/scripts/self_play_kaggle_dual_gpu.py
  --workers "$WORKERS"
  --device "$DEVICE"
  --episodes-per-shard "$EPISODES_PER_SHARD"
  --torch-threads-per-worker "$TORCH_THREADS_PER_WORKER"
  --self-play-seed "$SELF_PLAY_SEED"
  --network-seed "$NETWORK_SEED"
  --n-simulations "$N_SIMULATIONS"
  --progress-interval "$PROGRESS_INTERVAL"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$MODEL_PATH" ]]; then
  cmd+=(--model-path "$MODEL_PATH")
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
  EPISODES_PER_WORKER="${EPISODES_PER_WORKER:-2}"
  cmd+=(--episodes-per-worker "$EPISODES_PER_WORKER")
  episode_mode="episodes_per_worker=$EPISODES_PER_WORKER"
fi

if [[ -n "$C_PUCT" ]]; then
  cmd+=(--c-puct "$C_PUCT")
fi

if [[ -n "$TEMPERATURE" ]]; then
  cmd+=(--temperature "$TEMPERATURE")
fi

if [[ -n "$TEMPERATURE_DROP_STEP" ]]; then
  cmd+=(--temperature-drop-step "$TEMPERATURE_DROP_STEP")
fi

if [[ -n "$DIRICHLET_ALPHA" ]]; then
  cmd+=(--dirichlet-alpha "$DIRICHLET_ALPHA")
fi

if [[ -n "$ROOT_EXPLORATION_FRACTION" ]]; then
  cmd+=(--root-exploration-fraction "$ROOT_EXPLORATION_FRACTION")
fi

if [[ -n "$MAX_EXPAND_ACTIONS_PER_AGENT" ]]; then
  cmd+=(--max-expand-actions-per-agent "$MAX_EXPAND_ACTIONS_PER_AGENT")
fi

if [[ -n "$MAX_STEPS_PER_EPISODE" ]]; then
  cmd+=(--max-steps-per-episode "$MAX_STEPS_PER_EPISODE")
fi

if [[ -n "$RESULT_TIMEOUT" ]]; then
  cmd+=(--result-timeout "$RESULT_TIMEOUT")
fi

if [[ -n "$ENV_ID" ]]; then
  cmd+=(--env-id "$ENV_ID")
fi

if [[ -n "$DURATION" ]]; then
  cmd+=(--duration "$DURATION")
fi

if [[ -n "$OTHER_VEHICLES" ]]; then
  cmd+=(--other-vehicles "$OTHER_VEHICLES")
fi

if [[ -n "$POLICY_FREQUENCY" ]]; then
  cmd+=(--policy-frequency "$POLICY_FREQUENCY")
fi

if [[ -n "$SIMULATION_FREQUENCY" ]]; then
  cmd+=(--simulation-frequency "$SIMULATION_FREQUENCY")
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

model_display="${MODEL_PATH:-bootstrap:random-init}"

printf '[run-adversarial-self-play-dual-gpu-kaggle] scenario=%s config=%s model=%s workers=%s %s device=%s n_simulations=%s num_gpus=%s gpu_indices=%s output_dir=%s install_deps=%s result_timeout=%s\n' \
  "$ALPHAZERO_ADVERSARIAL_SCENARIO" \
  "$ALPHAZERO_ADVERSARIAL_CONFIG_PATH" \
  "$model_display" \
  "$WORKERS" \
  "$episode_mode" \
  "$DEVICE" \
  "$N_SIMULATIONS" \
  "${NUM_GPUS:-auto}" \
  "${GPU_INDICES:-all-visible}" \
  "$OUTPUT_DIR" \
  "$INSTALL_DEPS" \
  "${RESULT_TIMEOUT:-disabled}"

if [[ -n "$LOG_FILE" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
else
  exec "${cmd[@]}"
fi
