#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${REPO_ROOT}/source/AlphaZero-meta-adversarial-autonomous-driving"
HIGHWAY_ENV_ROOT="${REPO_ROOT}/source/highway-env"
cd "${REPO_ROOT}"

# Kaggle notebooks often export an inline matplotlib backend that breaks in subprocesses.
if [[ "${MPLBACKEND:-}" == "module://matplotlib_inline.backend_inline" || -z "${MPLBACKEND:-}" ]]; then
  export MPLBACKEND="Agg"
fi

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN_DEFAULT="python"
else
  PYTHON_BIN_DEFAULT="python3"
fi

PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN_DEFAULT}}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
UPGRADE_PIP="${UPGRADE_PIP:-0}"
TORCH_PACKAGE_SPEC="${TORCH_PACKAGE_SPEC:-torch}"
LOG_FILE="${LOG_FILE:-}"
PIP_FLAGS=()

export ALPHAZERO_META_ADVERSARIAL_SCENARIO="${ALPHAZERO_META_ADVERSARIAL_SCENARIO:-highway_meta_adversarial}"
export ALPHAZERO_META_ADVERSARIAL_CONFIG_PATH="${ALPHAZERO_META_ADVERSARIAL_CONFIG_PATH:-$PROJECT_ROOT/configs/highway_meta_adversarial.yaml}"
export PYTHONPATH="${REPO_ROOT}/source:${HIGHWAY_ENV_ROOT}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/alphazero_meta_adversarial_self_play}"
WORKERS="${WORKERS:-2}"
TOTAL_EPISODES="${TOTAL_EPISODES:-}"
EPISODES_PER_WORKER="${EPISODES_PER_WORKER:-}"
EPISODES_PER_SHARD="${EPISODES_PER_SHARD:-2}"
DEVICE="${DEVICE:-cuda}"
NUM_GPUS="${NUM_GPUS:-2}"
GPU_INDICES="${GPU_INDICES:-0,1}"
TORCH_THREADS_PER_WORKER="${TORCH_THREADS_PER_WORKER:-1}"
SELF_PLAY_SEED="${SELF_PLAY_SEED:-1000}"
ENV_SEED="${ENV_SEED:-10}"
NETWORK_SEED="${NETWORK_SEED:-42}"
MACHINE_RANK="${MACHINE_RANK:-0}"
GLOBAL_WORKER_OFFSET="${GLOBAL_WORKER_OFFSET:-}"
SEED_BLOCK_SIZE="${SEED_BLOCK_SIZE:-1000000}"
MODEL_PATH="${MODEL_PATH:-}"
N_SIMULATIONS="${N_SIMULATIONS:-24}"
C_PUCT="${C_PUCT:-}"
TEMPERATURE="${TEMPERATURE:-}"
TEMPERATURE_DROP_STEP="${TEMPERATURE_DROP_STEP:-}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-}"
ROOT_EXPLORATION_FRACTION="${ROOT_EXPLORATION_FRACTION:-}"
MAX_EXPAND_ACTIONS_PER_AGENT="${MAX_EXPAND_ACTIONS_PER_AGENT:-}"
REUSE_TREE_BETWEEN_STEPS="${REUSE_TREE_BETWEEN_STEPS:-1}"
MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-10}"
RESULT_TIMEOUT="${RESULT_TIMEOUT:-}"
ENV_ID="${ENV_ID:-}"
DURATION="${DURATION:-}"
VEHICLES_COUNT="${VEHICLES_COUNT:-}"
POLICY_FREQUENCY="${POLICY_FREQUENCY:-}"
SIMULATION_FREQUENCY="${SIMULATION_FREQUENCY:-}"

usage() {
  cat <<EOF
Usage:
  bash scripts/run_meta_adversarial_self_play_dual_gpu_kaggle.sh [self-play args...]

Environment variables:
  PYTHON_BIN               Python executable to use (default: ${PYTHON_BIN_DEFAULT})
  INSTALL_DEPS             1 to install missing deps and local editable packages (default: 1)
  UPGRADE_PIP              1 to upgrade pip/setuptools/wheel first (default: 0)
  TORCH_PACKAGE_SPEC       Torch package spec for pip (default: ${TORCH_PACKAGE_SPEC})
  TORCH_INDEX_URL          Optional pip index URL for torch wheels
  OUTPUT_DIR               Output directory (default: ${OUTPUT_DIR})
  WORKERS                  Number of worker processes (default: ${WORKERS})
  TOTAL_EPISODES           Optional total episode count split across workers
  EPISODES_PER_WORKER      Episodes per worker when TOTAL_EPISODES is unset (default: 2)
  EPISODES_PER_SHARD       Episodes per shard file (default: ${EPISODES_PER_SHARD})
  DEVICE                   Device selector (default: ${DEVICE})
  NUM_GPUS                 Number of GPUs to use when DEVICE is auto/cuda (default: ${NUM_GPUS})
  GPU_INDICES              Comma-separated GPU indices (default: ${GPU_INDICES})
  TORCH_THREADS_PER_WORKER Torch CPU threads per worker (default: ${TORCH_THREADS_PER_WORKER})
  SELF_PLAY_SEED           Base self-play seed (default: ${SELF_PLAY_SEED})
  ENV_SEED                 Base environment seed (default: ${ENV_SEED})
  NETWORK_SEED             Network bootstrap seed (default: ${NETWORK_SEED})
  MACHINE_RANK             Machine rank for distributed self-play (default: ${MACHINE_RANK})
  GLOBAL_WORKER_OFFSET     Optional explicit global worker offset
  SEED_BLOCK_SIZE          Seed block size per global worker (default: ${SEED_BLOCK_SIZE})
  MODEL_PATH               Optional model checkpoint to load
  N_SIMULATIONS            MCTS simulations per decision (default: ${N_SIMULATIONS})
  C_PUCT                   Optional MCTS exploration constant
  TEMPERATURE              Optional action sampling temperature
  TEMPERATURE_DROP_STEP    Optional temperature drop step
  DIRICHLET_ALPHA          Optional root dirichlet alpha
  ROOT_EXPLORATION_FRACTION Optional root exploration fraction
  MAX_EXPAND_ACTIONS_PER_AGENT Optional max expand actions per agent
  REUSE_TREE_BETWEEN_STEPS 0/false to disable tree reuse (default: ${REUSE_TREE_BETWEEN_STEPS})
  MAX_STEPS_PER_EPISODE    Optional per-episode cap
  PROGRESS_INTERVAL        Step log interval (default: ${PROGRESS_INTERVAL})
  RESULT_TIMEOUT           Optional manager timeout in seconds
  ENV_ID                   Optional env id override
  DURATION                 Optional env duration override
  VEHICLES_COUNT           Optional env vehicles-count override
  POLICY_FREQUENCY         Optional env policy-frequency override
  SIMULATION_FREQUENCY     Optional env simulation-frequency override
  LOG_FILE                 Optional file to tee stdout/stderr into

Example:
  INSTALL_DEPS=0 PROGRESS_INTERVAL=1 WORKERS=8 NUM_GPUS=2 GPU_INDICES=0,1 \\
  EPISODES_PER_WORKER=10 MACHINE_RANK=0 MODEL_PATH=source/AlphaZero-meta-adversarial-autonomous-driving/models/bootstrap_model_seed_42.pth \\
  bash scripts/run_meta_adversarial_self_play_dual_gpu_kaggle.sh
EOF
}

ensure_python_module() {
  local module_name="$1"
  local package_spec="$2"
  if ! "${PYTHON_BIN}" -c "import ${module_name}" >/dev/null 2>&1; then
    echo "[install] missing module=${module_name}, installing ${package_spec}" >&2
    "${PYTHON_BIN}" -m pip install "${PIP_FLAGS[@]}" -q "${package_spec}"
  fi
}

install_local_editable() {
  local package_path="$1"
  echo "[install] editable ${package_path}" >&2
  "${PYTHON_BIN}" -m pip install -q --no-deps -e "${package_path}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "Meta-adversarial project not found at: ${PROJECT_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${HIGHWAY_ENV_ROOT}" ]]; then
  echo "Local highway-env repo not found at: ${HIGHWAY_ENV_ROOT}" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  PIP_FLAGS+=(--index-url "${TORCH_INDEX_URL}")
fi

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  if [[ "${UPGRADE_PIP}" == "1" ]]; then
    echo "[install] upgrading pip/setuptools/wheel" >&2
    "${PYTHON_BIN}" -m pip install "${PIP_FLAGS[@]}" -q --upgrade pip setuptools wheel
  fi

  ensure_python_module gymnasium "gymnasium>=1.2.3"
  ensure_python_module yaml "PyYAML>=6.0.2"
  ensure_python_module pygame "pygame>=2.6.0"
  ensure_python_module numpy "numpy>=1.26.0"
  ensure_python_module torch "${TORCH_PACKAGE_SPEC}"

  install_local_editable "${HIGHWAY_ENV_ROOT}"
  install_local_editable "${PROJECT_ROOT}"
fi

cmd=(
  "${PYTHON_BIN}"
  "${PROJECT_ROOT}/AlphaZeroMetaAdversarial/scripts/self_play_kaggle_dual_gpu.py"
  --workers "${WORKERS}"
  --device "${DEVICE}"
  --episodes-per-shard "${EPISODES_PER_SHARD}"
  --torch-threads-per-worker "${TORCH_THREADS_PER_WORKER}"
  --self-play-seed "${SELF_PLAY_SEED}"
  --env-seed "${ENV_SEED}"
  --network-seed "${NETWORK_SEED}"
  --machine-rank "${MACHINE_RANK}"
  --seed-block-size "${SEED_BLOCK_SIZE}"
  --n-simulations "${N_SIMULATIONS}"
  --progress-interval "${PROGRESS_INTERVAL}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${MODEL_PATH}" ]]; then
  cmd+=(--model-path "${MODEL_PATH}")
fi

if [[ -n "${GLOBAL_WORKER_OFFSET}" ]]; then
  cmd+=(--global-worker-offset "${GLOBAL_WORKER_OFFSET}")
fi

if [[ -n "${TOTAL_EPISODES}" && -n "${EPISODES_PER_WORKER}" ]]; then
  echo "Set only one of TOTAL_EPISODES or EPISODES_PER_WORKER." >&2
  exit 1
fi

episode_mode=""
if [[ -n "${TOTAL_EPISODES}" ]]; then
  cmd+=(--total-episodes "${TOTAL_EPISODES}")
  episode_mode="total_episodes=${TOTAL_EPISODES}"
else
  EPISODES_PER_WORKER="${EPISODES_PER_WORKER:-2}"
  cmd+=(--episodes-per-worker "${EPISODES_PER_WORKER}")
  episode_mode="episodes_per_worker=${EPISODES_PER_WORKER}"
fi

if [[ -n "${C_PUCT}" ]]; then
  cmd+=(--c-puct "${C_PUCT}")
fi

if [[ -n "${TEMPERATURE}" ]]; then
  cmd+=(--temperature "${TEMPERATURE}")
fi

if [[ -n "${TEMPERATURE_DROP_STEP}" ]]; then
  cmd+=(--temperature-drop-step "${TEMPERATURE_DROP_STEP}")
fi

if [[ -n "${DIRICHLET_ALPHA}" ]]; then
  cmd+=(--dirichlet-alpha "${DIRICHLET_ALPHA}")
fi

if [[ -n "${ROOT_EXPLORATION_FRACTION}" ]]; then
  cmd+=(--root-exploration-fraction "${ROOT_EXPLORATION_FRACTION}")
fi

if [[ -n "${MAX_EXPAND_ACTIONS_PER_AGENT}" ]]; then
  cmd+=(--max-expand-actions-per-agent "${MAX_EXPAND_ACTIONS_PER_AGENT}")
fi

reuse_tree_lower="${REUSE_TREE_BETWEEN_STEPS,,}"
if [[ "${REUSE_TREE_BETWEEN_STEPS}" == "0" || "${reuse_tree_lower}" == "false" ]]; then
  cmd+=(--no-reuse-tree-between-steps)
fi

if [[ -n "${MAX_STEPS_PER_EPISODE}" ]]; then
  cmd+=(--max-steps-per-episode "${MAX_STEPS_PER_EPISODE}")
fi

if [[ -n "${RESULT_TIMEOUT}" ]]; then
  cmd+=(--result-timeout "${RESULT_TIMEOUT}")
fi

if [[ -n "${ENV_ID}" ]]; then
  cmd+=(--env-id "${ENV_ID}")
fi

if [[ -n "${DURATION}" ]]; then
  cmd+=(--duration "${DURATION}")
fi

if [[ -n "${VEHICLES_COUNT}" ]]; then
  cmd+=(--vehicles-count "${VEHICLES_COUNT}")
fi

if [[ -n "${POLICY_FREQUENCY}" ]]; then
  cmd+=(--policy-frequency "${POLICY_FREQUENCY}")
fi

if [[ -n "${SIMULATION_FREQUENCY}" ]]; then
  cmd+=(--simulation-frequency "${SIMULATION_FREQUENCY}")
fi

device_lower="${DEVICE,,}"
if [[ "${device_lower}" == "cuda" || "${device_lower}" == "auto" ]]; then
  if [[ -n "${NUM_GPUS}" ]]; then
    cmd+=(--num-gpus "${NUM_GPUS}")
  fi
  if [[ -n "${GPU_INDICES}" ]]; then
    cmd+=(--gpu-indices "${GPU_INDICES}")
  fi
fi

cmd+=("$@")

model_display="${MODEL_PATH:-bootstrap:random-init}"

printf '[run-meta-adversarial-self-play-dual-gpu-kaggle] scenario=%s config=%s model=%s workers=%s %s device=%s env_seed=%s self_play_seed=%s network_seed=%s machine_rank=%s global_worker_offset=%s seed_block_size=%s num_gpus=%s gpu_indices=%s output_dir=%s install_deps=%s\n' \
  "${ALPHAZERO_META_ADVERSARIAL_SCENARIO}" \
  "${ALPHAZERO_META_ADVERSARIAL_CONFIG_PATH}" \
  "${model_display}" \
  "${WORKERS}" \
  "${episode_mode}" \
  "${DEVICE}" \
  "${ENV_SEED}" \
  "${SELF_PLAY_SEED}" \
  "${NETWORK_SEED}" \
  "${MACHINE_RANK}" \
  "${GLOBAL_WORKER_OFFSET:-auto}" \
  "${SEED_BLOCK_SIZE}" \
  "${NUM_GPUS:-auto}" \
  "${GPU_INDICES:-all-visible}" \
  "${OUTPUT_DIR}" \
  "${INSTALL_DEPS}"

printf 'reuse_tree_between_steps=%s progress_interval=%s result_timeout=%s\n' \
  "${REUSE_TREE_BETWEEN_STEPS}" \
  "${PROGRESS_INTERVAL}" \
  "${RESULT_TIMEOUT:-disabled}"

if [[ -n "${LOG_FILE}" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
else
  exec "${cmd[@]}"
fi
