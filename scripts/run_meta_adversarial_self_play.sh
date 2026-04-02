#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${REPO_ROOT}/source/AlphaZero-meta-adversarial-autonomous-driving"
HIGHWAY_ENV_ROOT="${REPO_ROOT}/source/highway-env"
cd "${REPO_ROOT}"

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

EPISODES="${EPISODES:-1}"
EPISODES_PER_SHARD="${EPISODES_PER_SHARD:-1}"
SEED_START="${SEED_START:-21}"
EPISODE_INDEX_START="${EPISODE_INDEX_START:-0}"
ENV_SEED="${ENV_SEED:-10}"
DEVICE="${DEVICE:-auto}"
NETWORK_SEED="${NETWORK_SEED:-42}"
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-10}"
QUIET="${QUIET:-0}"

usage() {
  cat <<EOF
Usage:
  bash scripts/run_meta_adversarial_self_play.sh [self-play args...]

Environment variables:
  PYTHON_BIN               Python executable to use (default: ${PYTHON_BIN_DEFAULT})
  INSTALL_DEPS             1 to install missing deps and local editable packages (default: 1)
  UPGRADE_PIP              1 to upgrade pip/setuptools/wheel first (default: 0)
  TORCH_PACKAGE_SPEC       Torch package spec for pip (default: ${TORCH_PACKAGE_SPEC})
  TORCH_INDEX_URL          Optional pip index URL for torch wheels
  EPISODES                 Default --episodes value (default: ${EPISODES})
  EPISODES_PER_SHARD       Default --episodes-per-shard value (default: ${EPISODES_PER_SHARD})
  SEED_START               Default --seed-start value (default: ${SEED_START})
  EPISODE_INDEX_START      Default --episode-index-start value (default: ${EPISODE_INDEX_START})
  ENV_SEED                 Default --env-seed value (default: ${ENV_SEED})
  DEVICE                   Default --device value (default: ${DEVICE})
  NETWORK_SEED             Default --network-seed value (default: ${NETWORK_SEED})
  MODEL_PATH               Optional --model-path value
  OUTPUT_DIR               Optional --output-dir value
  MAX_STEPS_PER_EPISODE    Optional --max-steps-per-episode value
  PROGRESS_INTERVAL        Default --progress-interval value (default: ${PROGRESS_INTERVAL})
  QUIET                    1/true to pass --quiet
  LOG_FILE                 Optional file to tee stdout/stderr into
  ALPHAZERO_META_ADVERSARIAL_SCENARIO
                           Scenario name (default: ${ALPHAZERO_META_ADVERSARIAL_SCENARIO})
  ALPHAZERO_META_ADVERSARIAL_CONFIG_PATH
                           Scenario config path (default: ${ALPHAZERO_META_ADVERSARIAL_CONFIG_PATH})

Examples:
  bash scripts/run_meta_adversarial_self_play.sh --episodes 4 --episodes-per-shard 2
  INSTALL_DEPS=0 NETWORK_SEED=42 bash scripts/run_meta_adversarial_self_play.sh
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

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "Python 3.12+ is required by AlphaZero-meta-adversarial-autonomous-driving." >&2
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
  "${PROJECT_ROOT}/AlphaZeroMetaAdversarial/scripts/self_play_save.py"
  --episodes "${EPISODES}"
  --episodes-per-shard "${EPISODES_PER_SHARD}"
  --seed-start "${SEED_START}"
  --episode-index-start "${EPISODE_INDEX_START}"
  --env-seed "${ENV_SEED}"
  --device "${DEVICE}"
  --network-seed "${NETWORK_SEED}"
  --progress-interval "${PROGRESS_INTERVAL}"
)

if [[ -n "${MODEL_PATH}" ]]; then
  cmd+=(--model-path "${MODEL_PATH}")
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ -n "${MAX_STEPS_PER_EPISODE}" ]]; then
  cmd+=(--max-steps-per-episode "${MAX_STEPS_PER_EPISODE}")
fi

quiet_lower="${QUIET,,}"
if [[ "${QUIET}" == "1" || "${quiet_lower}" == "true" ]]; then
  cmd+=(--quiet)
fi

cmd+=("$@")

printf '[run-meta-adversarial-self-play] scenario=%s config=%s episodes=%s episodes_per_shard=%s device=%s network_seed=%s output_dir=%s install_deps=%s\n' \
  "${ALPHAZERO_META_ADVERSARIAL_SCENARIO}" \
  "${ALPHAZERO_META_ADVERSARIAL_CONFIG_PATH}" \
  "${EPISODES}" \
  "${EPISODES_PER_SHARD}" \
  "${DEVICE}" \
  "${NETWORK_SEED}" \
  "${OUTPUT_DIR:-auto}" \
  "${INSTALL_DEPS}"

if [[ -n "${LOG_FILE}" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
else
  exec "${cmd[@]}"
fi
