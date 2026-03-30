#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${REPO_ROOT}/AlphaZero-meta-adversarial-autonomous-driving"
HIGHWAY_ENV_ROOT="${REPO_ROOT}/highway-env"
cd "${REPO_ROOT}"

# Notebook environments often leak an inline backend that breaks subprocess rendering.
if [[ "${MPLBACKEND:-}" == "module://matplotlib_inline.backend_inline" || -z "${MPLBACKEND:-}" ]]; then
  export MPLBACKEND="Agg"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
UPGRADE_PIP="${UPGRADE_PIP:-0}"
LOG_FILE="${LOG_FILE:-}"
TORCH_PACKAGE_SPEC="${TORCH_PACKAGE_SPEC:-torch}"
PIP_FLAGS=()

export ALPHAZERO_META_ADVERSARIAL_SCENARIO="${ALPHAZERO_META_ADVERSARIAL_SCENARIO:-highway_meta_adversarial}"
export ALPHAZERO_META_ADVERSARIAL_CONFIG_PATH="${ALPHAZERO_META_ADVERSARIAL_CONFIG_PATH:-$PROJECT_ROOT/configs/highway_meta_adversarial.yaml}"
export PYTHONPATH="${HIGHWAY_ENV_ROOT}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

ITERATIONS="${ITERATIONS:-1}"
EPISODES_PER_ITERATION="${EPISODES_PER_ITERATION:-1}"
SEED_START="${SEED_START:-21}"
ENV_SEED="${ENV_SEED:-10}"
DEVICE="${DEVICE:-auto}"
MAX_STEPS="${MAX_STEPS:-}"
RESUME_MODEL="${RESUME_MODEL:-}"
SAVE_PATH="${SAVE_PATH:-}"
QUIET="${QUIET:-0}"

usage() {
  cat <<EOF
Usage:
  bash scripts/run_meta_adversarial_train.sh [train args...]

Environment variables:
  PYTHON_BIN               Python executable to use (default: python)
  INSTALL_DEPS             1 to install missing deps and local editable packages (default: 1)
  UPGRADE_PIP              1 to upgrade pip/setuptools/wheel first (default: 0)
  TORCH_PACKAGE_SPEC       Torch package spec for pip (default: ${TORCH_PACKAGE_SPEC})
  TORCH_INDEX_URL          Optional pip index URL for torch wheels
  ITERATIONS               Default --iterations value (default: ${ITERATIONS})
  EPISODES_PER_ITERATION   Default --episodes-per-iteration value (default: ${EPISODES_PER_ITERATION})
  SEED_START               Default --seed-start value (default: ${SEED_START})
  ENV_SEED                 Default --env-seed value (default: ${ENV_SEED})
  DEVICE                   Default --device value (default: ${DEVICE})
  MAX_STEPS                Optional --max-steps value
  RESUME_MODEL             Optional --resume-model value
  SAVE_PATH                Optional --save-path value
  QUIET                    1/true to pass --quiet
  LOG_FILE                 Optional file to tee stdout/stderr into
  ALPHAZERO_META_ADVERSARIAL_SCENARIO
                           Scenario name (default: ${ALPHAZERO_META_ADVERSARIAL_SCENARIO})
  ALPHAZERO_META_ADVERSARIAL_CONFIG_PATH
                           Scenario config path (default: ${ALPHAZERO_META_ADVERSARIAL_CONFIG_PATH})

Examples:
  bash scripts/run_meta_adversarial_train.sh --iterations 3 --episodes-per-iteration 2
  INSTALL_DEPS=0 DEVICE=cuda bash scripts/run_meta_adversarial_train.sh
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
  "${PROJECT_ROOT}/AlphaZeroMetaAdversarial/scripts/train.py"
  --iterations "${ITERATIONS}"
  --episodes-per-iteration "${EPISODES_PER_ITERATION}"
  --seed-start "${SEED_START}"
  --env-seed "${ENV_SEED}"
  --device "${DEVICE}"
)

if [[ -n "${MAX_STEPS}" ]]; then
  cmd+=(--max-steps "${MAX_STEPS}")
fi

if [[ -n "${RESUME_MODEL}" ]]; then
  cmd+=(--resume-model "${RESUME_MODEL}")
fi

if [[ -n "${SAVE_PATH}" ]]; then
  cmd+=(--save-path "${SAVE_PATH}")
fi

quiet_lower="${QUIET,,}"
if [[ "${QUIET}" == "1" || "${quiet_lower}" == "true" ]]; then
  cmd+=(--quiet)
fi

cmd+=("$@")

printf '[run-meta-adversarial-train] scenario=%s config=%s iterations=%s episodes_per_iteration=%s device=%s install_deps=%s save_path=%s\n' \
  "${ALPHAZERO_META_ADVERSARIAL_SCENARIO}" \
  "${ALPHAZERO_META_ADVERSARIAL_CONFIG_PATH}" \
  "${ITERATIONS}" \
  "${EPISODES_PER_ITERATION}" \
  "${DEVICE}" \
  "${INSTALL_DEPS}" \
  "${SAVE_PATH:-default}"

if [[ -n "${LOG_FILE}" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
else
  exec "${cmd[@]}"
fi
