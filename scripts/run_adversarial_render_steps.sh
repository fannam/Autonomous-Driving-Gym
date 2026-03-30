#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
UPGRADE_PIP="${UPGRADE_PIP:-0}"

export ALPHAZERO_ADVERSARIAL_SCENARIO="${ALPHAZERO_ADVERSARIAL_SCENARIO:-racetrack_adversarial}"
export ALPHAZERO_ADVERSARIAL_CONFIG_PATH="${ALPHAZERO_ADVERSARIAL_CONFIG_PATH:-$REPO_ROOT/AlphaZero-adversarial-autonomous-driving/configs/racetrack_adversarial.yaml}"
export PYTHONPATH="$REPO_ROOT/highway-env:$REPO_ROOT/AlphaZero-adversarial-autonomous-driving${PYTHONPATH:+:$PYTHONPATH}"

STAGE="${STAGE:-self_play}"
ENV_SEED="${ENV_SEED:-21}"
STEPS="${STEPS:-8}"
CONTROLLED_VEHICLES="${CONTROLLED_VEHICLES:-2}"
RENDER_MODE="${RENDER_MODE:-rgb_array}"
EGO_POLICY="${EGO_POLICY:-random}"
NPC_POLICY="${NPC_POLICY:-random}"
SLEEP_S="${SLEEP_S:-0.15}"
SAVE_FRAMES_DIR="${SAVE_FRAMES_DIR:-}"
DURATION="${DURATION:-}"
OTHER_VEHICLES="${OTHER_VEHICLES:-}"
POLICY_FREQUENCY="${POLICY_FREQUENCY:-}"
SIMULATION_FREQUENCY="${SIMULATION_FREQUENCY:-}"
ENV_ID="${ENV_ID:-}"
SCENARIO_NAME="${SCENARIO_NAME:-}"

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
  ensure_python_module PIL "Pillow>=10.0.0"

  install_local_editable "$REPO_ROOT/highway-env"
  install_local_editable "$REPO_ROOT/AlphaZero-adversarial-autonomous-driving"
fi

cmd=(
  "$PYTHON_BIN"
  AlphaZero-adversarial-autonomous-driving/AlphaZeroAdversarial/scripts/render_steps.py
  --stage "$STAGE"
  --env-seed "$ENV_SEED"
  --steps "$STEPS"
  --controlled-vehicles "$CONTROLLED_VEHICLES"
  --render-mode "$RENDER_MODE"
  --ego-policy "$EGO_POLICY"
  --npc-policy "$NPC_POLICY"
  --sleep-s "$SLEEP_S"
)

if [[ -n "$SAVE_FRAMES_DIR" ]]; then
  cmd+=(--save-frames-dir "$SAVE_FRAMES_DIR")
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

if [[ -n "$ENV_ID" ]]; then
  cmd+=(--env-id "$ENV_ID")
fi

if [[ -n "$SCENARIO_NAME" ]]; then
  cmd+=(--scenario-name "$SCENARIO_NAME")
fi

cmd+=("$@")

printf '[run-adversarial-render-steps] scenario=%s config=%s stage=%s steps=%s controlled_vehicles=%s render_mode=%s ego_policy=%s npc_policy=%s\n' \
  "$ALPHAZERO_ADVERSARIAL_SCENARIO" \
  "$ALPHAZERO_ADVERSARIAL_CONFIG_PATH" \
  "$STAGE" \
  "$STEPS" \
  "$CONTROLLED_VEHICLES" \
  "$RENDER_MODE" \
  "$EGO_POLICY" \
  "$NPC_POLICY"

exec "${cmd[@]}"
