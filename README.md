# Autonomous Driving Gym

This repository contains several AlphaZero-style autonomous driving codebases plus a local fork of `highway-env`. The directory layout is now split by lifecycle:

- `source/`
  Actively maintained packages:
  - `AlphaZero-adversarial-autonomous-driving/`
  - `AlphaZero-meta-adversarial-autonomous-driving/`
  - `autonomous_driving_shared/`
  - `highway-env/`
- `legacy/`
  Older but still runnable single-agent AlphaZero code:
  - `AlphaZero-based-autonomous-driving/`
- `scripts/`
  Optional bash wrappers around the Python entrypoints.
- `notebooks/`
  Workflow and analysis notebooks.
- `tools/`
  Layout/bootstrap helpers such as `tools/repo_layout.py`.

## Installation

Requirements:

- Python `>= 3.12`
- `uv`

From the repository root:

```bash
uv sync
```

This installs the root workspace and the local editable `source/highway-env` package.

## Layout Guide

Use `source/` for new development. Keep compatibility with the legacy single-agent implementation by treating `legacy/` as read-mostly unless you are fixing old workflows.

- `source/AlphaZero-adversarial-autonomous-driving/`
  Two-agent adversarial AlphaZero on top of local `highway-env`.
- `source/AlphaZero-meta-adversarial-autonomous-driving/`
  Two-agent adversarial AlphaZero using `DiscreteMetaAction` on `highway-v0`.
- `source/autonomous_driving_shared/`
  Shared core utilities factored out of the adversarial and meta-adversarial packages.
- `legacy/AlphaZero-based-autonomous-driving/`
  Original single-agent AlphaZero training and self-play code.

The Python package names are unchanged:

- `AlphaZeroAdversarial`
- `AlphaZeroMetaAdversarial`
- `AlphaZero`

## Main Entry Points

Run commands from the repository root.

Legacy single-agent self-play:

```bash
uv run python legacy/AlphaZero-based-autonomous-driving/AlphaZero/scripts/self_play_parallel.py \
  --workers 2 \
  --episodes-per-worker 2 \
  --finish-laps 1 \
  --duration 80
```

Adversarial render/debug flow on `highway-v0` with two controlled vehicles:

```bash
uv run python source/AlphaZero-adversarial-autonomous-driving/AlphaZeroAdversarial/scripts/render_steps.py \
  --env-id highway-v0 \
  --scenario-name highway_adversarial \
  --controlled-vehicles 2 \
  --steps 8
```

Single-agent render smoke test on `highway-v0`:

```bash
uv run python source/highway-env/scripts/render_highway_v0.py \
  --steps 20 \
  --render-mode rgb_array
```

Two-agent `DiscreteMetaAction` render on `highway-v0` with both controlled vehicles visible:

```bash
uv run python source/highway-env/scripts/render_highway_v0_discrete_meta.py \
  --steps 20 \
  --render-mode human \
  --camera-mode midpoint
```

Meta-adversarial training:

```bash
uv run python source/AlphaZero-meta-adversarial-autonomous-driving/AlphaZeroMetaAdversarial/scripts/train.py \
  --iterations 3 \
  --episodes-per-iteration 2
```

Meta-adversarial self-play export only:

```bash
bash scripts/run_meta_adversarial_self_play.sh --episodes 4 --episodes-per-shard 2
```

Equivalent bash wrappers remain under `scripts/`. If you are on Windows, prefer the `uv run python ...` commands directly.

## Notes

- The legacy racetrack workflow still uses lap-based finish semantics in the local `highway-env` fork.
- Adversarial and meta-adversarial packages share common logic via `source/autonomous_driving_shared/`.
- Tests and local path bootstrapping are aligned with the new layout through `tests/conftest.py` and `tools/repo_layout.py`.
