# AlphaZero-based Autonomous Driving

This repository contains AlphaZero-style experiments for autonomous driving on top of a local editable fork of `highway-env`.

The actively maintained code path is the package in `AlphaZero-based-autonomous-driving/AlphaZero/`. The older notebooks in the repository are still useful for historical context, but they are no longer the source of truth for the current implementation.

## What is in this repository

- `AlphaZero-based-autonomous-driving/AlphaZero/`
  Current AlphaZero codebase:
  - `core/`: MCTS, configs, policy normalization, state stacking
  - `network/`: residual policy/value network
  - `training/`: self-play and training logic
  - `scripts/`: runnable workflows
- `highway-env/`
  Local fork of `highway-env`. The racetrack environment has been extended with a lap-based finish condition.
- `README.md`
  This file.

## Current algorithm

The current code follows the standard AlphaZero training target structure:

- The network predicts `(p, v)` from a stacked occupancy-grid state.
- MCTS uses PUCT with root Dirichlet noise during self-play.
- Self-play stores samples as `(s, pi, z)`.
- `pi` comes from MCTS visit counts.
- `z` is the final episode outcome:
  - `+1` for finishing the configured number of laps
  - `-1` for crash or off-road termination
  - `0` for timeout or externally capped episodes

For the racetrack workflow, success is defined explicitly by lap completion rather than by `truncated`.

## Racetrack finish semantics

The local `RacetrackEnv` is a closed loop. It now exposes a finish condition based on:

- `finish_laps`
- `finish_line_segment`
- `finish_line_longitudinal`
- `terminate_on_finish`

By default, the parallel racetrack self-play script uses:

- `finish_laps = 1`
- finish line on segment `("a", "b")`
- `terminate_on_finish = True`

This means the environment can distinguish:

- success: completed the required number of laps
- failure: crashed or left the road
- timeout: reached the environment duration limit

## Installation

The root project uses `uv` and a local editable workspace dependency for `highway-env`.

Requirements:

- Python `>= 3.12`
- `uv`

Install everything from the repository root:

```bash
uv sync
```

This installs:

- `torch`
- `numpy`
- `gymnasium`
- the local editable `highway-env` fork

## Main entry points

Run commands from the repository root.

Single-process self-play:

```bash
uv run python AlphaZero-based-autonomous-driving/AlphaZero/scripts/self_play.py
```

Parallel racetrack self-play:

```bash
uv run python AlphaZero-based-autonomous-driving/AlphaZero/scripts/self_play_parallel_racetrack.py \
  --workers 2 \
  --episodes-per-worker 2 \
  --finish-laps 1 \
  --duration 80
```

Inference with MCTS:

```bash
uv run python AlphaZero-based-autonomous-driving/AlphaZero/scripts/infer.py
```

Evaluation with MCTS:

```bash
uv run python AlphaZero-based-autonomous-driving/AlphaZero/scripts/evaluate.py
```

`infer.py` and `evaluate.py` read their checkpoint path from `INFERENCE_CONFIG.model_path` and `EVALUATION_CONFIG.model_path` in `AlphaZero-based-autonomous-driving/AlphaZero/core/settings.py`. If you want to use another checkpoint without editing the file, call the underlying Python functions directly and pass `model_path=...`.

## Important script behavior

`self_play_parallel_racetrack.py`:

- generates self-play data in parallel worker processes
- saves one `.pt` file per episode
- uses the lap-based racetrack finish condition
- does not apply an extra hard step cap by default

Default output directory:

```text
AlphaZero-based-autonomous-driving/outputs/racetrack_self_play_parallel
```

Useful flags:

- `--workers`
- `--episodes-per-worker`
- `--n-simulations`
- `--c-puct`
- `--temperature`
- `--temperature-drop-step`
- `--finish-laps`
- `--duration`
- `--other-vehicles`
- `--device`
- `--max-steps-per-episode`

If you set `--max-steps-per-episode`, that cap is treated as an external cutoff and produces value target `0` unless the episode has already succeeded or failed.

## Configuration

Shared AlphaZero defaults live in:

```text
AlphaZero-based-autonomous-driving/AlphaZero/core/settings.py
```

Important parameters there include:

- `n_actions`
- `n_residual_layers`
- `c_puct`
- `n_simulations`
- `temperature`
- `root_dirichlet_alpha`
- `root_exploration_fraction`
- `learning_rate`
- `weight_decay`

## Training note

The repository currently provides:

- self-play generation
- model loading/saving
- `AlphaZeroTrainer.train()` for supervised updates on collected samples

There is not yet a dedicated standalone CLI training script in the current codebase. Training is driven through the Python API and the collected self-play tensors.

## Repository status

The codebase has recently been updated to:

- use AlphaZero-style `(s, pi, z)` targets
- use PUCT-style MCTS without random playout rollouts
- use MCTS for inference and evaluation instead of raw policy-head argmax
- add a lap-based finish definition to `racetrack-v0`

If you are comparing behavior with older notebooks or earlier commits, expect differences in:

- value targets
- MCTS terminal handling
- racetrack success semantics
- self-play horizon
