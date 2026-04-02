# AlphaZero Module Structure

This folder is organized by feature packages.

## New structure

- `core/`: search logic and state preprocessing
  - `mcts.py`: `MCTSNode`, `MCTS`
  - `policy.py`: `softmax_policy`
  - `settings.py`: stage presets loaded from repo-level scenario configs under `configs/`
  - `state_stack.py`: `StateStackManager`, stack helpers with normalized speed context and an optional raw ego-speed plane
- `network/`: neural network definitions
  - `alphazero_network.py`: `AlphaZeroNetwork`
- `training/`: training pipeline
  - `trainer.py`: `AlphaZeroTrainer`
- `environment/`: environment creation/config
  - `config.py`: `EnvironmentFactory`, `init_env`
- `scripts/`: runnable script workflows
  - `self_play.py`, `infer.py`, `evaluate.py`, `playground_test.py`

## Typical usage

Run scripts directly:

```bash
python AlphaZero/scripts/self_play.py
python AlphaZero/scripts/infer.py
python AlphaZero/scripts/evaluate.py
```

Or use the new structured modules directly in code:

```python
from AlphaZero.network.alphazero_network import AlphaZeroNetwork
from AlphaZero.training.trainer import AlphaZeroTrainer
from AlphaZero.environment.config import init_env
```

## Configure Once

Tune scenario defaults in one place:

- `configs/racetrack.yaml`
- `configs/highway.yaml`

Examples:

- `StackConfig.grid_size`, `StackConfig.ego_position`
- `StackConfig.append_raw_ego_speed_plane` (whether to append a raw ego-speed plane)
- `AlphaZeroConfig.n_actions`, `AlphaZeroConfig.n_residual_layers`
- Presets: `SELF_PLAY_CONFIG`, `INFERENCE_CONFIG`, `EVALUATION_CONFIG`

## Scenario Switching

- By default the stack reads `configs/racetrack.yaml`.
- Set `ALPHAZERO_SCENARIO=highway` to load `configs/highway.yaml`.
- Optionally set `ALPHAZERO_CONFIG_PATH=/path/to/custom.yaml` to use a different config file altogether.
