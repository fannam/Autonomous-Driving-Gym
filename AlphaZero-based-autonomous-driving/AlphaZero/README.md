# AlphaZero Module Structure

This folder is organized by feature packages.

## New structure

- `core/`: search logic and state preprocessing
  - `mcts.py`: `MCTSNode`, `MCTS`
  - `policy.py`: `softmax_policy`
  - `settings.py`: central runtime configs (input shape, grid, ego position, stack planes)
  - `state_stack.py`: `StateStackManager`, 8-layer and 9-layer stack helpers
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

Tune hardcoded defaults in one place:

- `AlphaZero/core/settings.py`

Examples:

- `StackConfig.grid_size`, `StackConfig.ego_position`
- `StackConfig.include_absolute_speed` (8-layer vs 9-layer stack)
- `AlphaZeroConfig.n_actions`, `AlphaZeroConfig.n_residual_layers`
- Presets: `SELF_PLAY_CONFIG`, `INFERENCE_CONFIG`, `EVALUATION_CONFIG`
