# AlphaZero Module Structure

This folder has been refactored into feature-based packages while keeping
legacy entry files for backward compatibility.

## New structure

- `core/`: search logic and state preprocessing
  - `mcts.py`: `MCTSNode`, `MCTS`
  - `policy.py`: `softmax_policy`
  - `state_stack.py`: `StateStackManager`, 8-layer and 9-layer stack helpers
- `network/`: neural network definitions
  - `alphazero_network.py`: `AlphaZeroNetwork`
- `training/`: training pipeline
  - `trainer.py`: `AlphaZeroTrainer`
- `environment/`: environment creation/config
  - `config.py`: `EnvironmentFactory`, `init_env`
- `scripts/`: runnable script workflows
  - `self_play.py`, `infer.py`, `evaluate.py`, `playground_test.py`

## Backward compatibility

Original files in this folder are still available (`MCTS.py`, `trainer.py`,
`CNN_alphazero.py`, `stack_of_planes.py`, etc.) and now re-export from the new
modules.

## Typical usage

Run legacy entrypoints (unchanged command style):

```bash
python AlphaZero/self_play.py
python AlphaZero/infer.py
python AlphaZero/evaluate.py
```

Or use the new structured modules directly in code:

```python
from AlphaZero.network.alphazero_network import AlphaZeroNetwork
from AlphaZero.training.trainer import AlphaZeroTrainer
from AlphaZero.environment.config import init_env
```
