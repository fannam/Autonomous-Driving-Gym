# PPO Evolutionary Highway Driving

This package implements a synchronous PPO-Evolutionary Reinforcement Learning baseline for `highway-v0`.

- Environment: single-agent `highway-v0`
- Observation: `DetailedOccupancyGrid`
- Actions: `DiscreteMetaAction` with 5 actions
- Learner: custom PyTorch PPO with a residual CNN actor-critic
- Evolution: elitism plus Gaussian parameter mutation
- Parallelism: synchronous rollout collection with Python `multiprocessing`

## Layout

- `configs/`
  Runtime scenario config and train/evaluation presets
- `PPOEvolutionary/core/`
  Runtime config loading, settings dataclasses, serialized training types
- `PPOEvolutionary/environment/`
  Environment factory and reward wrapper
- `PPOEvolutionary/network/`
  Residual CNN actor-critic
- `PPOEvolutionary/training/`
  Rollout, evolution, and PPO trainer logic
- `PPOEvolutionary/scripts/`
  `train.py` and `evaluate.py`

## Usage

From the repository root:

```bash
uv run python source/algorithm/PPO-based/PPO-evolutionary-algorithm/PPOEvolutionary/scripts/train.py \
  --generations 2 \
  --population-size 4 \
  --workers 2
```

```bash
uv run python source/algorithm/PPO-based/PPO-evolutionary-algorithm/PPOEvolutionary/scripts/evaluate.py \
  --checkpoint-path source/algorithm/PPO-based/PPO-evolutionary-algorithm/models/ppo_evolutionary_highway.pt \
  --episodes 2
```

## Notes

- PPO only consumes rollout data from the current generation.
- The reward used for training and evaluation is shaped by the local wrapper, not by patching `highway-env`.
- Checkpoints store both the latest PPO master policy and the best-so-far evolved policy.
