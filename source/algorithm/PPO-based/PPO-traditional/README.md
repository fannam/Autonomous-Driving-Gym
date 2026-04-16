# PPO Traditional Highway Driving

This package implements a traditional on-policy PPO baseline for `highway-v0`.

- Environment: single-agent `highway-v0`
- Observation: `DetailedOccupancyGrid`
- Actions: `DiscreteMetaAction` with 5 actions
- Learner: custom PyTorch PPO with a residual CNN actor-critic
- Parallelism: vectorized rollout collection with Gymnasium vector environments

## Layout

- `configs/`
  Runtime scenario config and train/evaluation presets
- `PPOTraditional/core/`
  Runtime config loading, settings dataclasses, serialized rollout/training types
- `PPOTraditional/environment/`
  Environment factory and reward wrapper
- `PPOTraditional/network/`
  Residual CNN actor-critic
- `PPOTraditional/training/`
  Vectorized collector and PPO trainer logic
- `PPOTraditional/scripts/`
  `train.py` and `evaluate.py`

## Usage

From the repository root:

```bash
uv run python source/algorithm/PPO-based/PPO-traditional/PPOTraditional/scripts/train.py \
  --updates 2 \
  --n-envs 4 \
  --steps-per-env 32
```

```bash
uv run python source/algorithm/PPO-based/PPO-traditional/PPOTraditional/scripts/evaluate.py \
  --checkpoint-path source/algorithm/PPO-based/PPO-traditional/models/ppo_traditional_highway.pt \
  --episodes 2
```

## Notes

- PPO uses only on-policy rollout data from the current update.
- Reward shaping matches `PPOEvolutionary` for a fair baseline comparison.
- Checkpoints store both the latest policy and the best-so-far policy.
