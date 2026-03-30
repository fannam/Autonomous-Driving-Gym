# AlphaZero Meta-Adversarial Highway Driving

This package is a separate AlphaZero-style implementation for a two-agent adversarial highway driving game using `MultiAgentAction` + `DiscreteMetaAction`.

- Agent 0: ego vehicle that tries to survive and finish safely on `highway-v0`.
- Agent 1: adversarial NPC that tries to disrupt ego with the same discrete meta actions.
- Search: simultaneous-move MCTS with decoupled PUCT statistics.
- Model: one shared-weight network used from both viewpoints, with a flat policy over meta actions.

## Layout

- `AlphaZeroMetaAdversarial/core/`
  - `runtime_config.py`: config loading from the local `configs/` folder
  - `settings.py`: dataclass presets for tensor/search/training config
  - `game.py`: terminal-state logic, action helpers, zero-sum payoff mapping
  - `perspective_stack.py`: `2N + k` tensor builder with self/opponent history
  - `mcts.py`: simultaneous adversarial MCTS
- `AlphaZeroMetaAdversarial/environment/`
  - `config.py`: multi-agent environment factory
- `AlphaZeroMetaAdversarial/network/`
  - `alphazero_network.py`: shared policy/value network
- `AlphaZeroMetaAdversarial/training/`
  - `trainer.py`: self-play, replay buffer, learner loop, curriculum warmup
- `AlphaZeroMetaAdversarial/scripts/`
  - `self_play.py`: run one self-play episode
  - `train.py`: iterative self-play + training
  - `evaluate.py`: greedy evaluation with a saved checkpoint

## Scenario Config

The default scenario is `configs/highway_meta_adversarial.yaml`.

It configures:

- `highway-v0`
- `controlled_vehicles=2`
- `MultiAgentAction`
- `DiscreteMetaAction`
- `MultiAgentObservation`
- a local occupancy-grid observation used together with mirrored self/opponent history tensors

## Usage

```bash
python AlphaZeroMetaAdversarial/scripts/self_play.py
python AlphaZeroMetaAdversarial/scripts/train.py --iterations 3 --episodes-per-iteration 2
python AlphaZeroMetaAdversarial/scripts/evaluate.py --model-path models/alphazero_meta_adversarial_highway.pth
```

## Notes

- The adversarial game is zero-sum at the payoff level, even though `highway-v0` itself is not.
- The action space is the flat `DiscreteMetaAction` set instead of factorized accelerate/steering bins.
- Ego collision is treated as an NPC win.
- Ego timeout while still safe and above the configured minimum speed is treated as an ego win.
- Off-road violations and NPC self-destruction are mapped to draws to reduce reward hacking.
