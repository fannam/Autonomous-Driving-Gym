# AlphaZero Adversarial Autonomous Driving

This package is a second AlphaZero-style implementation for the repository, focused on a two-agent adversarial driving game:

- Agent 0: ego vehicle that tries to survive and finish safely.
- Agent 1: adversarial NPC that tries to force ego into a collision.
- Search: simultaneous-move MCTS with decoupled PUCT statistics.
- Model: one shared-weight network used from both viewpoints.

## Layout

- `AlphaZeroAdversarial/core/`
  - `runtime_config.py`: config loading from the local `configs/` folder
  - `settings.py`: dataclass presets for tensor/search/training config
  - `game.py`: terminal-state logic, action helpers, zero-sum payoff mapping
  - `perspective_stack.py`: `2N + k` tensor builder with self/opponent history
  - `mcts.py`: simultaneous adversarial MCTS
- `AlphaZeroAdversarial/environment/`
  - `config.py`: multi-agent environment factory
- `AlphaZeroAdversarial/network/`
  - `alphazero_network.py`: shared policy/value network
- `AlphaZeroAdversarial/training/`
  - `trainer.py`: self-play, replay buffer, learner loop, curriculum warmup
- `AlphaZeroAdversarial/scripts/`
  - `self_play.py`: run one self-play episode
  - `train.py`: iterative self-play + training
  - `evaluate.py`: greedy evaluation with a saved checkpoint

## Scenario Config

The default scenario is `configs/racetrack_adversarial.yaml`.

It configures:

- `racetrack-v0`
- `controlled_vehicles=2`
- `MultiAgentAction`
- `MultiAgentObservation`
- a local occupancy-grid observation used only for static road features

## Usage

```bash
python AlphaZeroAdversarial/scripts/self_play.py
python AlphaZeroAdversarial/scripts/train.py --iterations 3 --episodes-per-iteration 2
python AlphaZeroAdversarial/scripts/evaluate.py --model-path models/alphazero_adversarial_racetrack.pth
```

## Notes

- The adversarial game is zero-sum at the payoff level, even though the environment itself is not.
- Ego collision is treated as an NPC win.
- Ego timeout while still safe and above the configured minimum speed is treated as an ego win.
- Off-road violations and NPC self-destruction are mapped to draws to reduce reward hacking.
