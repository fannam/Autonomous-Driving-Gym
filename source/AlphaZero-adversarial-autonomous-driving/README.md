# AlphaZero Adversarial Autonomous Driving

This package is a second AlphaZero-style implementation for the repository, focused on a two-agent adversarial driving game:

- Agent 0: ego vehicle that tries to survive and finish safely.
- Agent 1: adversarial NPC that tries to force ego into a collision.
- Search: simultaneous-move MCTS with decoupled PUCT statistics.
- Model: one shared-weight late-fusion network used from both viewpoints.

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
- a role-aware target vector fused with the CNN embedding before the policy/value heads

## Target Vector

The racetrack adversarial model now uses late fusion:

- CNN branch: local occupancy grid plus self/opponent history
- Target-vector branch: low-dimensional global guidance signal
- Fusion: concatenate both embeddings before the shared factorized policy/value heads

Target semantics are role-specific:

- Ego target: a route waypoint projected forward along the active lane/route
- NPC target: an ego intercept point computed from distance and relative speed

The target vector is normalized into `[-1, 1]` and contains:

- `dx, dy`: local-frame target displacement
- `dvx, dvy`: local-frame relative target velocity
- `sin(bearing), cos(bearing)`: heading-safe angular target cue
- `role_bit`: `+1` for ego, `-1` for NPC
- `target_type_bit`: `+1` for route-lookahead, `-1` for intercept

## Usage

```bash
uv run python source/AlphaZero-adversarial-autonomous-driving/AlphaZeroAdversarial/scripts/self_play.py
uv run python source/AlphaZero-adversarial-autonomous-driving/AlphaZeroAdversarial/scripts/train.py --iterations 3 --episodes-per-iteration 2
uv run python source/AlphaZero-adversarial-autonomous-driving/AlphaZeroAdversarial/scripts/evaluate.py --model-path source/AlphaZero-adversarial-autonomous-driving/models/alphazero_adversarial_racetrack.pth
```

For the quick render/debug flow on `highway-v0` with exactly two controlled vehicles:

```bash
uv run python source/AlphaZero-adversarial-autonomous-driving/AlphaZeroAdversarial/scripts/render_steps.py \
  --env-id highway-v0 \
  --scenario-name highway_adversarial \
  --controlled-vehicles 2 \
  --steps 8
```

## Notes

- Terminal payoffs distinguish direct NPC hits from self-inflicted failures.
- Direct NPC-to-ego collisions are treated as NPC wins.
- Ego self-collisions/off-road failures penalize ego without rewarding NPC.
- NPC self-collisions/off-road failures penalize NPC without rewarding ego.
- Ego timeout while still safe and above the configured minimum speed is treated as an ego win.
- Self-play shards now store `target_vectors` alongside `states`, factorized policy targets, and `values`.
- Checkpoints created before the late-fusion target-vector upgrade are not compatible with the current network state dict.
