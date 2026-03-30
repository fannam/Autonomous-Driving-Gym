# AlphaZero Meta-Adversarial Highway Driving

This package is a separate AlphaZero-style implementation for a two-agent adversarial highway driving game using `MultiAgentAction` + `DiscreteMetaAction`.

- Agent 0: ego vehicle that tries to survive and finish safely on `highway-v0`.
- Agent 1: adversarial NPC that tries to disrupt ego with the same discrete meta actions.
- Search: simultaneous-move MCTS with decoupled PUCT statistics.
- Model: one shared-weight late-fusion network used from both viewpoints, with a flat policy over meta actions.

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
- an auxiliary target vector that is fused after the CNN trunk

## Target Vector

The meta-adversarial model now uses late fusion:

- CNN branch: local occupancy grid plus self/opponent history
- Target-vector branch: low-dimensional global guidance signal
- Fusion: concatenate both embeddings before the shared policy/value heads

Target semantics are role-specific:

- Ego target: a dynamic route waypoint projected forward along its lane/route
- NPC target: an ego intercept point computed from distance and relative speed

The target vector is normalized into `[-1, 1]` and contains:

- `dx, dy`: local-frame target displacement
- `dvx, dvy`: local-frame relative target velocity
- `sin(bearing), cos(bearing)`: heading-safe angular target cue
- `role_bit`: `+1` for ego, `-1` for NPC
- `target_type_bit`: `+1` for route-lookahead, `-1` for intercept

## Usage

From the repo root:

```bash
bash scripts/run_meta_adversarial_train.sh --iterations 3 --episodes-per-iteration 2
python AlphaZero-meta-adversarial-autonomous-driving/AlphaZeroMetaAdversarial/scripts/evaluate.py --model-path AlphaZero-meta-adversarial-autonomous-driving/models/alphazero_meta_adversarial_highway.pth
```

`bash scripts/run_meta_adversarial_train.sh` is the main end-to-end entrypoint. It lives at the repo root, uses the `highway-env` source from this repo via local editable install and `PYTHONPATH`, and follows the same lightweight install style as the other repo bash scripts.

If the current Python environment is already prepared, you can skip installs:

```bash
INSTALL_DEPS=0 bash scripts/run_meta_adversarial_train.sh --iterations 1 --episodes-per-iteration 1
```

## Notes

- The action space is the flat `DiscreteMetaAction` set instead of factorized accelerate/steering bins.
- Self-play shards now store `target_vectors` alongside `states`, `policies`, and `values`.
- Terminal payoffs distinguish direct NPC hits from self-inflicted failures.
- Direct NPC-to-ego collisions are treated as NPC wins.
- Ego self-collisions/off-road failures penalize ego without rewarding NPC.
- NPC self-collisions/off-road failures penalize NPC without rewarding ego.
- Ego timeout while still safe and above the configured minimum speed is treated as an ego win.
- Checkpoints created before the late-fusion target-vector upgrade are not compatible with the current network state dict.
