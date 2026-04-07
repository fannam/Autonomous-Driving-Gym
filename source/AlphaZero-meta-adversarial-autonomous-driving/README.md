# AlphaZero Meta-Adversarial Highway Driving

This package is a separate AlphaZero-style implementation for a two-agent adversarial highway driving game using `MultiAgentAction` + `DiscreteMetaAction`.

- Agent 0: ego vehicle that tries to survive and finish safely on `highway-v0`.
- Agent 1: adversarial NPC that tries to disrupt ego with the same discrete meta actions.
  In the default scenario it uses a slightly faster target-speed ladder than ego.
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
  - `self_play_save.py`: run self-play only and save shard/manifest outputs
  - `train.py`: iterative self-play + training
  - `evaluate.py`: greedy evaluation with a saved checkpoint

## Scenario Config

The default scenario is `configs/highway_meta_adversarial.yaml`.

It configures:

- `highway-v0`
- `controlled_vehicles=2`
- `MultiAgentAction`
- `DiscreteMetaAction`
- ego base target speeds `[18, 22, 26, 30, 34, 38]`
- NPC per-agent target-speed override `[22, 26, 30, 34, 38, 42]`
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

## Terminal Outcomes

The shared terminal-state logic lives in `autonomous_driving_shared/alphazero_adversarial/core/game.py`.
For the default `highway_meta_adversarial` scenario, `remove_npc_on_self_fault=true`, so NPC self-faults assign NPC `-1.0` and remove it from the episode instead of always ending the full episode immediately.

| `reason` | `terminal` | `ego_value` | `npc_value` | Meaning |
| --- | --- | --- | --- | --- |
| `ongoing` | `False` | current runtime value, usually `0.0` | current runtime value, usually `0.0` | Episode continues. After an NPC self-fault in the default config, this can be `(0.0, -1.0)` while ego keeps driving. |
| `ego_finished` | `True` | `+1.0` | `-1.0` | Ego reaches the success condition safely. |
| `npc_hit_ego` | `True` | `-1.0` | `+1.0` | Direct ego/NPC collision attributed to NPC. |
| `double_self_collision` | `True` | `-1.0` | `-1.0` | Both agents crash, but not by hitting each other directly. |
| `ego_self_collision` | `True` | `-1.0` | `0.0` | Ego crashes without NPC earning credit. |
| `npc_self_collision` | config-dependent | `0.0` | `-1.0` | NPC crashes without rewarding ego. In the default config, NPC is penalized and removed, so the episode continues. |
| `double_offroad` | `True` | `-1.0` | `-1.0` | Both agents leave the road. |
| `ego_offroad` | `True` | `-1.0` | `0.0` | Ego leaves the road without NPC earning credit. |
| `npc_offroad` | config-dependent | `0.0` | `-1.0` | NPC leaves the road without rewarding ego. In the default config, NPC is penalized and removed, so the episode continues. |
| `ego_timeout_safe` | `True` | `+1.0` | `-1.0` | The time limit is reached while ego remains safe and above `minimum_safe_speed`. |
| `timeout_draw` | `True` | current runtime value, usually `0.0` | current runtime value, usually `0.0` | Time limit draw. If NPC already self-faulted earlier in the default config, this becomes `(0.0, -1.0)`. |
| `terminated_draw` | `True` | current runtime value, usually `0.0` | current runtime value, usually `0.0` | Fallback terminal draw that preserves any previously recorded runtime values. |

## Usage

From the repo root:

```bash
bash scripts/run_meta_adversarial_train.sh --iterations 3 --episodes-per-iteration 2
bash scripts/run_meta_adversarial_self_play.sh --episodes 4 --episodes-per-shard 2
uv run python source/AlphaZero-meta-adversarial-autonomous-driving/AlphaZeroMetaAdversarial/scripts/evaluate.py --model-path source/AlphaZero-meta-adversarial-autonomous-driving/models/alphazero_meta_adversarial_highway.pth
```

`bash scripts/run_meta_adversarial_train.sh` is the main end-to-end entrypoint. It lives at the repo root, uses the `highway-env` source from this repo via local editable install and `PYTHONPATH`, and follows the same lightweight install style as the other repo bash scripts.

If the current Python environment is already prepared, you can skip installs:

```bash
INSTALL_DEPS=0 bash scripts/run_meta_adversarial_train.sh --iterations 1 --episodes-per-iteration 1
INSTALL_DEPS=0 bash scripts/run_meta_adversarial_self_play.sh --episodes 2 --network-seed 42
```

For distributed self-play across multiple machines, prefer `bash scripts/run_meta_adversarial_self_play.sh`.
When `--model-path` is omitted, it creates and records a bootstrap checkpoint from a fixed `--network-seed` (default `42`), saves shard `.pt` files plus `manifest.json`, and includes the model SHA-256 checksum so different machines can verify they used the same initial network.

## Notes

- The action space is the flat `DiscreteMetaAction` set instead of factorized accelerate/steering bins.
- Self-play shards now store `target_vectors` alongside `states`, `policies`, and `values`.
- Terminal payoffs distinguish direct NPC hits from self-inflicted failures.
- Direct NPC-to-ego collisions are treated as NPC wins.
- Ego self-collisions/off-road failures penalize ego without rewarding NPC.
- NPC self-collisions/off-road failures penalize NPC without rewarding ego.
- Ego timeout while still safe and above the configured minimum speed is treated as an ego win.
- Checkpoints created before the late-fusion target-vector upgrade are not compatible with the current network state dict.
