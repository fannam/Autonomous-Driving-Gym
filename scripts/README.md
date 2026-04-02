# Shell Wrappers

This directory contains optional bash entrypoints that wrap the Python CLIs under `source/` and `legacy/`.

- `run_progressive_*.sh`
  Legacy single-agent AlphaZero workflows under `legacy/AlphaZero-based-autonomous-driving/`.
- `run_adversarial_*.sh`
  Two-agent adversarial workflows under `source/AlphaZero-adversarial-autonomous-driving/`.
- `run_meta_adversarial_self_play.sh`
  Meta-adversarial self-play only, with shard/manifest export under `source/AlphaZero-meta-adversarial-autonomous-driving/`.
- `run_meta_adversarial_train.sh`
  Meta-adversarial training under `source/AlphaZero-meta-adversarial-autonomous-driving/`.

If you are on Windows and do not want to run `.sh` files, call the underlying Python entrypoints directly with `uv run python ...` from the repository root.

Workflow notebooks live under `../notebooks/`.
