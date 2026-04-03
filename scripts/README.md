# Shell Wrappers

This directory now keeps only the Kaggle-oriented bash wrappers.

- `run_adversarial_self_play_dual_gpu_kaggle.sh`
  Kaggle dual-GPU self-play wrapper for `source/AlphaZero-adversarial-autonomous-driving/`.
- `run_meta_adversarial_self_play_dual_gpu_kaggle.sh`
  Kaggle dual-GPU self-play wrapper for `source/AlphaZero-meta-adversarial-autonomous-driving/`.
- `run_progressive_self_play_highway_dual_gpu_kaggle.sh`
  Kaggle dual-GPU self-play wrapper for the legacy `highway` progressive workflow under `legacy/AlphaZero-based-autonomous-driving/`.

All non-Kaggle workflows should call the Python entrypoints directly from the repository root with `uv run python ...`.

Workflow notebooks live under `../notebooks/`.
