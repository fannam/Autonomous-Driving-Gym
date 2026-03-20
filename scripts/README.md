# Root Notebooks

This folder sits at the repository root, alongside `highway-env/`.

It contains Jupyter notebooks that wrap the maintained AlphaZero CLI scripts in:

- `AlphaZero-based-autonomous-driving/AlphaZero/scripts/`

Current notebooks:

- `01_self_play_parallel_racetrack.ipynb`
  Generate `.pt` self-play episodes with the parallel racetrack workflow.
- `02_train_from_self_play.ipynb`
  Train a checkpoint from saved self-play episodes.
- `03_loop_self_play_train.ipynb`
  Run an iterative self-play -> train loop across multiple iterations.

Open Jupyter from the repository root so the notebooks can resolve paths cleanly.
