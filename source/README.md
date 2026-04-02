# Source Packages

This directory holds the actively maintained code in the repository.

- `AlphaZero-adversarial-autonomous-driving/`
  Two-agent adversarial AlphaZero package.
- `AlphaZero-meta-adversarial-autonomous-driving/`
  Meta-adversarial AlphaZero package built around `DiscreteMetaAction`.
- `autonomous_driving_shared/`
  Shared core abstractions reused by both adversarial packages.
- `highway-env/`
  Local editable fork of `highway-env`.

The filesystem layout changed, but the import names did not. Continue importing `AlphaZeroAdversarial`, `AlphaZeroMetaAdversarial`, and `autonomous_driving_shared` as before.
