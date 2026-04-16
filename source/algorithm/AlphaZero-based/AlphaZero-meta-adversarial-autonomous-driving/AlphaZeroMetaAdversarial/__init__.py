"""Meta-action adversarial AlphaZero package for simultaneous-move highway driving."""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_ROOT = _REPO_ROOT / "source"

for path in (_REPO_ROOT, _SOURCE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


__all__ = ["core", "environment", "network", "training"]
