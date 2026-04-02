"""Meta-action adversarial AlphaZero package for simultaneous-move highway driving."""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_ROOT = _REPO_ROOT / "source"

for path in (_REPO_ROOT, _SOURCE_ROOT):
    resolved = str(path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


__all__ = ["core", "environment", "network", "training"]
