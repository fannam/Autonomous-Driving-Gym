"""Traditional PPO autonomous driving package."""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[5]
_SOURCE_ROOT = _REPO_ROOT / "source"
_HIGHWAY_ENV_ROOT = _SOURCE_ROOT / "highway-env"

for path in (_REPO_ROOT, _SOURCE_ROOT, _HIGHWAY_ENV_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


__all__ = ["core", "environment", "network", "training"]
