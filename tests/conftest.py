from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.repo_layout import (  # noqa: E402
    ALPHAZERO_ADVERSARIAL_ROOT,
    ALPHAZERO_BASED_ROOT,
    ALPHAZERO_META_ADVERSARIAL_ROOT,
    HIGHWAY_ENV_ROOT,
    prepend_sys_path,
)


prepend_sys_path(
    HIGHWAY_ENV_ROOT,
    ALPHAZERO_ADVERSARIAL_ROOT,
    ALPHAZERO_BASED_ROOT,
    ALPHAZERO_META_ADVERSARIAL_ROOT,
)
