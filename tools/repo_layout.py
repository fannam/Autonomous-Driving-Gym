from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source"
LEGACY_ROOT = REPO_ROOT / "legacy"
HIGHWAY_ENV_ROOT = SOURCE_ROOT / "highway-env"
AUTONOMOUS_DRIVING_SHARED_ROOT = SOURCE_ROOT / "autonomous_driving_shared"
ALPHAZERO_BASED_ROOT = LEGACY_ROOT / "AlphaZero-based-autonomous-driving"
ALPHAZERO_ADVERSARIAL_ROOT = SOURCE_ROOT / "AlphaZero-adversarial-autonomous-driving"
ALPHAZERO_META_ADVERSARIAL_ROOT = SOURCE_ROOT / "AlphaZero-meta-adversarial-autonomous-driving"

SCRIPTS_ROOT = REPO_ROOT / "scripts"
TOOLS_ROOT = REPO_ROOT / "tools"
DOCS_ROOT = REPO_ROOT / "docs"
PLANS_ROOT = DOCS_ROOT / "plans"
NOTEBOOKS_ROOT = REPO_ROOT / "notebooks"
WORKFLOW_NOTEBOOKS_ROOT = NOTEBOOKS_ROOT / "workflows"
ANALYSIS_NOTEBOOKS_ROOT = NOTEBOOKS_ROOT / "analysis"

PROJECT_ROOTS = {
    "alphazero_based": ALPHAZERO_BASED_ROOT,
    "alphazero_adversarial": ALPHAZERO_ADVERSARIAL_ROOT,
    "alphazero_meta_adversarial": ALPHAZERO_META_ADVERSARIAL_ROOT,
    "autonomous_driving_shared": AUTONOMOUS_DRIVING_SHARED_ROOT,
    "highway_env": HIGHWAY_ENV_ROOT,
}


def prepend_sys_path(*paths: Path) -> None:
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
