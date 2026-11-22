"""Path anchors for the daily swing track.

ROOT is discovered by walking up from this file until config.yaml is found.
"""
from __future__ import annotations

from pathlib import Path

_THIS = Path(__file__).resolve()


def _find_root() -> Path:
    for parent in _THIS.parents:
        if (parent / "config.yaml").exists():
            return parent
    raise RuntimeError(f"config.yaml not found above {_THIS}")


ROOT = _find_root()
REV2_PROCESSED = ROOT / "data" / "rev2_processed"
REV2_RAW = ROOT / "data" / "rev2_raw"
OUTPUTS = ROOT / "outputs"
