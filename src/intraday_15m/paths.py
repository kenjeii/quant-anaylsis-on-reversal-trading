"""Path anchors for the intraday 15m track.

ROOT is discovered by walking up from this file until config.yaml is found,
so the package works from any working directory.
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
ALPACA_PROCESSED = ROOT / "data" / "alpaca_processed"
ALPACA_RAW = ROOT / "data" / "alpaca_raw"
OUTPUTS = ROOT / "outputs"
