"""Market-regime helpers and historical event-time estimation (warm-up only)."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd

from src.intraday_15m.event_detector import detect_event


def describe_regime(row) -> dict:
    return {
        "regime_score": int(row["regime_score"]),
        "regime_state": str(row["regime_state"]),
        "regime_multiplier": float(row["regime_multiplier"]),
        "qqq_daily_return": float(row["qqq_daily_return"]),
        "qqq_vwap_gap": float(row["qqq_vwap_gap"]),
        "qqq_trend_r2": None if pd.isna(row["trend_r2"]) else float(row["trend_r2"]),
        "vix_available_value": None if pd.isna(row["vix_prev_close"]) else float(row["vix_prev_close"]),
    }


def _first_50_percent_bar(frame, event_index: int, direction: str, max_bars: int = 12) -> int | None:
    day = frame.iloc[event_index]["date"]
    prev_close = float(frame.iloc[event_index]["prev_daily_close"])
    peak = float(frame.iloc[event_index]["high"])
    trough = float(frame.iloc[event_index]["low"])
    for offset in range(1, max_bars + 1):
        idx = event_index + offset
        if idx >= len(frame) or frame.iloc[idx]["date"] != day:
            break
        bar = frame.iloc[idx]
        peak = max(peak, float(bar["high"]))
        trough = min(trough, float(bar["low"]))
        if direction == "up":
            impulse = peak - prev_close
            if impulse > 0 and (peak - float(bar["close"])) / impulse >= 0.50:
                return offset
        else:
            impulse = prev_close - trough
            if impulse > 0 and (float(bar["close"]) - trough) / impulse >= 0.50:
                return offset
    return None


def estimate_expected_hold(config: dict, warmup_frames: dict, test_start) -> tuple[dict, list[dict]]:
    """Estimate expected hold from completed warm-up events only."""
    cfg = config["exit"]
    details = []
    by_group = defaultdict(list)
    for symbol, frame in warmup_frames.items():
        frame = frame[frame["date"] < test_start].sort_values("timestamp").reset_index(drop=True)
        if frame.empty:
            continue
        triggered_days = set()
        for i in range(len(frame)):
            row = frame.iloc[i]
            date = row["date"]
            if date in triggered_days:
                continue
            direction, reason = detect_event(row, config, False)
            if direction is None:
                continue
            triggered_days.add(date)
            bars = _first_50_percent_bar(frame, i, direction)
            if bars is not None:
                by_group[(symbol, direction)].append(bars)
                details.append({"symbol": symbol, "direction": direction,
                                "date": str(date), "bars_to_50pct": bars, "used": True})
            else:
                details.append({"symbol": symbol, "direction": direction,
                                "date": str(date), "bars_to_50pct": None, "used": False})

    pooled_up = [v for (_, direction), values in by_group.items() if direction == "up" for v in values]
    pooled_down = [v for (_, direction), values in by_group.items() if direction == "down" for v in values]
    pooled = {"up": float(np.median(pooled_up)) if pooled_up else 6.0,
              "down": float(np.median(pooled_down)) if pooled_down else 6.0}
    prior = float(cfg["expected_hold_prior"])
    result = {}
    for symbol in warmup_frames:
        for direction in ("up", "down"):
            sample = by_group.get((symbol, direction), [])
            n = len(sample)
            weight = n / (n + prior) if n else 0.0
            group_median = float(np.median(sample)) if n else pooled[direction]
            estimate = weight * group_median + (1 - weight) * pooled[direction]
            result[f"{symbol}_{direction}"] = {
                "n": n, "median_bars": float(np.median(sample)) if n else None,
                "pooled_median_bars": pooled[direction], "weight": round(weight, 6),
                "expected_bars": int(np.clip(round(estimate), cfg["expected_hold_floor"], cfg["expected_hold_cap"])),
            }
    return result, details
