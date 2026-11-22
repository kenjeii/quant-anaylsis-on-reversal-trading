"""Layer A: extreme-event detection. Thresholds come only from config.yaml."""
from __future__ import annotations
import math


def _time_ok(timestamp, latest: str, earliest: str | None = None) -> bool:
    if earliest is None:
        earliest = "09:30"
    start_hour, start_minute = map(int, earliest.split(":"))
    end_hour, end_minute = map(int, latest.split(":"))
    current = (timestamp.hour, timestamp.minute)
    return (start_hour, start_minute) <= current <= (end_hour, end_minute)


def detect_event(row, config: dict, already_triggered_today: bool = False) -> tuple[str | None, str]:
    """Return (direction, rejection_reason). No trading decision is made here."""
    event = config["event"]
    reasons = []
    if already_triggered_today:
        return None, "one_event_per_symbol_per_day"
    latest = event["latest_event_time"]
    earliest = event.get("earliest_event_time", "09:30")
    if not _time_ok(row["timestamp"], latest, earliest):
        current = (row["timestamp"].hour, row["timestamp"].minute)
        end = tuple(map(int, latest.split(":")))
        reason = "after_latest_event_time" if current > end else "before_earliest_event_time"
        return None, reason

    raw_move = float(row["r_day"])
    min_move = float(event["minimum_raw_move"])
    up = raw_move >= min_move
    down = raw_move <= -min_move
    if not (up or down):
        return None, "raw_move_below_threshold"

    direction = "up" if up else "down"
    z_daily = float(row["z_daily"])
    z_slot = float(row["z_slot"])
    z_car = float(row["z_car"])
    four_bar = float(row["four_bar_return"])
    move_atr = float(row["move_atr"])
    rvol = float(row["rvol"])

    checks = {
        "daily_z": z_daily >= event["daily_z"] if up else z_daily <= -event["daily_z"],
        "slot_or_momentum": (z_slot >= event["slot_z"] or four_bar >= event["four_bar_move"])
        if up else (z_slot <= -event["slot_z"] or four_bar <= -event["four_bar_move"]),
        "residual_z": z_car >= event["residual_z"] if up else z_car <= -event["residual_z"],
        "move_atr": move_atr >= event["move_atr"],
        "rvol": rvol >= event["relative_volume"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        return None, "failed:" + ",".join(failed)
    if any(not math.isfinite(float(row[key])) for key in
           ("z_daily", "z_slot", "z_car", "move_atr", "rvol")):
        return None, "non_finite_feature"
    return direction, "ok"


def detect_event_score(row, config: dict, already_triggered_today: bool = False) -> tuple:
    """Score-based Layer A detector. Returns direction, reasons, score and component detail."""
    event = config["event"]
    if already_triggered_today:
        return None, "one_event_per_symbol_per_day", ["one_event_per_symbol_per_day"], 0, {}, False
    latest = event["latest_event_time"]
    earliest = event.get("earliest_event_time", "09:30")
    if not _time_ok(row["timestamp"], latest, earliest):
        current = (row["timestamp"].hour, row["timestamp"].minute)
        end = tuple(map(int, latest.split(":")))
        reason = "after_latest_event_time" if current > end else "before_earliest_event_time"
        return None, reason, [reason], 0, {}, False

    keys = ("r_day", "car", "z_daily", "z_slot", "z_car", "move_atr", "rvol",
            "four_bar_residual", "day_high_so_far", "day_low_so_far")
    if any(key not in row or pd_isna(row[key]) for key in keys):
        return None, "non_finite_feature", ["non_finite_feature"], 0, {}, False

    car = float(row["car"])
    r_day = float(row["r_day"])
    direction = "up" if car > 0 else "down"
    reasons = []
    hard_ok = True
    if (direction == "up" and r_day <= 0.0) or (direction == "down" and r_day >= 0.0):
        reasons.append("car_raw_direction_mismatch")
        hard_ok = False
    if abs(car) < float(event["minimum_abnormal_move"]):
        reasons.append("abnormal_move_below_1.25pct")
        hard_ok = False
    volatility_extreme = (
        abs(float(row["z_daily"])) >= float(event["daily_z_first_level"]) or
        abs(float(row["z_slot"])) >= float(event["slot_z_score"]) or
        float(row["move_atr"]) >= float(event["move_atr_score"]) or
        abs(float(row["four_bar_residual"])) >= float(event["four_bar_residual_move"]))
    if not volatility_extreme:
        reasons.append("no_volatility_extreme_condition")
        hard_ok = False

    sign = 1.0 if direction == "up" else -1.0
    z_daily = float(row["z_daily"])
    z_slot = float(row["z_slot"])
    z_car = float(row["z_car"])
    move_atr = float(row["move_atr"])
    rvol = float(row["rvol"])
    four_bar_resid = float(row["four_bar_residual"])
    day_range = float(row["day_high_so_far"]) - float(row["day_low_so_far"])
    extreme_zone = (float(row["close"]) >= float(row["day_high_so_far"]) -
                    float(event["extreme_zone_fraction"]) * day_range) if direction == "up" else \
                   (float(row["close"]) <= float(row["day_low_so_far"]) +
                    float(event["extreme_zone_fraction"]) * day_range)

    components = {
        "daily_z_first": abs(z_daily) >= float(event["daily_z_first_level"]),
        "daily_z_second_bonus": abs(z_daily) >= float(event["daily_z_second_level"]),
        "slot_z": abs(z_slot) >= float(event["slot_z_score"]),
        "residual_z": sign * z_car >= float(event["residual_z_score"]),
        "move_atr": move_atr >= float(event["move_atr_score"]),
        "relative_volume": rvol >= float(event["relative_volume_score"]),
        "four_bar_residual": sign * four_bar_resid >= float(event["four_bar_residual_move"]),
        "extreme_zone": bool(extreme_zone),
    }
    score = int(sum(components.values()))
    for name, passed in components.items():
        if not passed:
            reasons.append("score:" + name)
    if not hard_ok:
        primary = reasons[0]
    elif score < int(event["minimum_event_score"]):
        reasons.append(f"score_below_{event['minimum_event_score']}")
        primary = "score_below_minimum"
    else:
        primary = "ok"
    return direction if primary == "ok" else None, primary, reasons, score, components, hard_ok


def pd_isna(value) -> bool:
    try:
        import pandas as pd
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def oracle_outcome(frame, event_index: int, direction: str, config: dict) -> dict:
    """Post-hoc reversal label; never returned to the strategy."""
    window = int(config["oracle"]["forward_window_bars"])
    day = frame.iloc[event_index]["date"]
    segment = frame.iloc[event_index:event_index + 1 + window]
    segment = segment[segment["date"] == day]
    prev_close = float(frame.iloc[event_index]["prev_daily_close"])

    highs = segment["high"].to_numpy(dtype=float)
    lows = segment["low"].to_numpy(dtype=float)
    peak = float(highs.max())
    trough = float(lows.min())

    if direction == "up":
        peak_pos = int(highs.argmax())
        after = segment.iloc[peak_pos + 1:]
        denominator = peak - prev_close
        min_after = float(after["low"].min()) if len(after) else peak
        score = (peak - min_after) / denominator if denominator > 0 else 0.0
    else:
        trough_pos = int(lows.argmin())
        after = segment.iloc[trough_pos + 1:]
        denominator = prev_close - trough
        max_after = float(after["high"].max()) if len(after) else trough
        score = (max_after - trough) / denominator if denominator > 0 else 0.0

    true_cut = float(config["oracle"]["true_reversal"])
    partial_cut = float(config["oracle"]["partial_reversal"])
    label = "true" if score >= true_cut else "partial" if score >= partial_cut else "none"
    return {"oracle_reversal": round(score, 6), "reversal_label": label,
            "oracle_peak": round(peak, 6), "oracle_trough": round(trough, 6)}
