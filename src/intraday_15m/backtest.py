"""15-minute signal, next-open execution, event state machine and portfolio simulation."""
from __future__ import annotations
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.intraday_15m.event_detector import detect_event, detect_event_score, oracle_outcome
from src.intraday_15m.features import build_all
from src.intraday_15m.portfolio import Portfolio
from src.intraday_15m.regime import estimate_expected_hold

from .paths import ROOT, ALPACA_PROCESSED, ALPACA_RAW, OUTPUTS  # noqa: E402


def _slippage(price: float, atr: float, config: dict) -> float:
    costs = config["costs"]
    structural = costs["slippage_atr_fraction"] * atr / price if price > 0 else 0.0
    return max(costs["slippage_min_bps"] / 10000.0, structural)


def _fee(price: float, quantity: int, config: dict) -> float:
    return price * quantity * (config["costs"]["fee_round_trip_bps"] / 2.0 / 10000.0)


def _borrow_fee(notional: float, bars: int, config: dict) -> float:
    costs = config["costs"]
    hours = bars * 0.25
    return notional * costs["short_borrow_annual"] * hours / (
        costs["trading_days_per_year"] * costs["trading_hours_per_day"])


def _time_tuple(timestamp) -> tuple[int, int]:
    return timestamp.hour, timestamp.minute


def _is_trade_prohibited(row, trade_direction: str) -> bool:
    if trade_direction == "long":
        return bool(row["long_prohibited"])
    return bool(row["short_prohibited"])


def _trade_direction(event_direction: str, trade_style: str) -> str:
    if trade_style == "momentum":
        return "long" if event_direction == "up" else "short"
    return "short" if event_direction == "up" else "long"


def _regime_blocks(row, trade_direction: str, config: dict) -> bool:
    regime_cfg = config.get("regime", {})
    if not bool(regime_cfg.get("use_regime_multiplier", True)):
        return False
    if float(row["regime_multiplier"]) <= 0.0:
        return True
    if bool(regime_cfg.get("soft_sizing_only", False)):
        return False
    return _is_trade_prohibited(row, trade_direction)


def _missing_bar_gap(signal_timestamp, fill_timestamp) -> bool:
    try:
        return (pd.to_datetime(fill_timestamp) - pd.to_datetime(signal_timestamp)) > pd.Timedelta(minutes=15)
    except (TypeError, ValueError):
        return False


def _extension_decay(base_impulse: float, current_impulse: float, config: dict) -> float:
    if not bool(config.get("regime", {}).get("soft_sizing_only", False)):
        return 1.0
    base = float(base_impulse or 0.0)
    ext_ratio = float(current_impulse) / base if base > 0 else 1.0
    return max(0.25, 1.0 - max(0.0, ext_ratio - 1.0))


def _plan_trade(entry_price: float, peak: float, trough: float, direction: str,
                atr: float, event_price: float, prev_close: float, config: dict) -> dict:
    risk = config["risk"]
    exit_cfg = config["exit"]
    if exit_cfg.get("mode", "structural") == "triple_barrier":
        sl_dist = float(risk.get("tb_stop_atr_mult", 1.5)) * atr
        if exit_cfg.get("tb_single_exit", False):
            tp1_dist = float(risk.get("tb_tp_atr_mult", 2.0)) * atr
            tp2_dist = tp1_dist
        else:
            tp1_dist = float(risk.get("tb_tp1_atr_mult", 1.5)) * atr
            tp2_dist = float(risk.get("tb_tp2_atr_mult", 2.5)) * atr
        if direction == "short":
            stop = entry_price + sl_dist
            tp1 = entry_price - tp1_dist
            tp2 = entry_price - tp2_dist
        else:
            stop = entry_price - sl_dist
            tp1 = entry_price + tp1_dist
            tp2 = entry_price + tp2_dist
        return {"stop": stop, "tp1": tp1, "tp2": tp2, "distance": sl_dist,
                "rr1": tp1_dist / sl_dist if sl_dist > 0 else 0.0,
                "rr2": tp2_dist / sl_dist if sl_dist > 0 else 0.0,
                "impulse": 0.0, "structural_tp1": tp1, "structural_tp2": tp2}
    buffer = float(risk["stop_atr_buffer"])
    if direction == "short":
        impulse = max(peak - prev_close, 0.0)
        stop = peak + buffer * atr
        structural_tp1 = peak - float(exit_cfg["tp1_retracement"]) * impulse
        structural_tp2 = peak - float(exit_cfg["tp2_retracement"]) * impulse
    else:
        impulse = max(prev_close - trough, 0.0)
        stop = trough - buffer * atr
        structural_tp1 = trough + float(exit_cfg["tp1_retracement"]) * impulse
        structural_tp2 = trough + float(exit_cfg["tp2_retracement"]) * impulse
    distance = (stop - entry_price) if direction == "short" else (entry_price - stop)
    if distance <= 0:
        return {"stop": stop, "tp1": entry_price, "tp2": entry_price, "distance": 0.0,
                "rr1": 0.0, "rr2": 0.0, "impulse": impulse,
                "structural_tp1": structural_tp1, "structural_tp2": structural_tp2}
    if direction == "short":
        raw_tp1 = min(structural_tp1, entry_price - float(risk["tp1_r_multiple"]) * distance)
        raw_tp2 = min(structural_tp2, entry_price - float(risk["tp2_r_multiple"]) * distance)
        tp1 = max(raw_tp1, prev_close) if entry_price > prev_close else raw_tp1
        tp2 = max(raw_tp2, prev_close) if entry_price > prev_close else raw_tp2
        rr1 = (entry_price - tp1) / distance
        rr2 = (entry_price - tp2) / distance
    else:
        raw_tp1 = max(structural_tp1, entry_price + float(risk["tp1_r_multiple"]) * distance)
        raw_tp2 = max(structural_tp2, entry_price + float(risk["tp2_r_multiple"]) * distance)
        tp1 = min(raw_tp1, prev_close) if entry_price < prev_close else raw_tp1
        tp2 = min(raw_tp2, prev_close) if entry_price < prev_close else raw_tp2
        rr1 = (tp1 - entry_price) / distance
        rr2 = (tp2 - entry_price) / distance
    return {"stop": stop, "tp1": tp1, "tp2": tp2, "distance": distance,
            "rr1": rr1, "rr2": rr2, "impulse": impulse,
            "structural_tp1": structural_tp1, "structural_tp2": structural_tp2}


def _execute_exit(portfolio: Portfolio, position: dict, quantity: int, raw_price: float,
                  reason: str, timestamp, config: dict, raw_gross: float | None = None,
                  exit_slippage: float | None = None) -> None:
    if quantity <= 0:
        return
    slip = _slippage(raw_price, position["entry_atr"], config)
    direction = 1 if position["direction"] == "long" else -1
    fill = raw_price * (1 - direction * slip)
    gross = (fill - position["entry_price"]) * quantity * direction
    if raw_gross is None:
        entry_raw = position.get("entry_raw_price", position["entry_price"])
        raw_gross = (raw_price - entry_raw) * quantity * direction
    if exit_slippage is None:
        exit_slippage = abs(fill - raw_price) * quantity
    fee = _fee(fill, quantity, config)
    if position["direction"] == "short":
        fee += _borrow_fee(fill * quantity, position["holding_bars"] or 1, config)
    portfolio.register_exit(position, quantity, fill, fee, gross, reason, timestamp,
                            raw_gross=raw_gross, exit_slippage=exit_slippage)


def _process_exits(portfolio: Portfolio, position: dict, row, timestamp, config: dict) -> None:
    if position["remaining_shares"] <= 0:
        return
    risk = position["initial_risk"] / max(position["shares"], 1)
    direction = 1 if position["direction"] == "long" else -1
    favorable = (row["high"] - position["entry_price"]) if direction == 1 else \
                (position["entry_price"] - row["low"])
    adverse = (position["entry_price"] - row["low"]) if direction == 1 else \
              (row["high"] - position["entry_price"])
    position["mfe_r"] = max(position["mfe_r"], favorable / risk if risk else 0.0)
    position["mae_r"] = max(position["mae_r"], adverse / risk if risk else 0.0)
    position["holding_bars"] += 1

    stop = position["stop_price"]
    tp1, tp2 = position["tp1_price"], position["tp2_price"]
    if direction == 1:
        stop_hit = float(row["low"]) <= stop
        tp1_hit = float(row["high"]) >= tp1
        tp2_hit = float(row["high"]) >= tp2
    else:
        stop_hit = float(row["high"]) >= stop
        tp1_hit = float(row["low"]) <= tp1
        tp2_hit = float(row["low"]) <= tp2

    exit_cfg = config["exit"]
    if (exit_cfg.get("mode", "structural") == "triple_barrier" and
            exit_cfg.get("tb_single_exit", False)):
        max_hold = int(exit_cfg.get("tb_max_hold_bars", 2))
        if stop_hit:  # Conservative same-bar rule still applies.
            _execute_exit(portfolio, position, position["remaining_shares"], stop,
                          "stop", timestamp, config)
            return
        if tp1_hit or tp2_hit:
            _execute_exit(portfolio, position, position["remaining_shares"], tp1,
                          "tp_final", timestamp, config)
            return
        if position["holding_bars"] >= max_hold:
            _execute_exit(portfolio, position, position["remaining_shares"], row["close"],
                          "tb_time_barrier", timestamp, config)
            return
        force_hour, force_minute = map(int, config["project"]["force_exit"].split(":"))
        if _time_tuple(timestamp) >= (force_hour, force_minute):
            _execute_exit(portfolio, position, position["remaining_shares"], row["close"],
                          "force_exit_1545", timestamp, config)
        return

    # Conservative same-bar rule: stop wins even if a target was also touched.
    if stop_hit:
        reason = "stop" if not position["tp1_done"] else "stop_after_tp1_breakeven"
        _execute_exit(portfolio, position, position["remaining_shares"], stop, reason, timestamp, config)
        return

    half = max(1, position["shares"] // 2)
    if not position["tp1_done"] and tp1_hit:
        _execute_exit(portfolio, position, min(half, position["remaining_shares"]), tp1, "tp1", timestamp, config)
        position["tp1_done"] = True
        position["stop_price"] = position["entry_price"]
    if position["tp1_done"] and not position["tp2_done"] and tp2_hit:
        _execute_exit(portfolio, position, position["remaining_shares"], tp2, "tp2", timestamp, config)
        position["tp2_done"] = True
        return

    if config["exit"].get("mode", "structural") == "triple_barrier":
        max_hold = int(config["exit"].get("tb_max_hold_bars", 3))
        if position["holding_bars"] >= max_hold and position["remaining_shares"] > 0:
            _execute_exit(portfolio, position, position["remaining_shares"], row["close"],
                          "tb_time_barrier", timestamp, config)
            return
        force_hour, force_minute = map(int, config["project"]["force_exit"].split(":"))
        if _time_tuple(timestamp) >= (force_hour, force_minute) and position["remaining_shares"] > 0:
            _execute_exit(portfolio, position, position["remaining_shares"], row["close"],
                          "force_exit_1545", timestamp, config)
        return

    expected = position["expected_bars"]
    half_check = max(2, math.ceil(expected * 0.5))
    if (not position["time_reduced"] and position["holding_bars"] >= half_check and
            position["mfe_r"] < config["exit"]["progress_check_half_r"] and position["remaining_shares"] > 1):
        _execute_exit(portfolio, position, position["remaining_shares"] // 2, row["close"],
                      "time_no_0.25r_progress", timestamp, config)
        position["time_reduced"] = True
    if (position["holding_bars"] >= expected and
            position["mfe_r"] < config["exit"]["progress_check_full_r"]):
        _execute_exit(portfolio, position, position["remaining_shares"], row["close"],
                      "time_no_0.50r_progress", timestamp, config)
        return
    absolute_max = min(config["exit"]["absolute_max_hold_bars"], int(math.ceil(expected * 1.5)))
    if position["holding_bars"] >= absolute_max:
        _execute_exit(portfolio, position, position["remaining_shares"], row["close"],
                      "absolute_max_hold", timestamp, config)
        return
    force_hour, force_minute = map(int, config["project"]["force_exit"].split(":"))
    if _time_tuple(timestamp) >= (force_hour, force_minute):
        _execute_exit(portfolio, position, position["remaining_shares"], row["close"],
                      "force_exit_1545", timestamp, config)


def _event_feature_snapshot(row) -> dict:
    keys = ["z_daily", "z_slot", "z_car", "move_atr", "rvol", "beta", "residual", "r_day", "car", "four_bar_residual",
            "qqq_daily_return", "qqq_vwap_gap", "trend_r2", "vix_prev_close", "regime_score",
            "regime_state", "regime_multiplier"]
    out = {}
    for key in keys:
        value = row.get(key)
        if key == "regime_state":
            out[key] = str(value)
            continue
        out[key] = None if value is pd.NA or value is pd.NaT or pd.isna(value) else float(value)
    out["qqq_trend_r2"] = out.pop("trend_r2")
    out["vix_available_value"] = out.pop("vix_prev_close")
    out["regime_state"] = str(row["regime_state"])
    return out


def _new_event(row, symbol: str, direction: str, event_id: str, config: dict,
               score: int = 0, score_components: dict | None = None) -> dict:
    prev_close = float(row["prev_daily_close"])
    initial_impulse = (float(row["high"]) - prev_close) if direction == "up" else (prev_close - float(row["low"]))
    trade_style = config["event"].get("trade_style", "mean_reversion")
    event = {"event_id": event_id, "ticker": symbol, "direction": direction,
             "event_time": row["timestamp"], "event_price": float(row["close"]),
             "event_high": float(row["high"]), "event_low": float(row["low"]),
             "atr_15m": float(row["atr_15m"]), "daily_atr_prev": float(row["daily_atr_prev"]),
             "trade_style": trade_style, "trade_direction": _trade_direction(direction, trade_style),
             "event_prev_close": prev_close, "status": "armed",
             "confirmation_time": None, "entry_time": None, "rejection_reason": None,
             "primary_rejection_reason": None, "all_rejection_reasons": [],
             "event_score": score, "score_tier": _tier_for_score(score, config),
             "bars_waited": 0, "peak": float(row["high"]), "trough": float(row["low"]),
             "initial_impulse": max(initial_impulse, 0.0),
             **_event_feature_snapshot(row)}
    if score_components:
        event["score_components"] = score_components
    return event


def _tier_for_score(score: int, config: dict) -> str:
    tiers = config.get("signal_tiers", {})
    for name in ("tier_a", "tier_b", "tier_c"):
        tier = tiers.get(name)
        if tier and score >= int(tier["minimum_score"]):
            return name
    return "tier_c" if score else "none"


def _confirmation_checks(row, previous, event: dict, peak_before: float, trough_before: float,
                         config: dict, previous2=None) -> tuple[bool, str, list[str], float, dict]:
    confirmation = config["confirmation"]
    mode = confirmation.get("mode", "original")
    direction = event["direction"]
    trade_style = config["event"].get("trade_style", "mean_reversion")
    trade_dir = _trade_direction(direction, trade_style)
    prev_close = event["event_prev_close"]
    close = float(row["close"])
    is_up = direction == "up"
    impulse = (peak_before - prev_close) if is_up else (prev_close - trough_before)
    retrace = ((peak_before - close) if is_up else
               (close - trough_before)) / impulse if impulse > 0 else 0.0
    structure = (close < float(previous["low"])) if is_up else \
                (close > float(previous["high"]))
    momentum_base = previous2 if previous2 is not None else previous
    momentum = (close / float(momentum_base["close"]) - 1.0)
    momentum_ok = momentum < 0 if is_up else momentum > 0
    ema_ok = (close < float(row["ema4"])) if is_up else \
             (close > float(row["ema4"]))
    no_new_extreme = (float(row["high"]) <= peak_before) if is_up else \
                     (float(row["low"]) >= trough_before)

    checks = {"structure_break": bool(structure), "momentum": bool(momentum_ok),
              "ema": bool(ema_ok), "no_new_extreme": bool(no_new_extreme)}

    if mode == "direct":
        failed = []
        checks = {"direct_confirmation": True, **checks}
    elif mode == "momentum_fast1":
        atr = float(event.get("atr_15m", row.get("atr_15m", 0.0)))
        guard = 0.25 * atr
        event_price = float(event["event_price"])
        if is_up:
            continuation = close > event_price and float(row["low"]) >= float(event["event_low"]) - guard
        else:
            continuation = close < event_price and float(row["high"]) <= float(event["event_high"]) + guard
        checks = {"momentum_continuation": bool(continuation), "extreme_guard": bool(continuation)}
        failed = [] if continuation else ["momentum_confirmation_failed"]
    elif mode == "reversion_fast1":
        retracement_ok = retrace >= float(confirmation.get("fast_retracement", 0.15))
        candle_ok = close < float(row["open"]) if is_up else close > float(row["open"])
        checks = {"fast_retracement": bool(retracement_ok), "reversal_candle": bool(candle_ok)}
        failed = []
        if not retracement_ok:
            failed.append("fast_retracement")
        if not candle_ok:
            failed.append("reversal_candle_failed")
    elif mode == "original":
        required = retrace >= float(confirmation["retracement_fraction"])
        checks = {"retracement": bool(required), **checks}
        failed = [name for name, passed in checks.items() if not passed]
    elif mode == "two_of_three_fast" and int(event.get("event_score", 0)) >= int(confirmation["fast_score_threshold"]):
        required = (retrace >= float(confirmation["fast_retracement"]) and structure and no_new_extreme)
        checks = {"fast_retracement": bool(retrace >= float(confirmation["fast_retracement"])), **checks}
        failed = [name for name, passed in checks.items() if not passed and
                  name in {"fast_retracement", "structure_break", "no_new_extreme"}]
    else:
        required = (retrace >= float(confirmation["standard_retracement"]) and
                    sum([structure, momentum_ok, ema_ok]) >= int(confirmation["confirmation_votes_required"]))
        checks = {"standard_retracement": bool(retrace >= float(confirmation["standard_retracement"])), **checks}
        failed = []
        if retrace < float(confirmation["standard_retracement"]):
            failed.append("standard_retracement")
        votes = sum([structure, momentum_ok, ema_ok])
        if votes < int(confirmation["confirmation_votes_required"]):
            failed.append(f"only_{votes}_of_3_votes")

    if _regime_blocks(row, trade_dir, config):
        failed.append("regime_prohibition")
    latest = tuple(map(int, config["event"]["latest_confirmation_time"].split(":")))
    if _time_tuple(row["timestamp"]) > latest:
        failed.append("after_latest_confirmation")
    primary = failed[0] if failed else "ok"
    return (not failed), primary, failed, retrace, checks


def run_backtest(config: dict, make_plots: bool = True) -> dict:
    data = build_all(config)
    test_start = data["test_start"]
    frames = {}
    for symbol in config["data"]["symbols"]["trade"]:
        frame = data[symbol].copy()
        frame["ema4"] = frame["close"].ewm(span=int(config["confirmation"]["ema_length"]),
                                           adjust=False, min_periods=int(config["confirmation"]["ema_length"])).mean()
        frames[symbol] = frame
    expected_hold, hold_details = estimate_expected_hold(config, frames, test_start)

    portfolio = Portfolio(config)
    events: list[dict] = []
    trade_meta: dict[str, dict] = {}
    active_events: dict[str, dict | None] = {symbol: None for symbol in frames}
    pending_entries: list[dict] = []
    triggered_days: dict[str, set] = {symbol: set() for symbol in frames}
    cooldown_until: dict[str, int] = {symbol: -1 for symbol in frames}
    last_close = {}
    event_counter = 0
    global_index = 0
    funnel = {symbol: defaultdict(int) for symbol in frames}
    funnel["all"] = defaultdict(int)

    def mark_rejected(event: dict, primary: str, reasons: list[str] | None = None,
                      status: str = "rejected") -> None:
        all_reasons = list(dict.fromkeys(reasons or [primary]))
        event.update(status=status, rejection_reason=primary,
                     primary_rejection_reason=primary, all_rejection_reasons=all_reasons)

    def bump(stage: str, passed: bool, symbol_key: str = symbol) -> None:
        if passed:
            funnel[symbol_key][stage] += 1
            funnel["all"][stage] += 1

    timeline = []
    for symbol, frame in frames.items():
        subset = frame[frame["date"] >= test_start]
        for idx in subset.index:
            timeline.append((subset.at[idx, "timestamp"], symbol, idx))
    timeline.sort()

    for timestamp, symbol, idx in timeline:
        frame = frames[symbol]
        row = frame.iloc[idx]
        current_date = row["date"]
        last_close[symbol] = float(row["close"])
        price_map = dict(last_close)
        bump("all_15m_bars", True, symbol)

        event_cfg = config["event"]
        if event_cfg.get("mode", "score") == "score":
            stage_pass = {
                "raw_move": abs(float(row["car"])) >= float(event_cfg["minimum_abnormal_move"]),
                "daily_z": abs(float(row["z_daily"])) >= float(event_cfg["daily_z_first_level"]),
                "intraday_z": abs(float(row["z_slot"])) >= float(event_cfg["slot_z_score"]),
                "residual_z": abs(float(row["z_car"])) >= float(event_cfg["residual_z_score"]),
                "move_atr": float(row["move_atr"]) >= float(event_cfg["move_atr_score"]),
                "rvol": float(row["rvol"]) >= float(event_cfg["relative_volume_score"]),
            }
        else:
            stage_pass = {
                "raw_move": abs(float(row["r_day"])) >= float(event_cfg["minimum_raw_move"]),
                "daily_z": abs(float(row["z_daily"])) >= float(event_cfg["daily_z"]),
                "intraday_z": abs(float(row["z_slot"])) >= float(event_cfg["slot_z"]),
                "residual_z": abs(float(row["z_car"])) >= float(event_cfg["residual_z"]),
                "move_atr": float(row["move_atr"]) >= float(event_cfg["move_atr"]),
                "rvol": float(row["rvol"]) >= float(event_cfg["relative_volume"]),
            }
        for stage, passed in stage_pass.items():
            bump(stage, bool(passed), symbol)

        # 1) Open positions: manage exits from this bar's OHLC.
        for position in list(portfolio.positions):
            if position["symbol"] == symbol:
                _process_exits(portfolio, position, row, timestamp, config)

        # 2) Pending entries from prior confirmation fill at this bar's open.
        remaining_pending = []
        newly_opened = []
        for pending in pending_entries:
            if pending["symbol"] != symbol:
                remaining_pending.append(pending)
                continue
            event = pending["event"]
            signal_row = pending["signal_row"]
            if _missing_bar_gap(pending.get("signal_timestamp"), timestamp):
                mark_rejected(event, "missing_bar_gap_at_entry", ["missing_bar_gap_at_entry"],
                              status="cancelled")
                continue
            entry_raw = float(row["open"])
            slip = _slippage(entry_raw, float(signal_row["atr_15m"]), config)
            direction = pending["direction"]
            entry_price = entry_raw * (1 - slip) if direction == "short" else entry_raw * (1 + slip)
            plan = _plan_trade(entry_price, pending["peak"], pending["trough"], direction,
                               float(signal_row["atr_15m"]), event["event_price"],
                               float(event["event_prev_close"]), config)
            distance = plan["distance"]
            atr_15m = float(signal_row["atr_15m"])
            daily_atr = float(signal_row["daily_atr_prev"])
            stop_atr_15m = distance / atr_15m if atr_15m > 0 else 0.0
            stop_daily_atr = distance / daily_atr if daily_atr > 0 else 0.0
            event["proposed_stop_distance_atr"] = round(stop_atr_15m, 6)
            event["proposed_stop_distance_daily_atr"] = round(stop_daily_atr, 6)
            event["proposed_rr1"] = round(plan["rr1"], 6)
            event["proposed_rr2"] = round(plan["rr2"], 6)
            latest_entry = tuple(map(int, config["confirmation"]["latest_entry_time"].split(":")))
            entry_reasons = []
            if _time_tuple(timestamp) > latest_entry:
                entry_reasons.append("after_latest_entry_time")
            stop_ref = config["risk"].get("stop_bound_reference", "daily_atr_prev")
            if stop_ref == "daily_atr_prev":
                min_s = float(config["risk"].get("minimum_stop_daily_atr", 0.15))
                max_s = float(config["risk"].get("maximum_stop_daily_atr", 0.90))
                stop_ok = distance > 0 and (min_s <= stop_daily_atr <= max_s)
            else:
                stop_ok = distance > 0 and (
                    float(config["risk"]["minimum_stop_atr"]) <= stop_atr_15m <= float(config["risk"]["maximum_stop_atr"])
                )
            if not stop_ok:
                entry_reasons.append("stop_distance_outside_atr_bounds")
            else:
                bump("stop_distance_accepted", True, symbol)
            if plan["rr1"] < config["risk"]["minimum_tp1_rr"]:
                entry_reasons.append("rr1_below_minimum")
            if plan["rr2"] < config["risk"]["minimum_tp2_rr"]:
                entry_reasons.append("rr2_below_minimum")
            if not entry_reasons:
                bump("reward_to_risk_accepted", True, symbol)
            if entry_reasons:
                mark_rejected(event, entry_reasons[0], entry_reasons)
                continue
            equity = portfolio.mtm_equity(price_map)
            tier_multiplier = 1.0
            if event_cfg.get("mode", "score") == "score":
                tier = config["signal_tiers"].get(event.get("score_tier", "tier_c"), {})
                tier_multiplier = float(tier.get("size_multiplier", 0.5))
            current_impulse = ((pending["peak"] - float(event["event_prev_close"]))
                               if event["direction"] == "up" else
                               (float(event["event_prev_close"]) - pending["trough"]))
            decay = _extension_decay(event.get("initial_impulse", 0.0), current_impulse, config)
            use_regime = bool(config["regime"].get("use_regime_multiplier", True))
            regime_factor = float(signal_row["regime_multiplier"]) if use_regime else 1.0
            multiplier = regime_factor * tier_multiplier * decay
            event["extension_decay"] = round(decay, 6)
            event["regime_factor_used"] = round(regime_factor, 6)
            event["effective_size_multiplier"] = round(multiplier, 6)
            risk_capital = equity * config["risk"]["risk_per_trade"] * multiplier
            shares = int(math.floor(risk_capital / distance)) if distance > 0 else 0
            signal_context = dict(signal_row)
            signal_context["direction"] = direction
            ok, reason = portfolio.can_open(signal_context, shares, entry_price, plan["stop"], global_index, prices=price_map)
            if not ok:
                mark_rejected(event, reason, [reason])
                continue
            trade_id = f"{symbol}-{current_date.strftime('%Y%m%d')}-{event['event_id']}"
            hold_key = f"{symbol}_{'up' if direction == 'short' else 'down'}"
            expected = expected_hold[hold_key]["expected_bars"]
            position = {"trade_id": trade_id, "symbol": symbol, "direction": direction,
                        "event_time": event["event_time"], "event_price": event["event_price"],
                        "confirmation_time": event["confirmation_time"], "entry_time": timestamp,
                        "entry_price": entry_price, "entry_raw_price": entry_raw,
                        "entry_slippage_cost": abs(entry_price - entry_raw) * shares,
                        "peak_or_trough": pending["peak"] if direction == "short" else pending["trough"],
                        "stop_price": plan["stop"], "tp1_price": plan["tp1"], "tp2_price": plan["tp2"],
                        "shares": shares, "remaining_shares": shares, "initial_risk": shares * distance,
                        "entry_fee": _fee(entry_price, shares, config), "entry_atr": float(signal_row["atr_15m"]),
                        "expected_bars": expected, "holding_bars": 0, "mfe_r": 0.0, "mae_r": 0.0,
                        "tp1_done": False, "tp2_done": False, "time_reduced": False,
                        "exit_records": [], "confirmation_index": pending["confirmation_index"]}
            position["meta"] = {**_event_feature_snapshot(signal_row), "trade_id": trade_id,
                                "ticker": symbol, "direction": direction,
                                "event_time": event["event_time"], "event_price": event["event_price"],
                                "confirmation_time": event["confirmation_time"], "entry_time": timestamp,
                                "entry_price": entry_price, "peak_or_trough": position["peak_or_trough"],
                                "stop_price": plan["stop"], "tp1_price": plan["tp1"], "tp2_price": plan["tp2"]}
            portfolio.cash -= position["entry_fee"]
            portfolio.total_costs += position["entry_fee"]
            portfolio.add_position(position)
            newly_opened.append(position)
            trade_meta[trade_id] = position
            bump("final_entry", True, symbol)
            triggered_days[symbol].add(current_date)
            event.update(status="entered", entry_time=timestamp, entry_price=entry_price,
                         trade_id=trade_id, rejection_reason=None,
                         primary_rejection_reason=None, all_rejection_reasons=[])
        pending_entries = remaining_pending

        if config["exit"].get("monitor_entry_bar", False):
            for position in newly_opened:
                _process_exits(portfolio, position, row, timestamp, config)

        # 3) Update active event and confirmation at this bar close.
        active = active_events[symbol]
        if active is not None:
            score_mode = event_cfg.get("mode", "score") == "score"
            active["bars_waited"] += 1
            peak_before, trough_before = active["peak"], active["trough"]
            active["peak"] = max(active["peak"], float(row["high"]))
            active["trough"] = min(active["trough"], float(row["low"]))
            current_impulse = (active["peak"] - active["event_prev_close"] if active["direction"] == "up"
                               else active["event_prev_close"] - active["trough"])
            base_impulse = float(active.get("initial_impulse", 0.0))
            max_ext = float(event_cfg.get("maximum_extension_multiple", 1.5))
            extended = (base_impulse > 0.0) and (current_impulse > max_ext * base_impulse)
            pre_retrace = ((active["peak"] - float(row["close"])) if active["direction"] == "up" else
                           (float(row["close"]) - active["trough"])) / current_impulse if current_impulse > 0 else 0.0
            stale_threshold = float(event_cfg["stale_zcar"]) if score_mode else 1.0
            max_wait = int(event_cfg["maximum_event_wait_bars"])
            trade_style = event_cfg.get("trade_style", "mean_reversion")
            trade_dir = _trade_direction(active["direction"], trade_style)
            confirmation_mode = config["confirmation"].get("mode", "original")
            simple_lifecycle = confirmation_mode in {"direct", "momentum_fast1", "reversion_fast1"}
            cancel_reasons = []
            if active["bars_waited"] > max_wait:
                cancel_reasons.append(f"max_wait_{max_wait}_bars")
            if tuple(_time_tuple(timestamp)) > tuple(map(int, event_cfg["latest_confirmation_time"].split(":"))):
                cancel_reasons.append("after_latest_confirmation")
            if _regime_blocks(row, trade_dir, config):
                cancel_reasons.append("regime_prohibition")
            if not simple_lifecycle:
                if extended:
                    cancel_reasons.append("extension_beyond_1.5x")
                if abs(float(row["z_car"])) < stale_threshold:
                    cancel_reasons.append(f"zcar_below_{stale_threshold:g}")
                if score_mode and pre_retrace > float(event_cfg["maximum_preentry_reversal"]):
                    cancel_reasons.append("preentry_reversal_above_60pct")
            if cancel_reasons:
                mark_rejected(active, cancel_reasons[0], cancel_reasons, status="cancelled")
                cooldown_until[symbol] = global_index + int(event_cfg.get("cancel_cooldown_bars", 2))
                active["bars_waited"] = active["bars_waited"]
                events.append(active)
                active_events[symbol] = None
            else:
                use_regime = bool(config["regime"].get("use_regime_multiplier", True))
                regime_ok = (not use_regime) or float(row["regime_multiplier"]) > 0
                bump("regime_accepted",
                     not _regime_blocks(row, trade_dir, config) and regime_ok,
                     symbol)
                if active["bars_waited"] >= 1:
                    previous = frame.iloc[idx - 1]
                    mode = config["confirmation"].get("mode", "original")
                    mom_lookback = 3 if mode == "original" else 2
                    momentum_base = frame.iloc[idx - mom_lookback] if idx >= mom_lookback else previous
                    confirmed, primary, all_reasons, retrace, checks = _confirmation_checks(
                        row, previous, active, peak_before, trough_before, config, momentum_base)
                    retrace_key = ("retracement" if mode == "original" else
                                   "momentum_continuation" if mode == "momentum_fast1" else
                                   "fast_retracement" if mode in {"reversion_fast1", "two_of_three_fast"}
                                   else "standard_retracement")
                    bump("retracement", bool(checks.get(retrace_key)), symbol)
                    bump("structure", bool(checks.get("structure_break")), symbol)
                    bump("momentum", bool(checks.get("momentum")), symbol)
                    bump("ema", bool(checks.get("ema")), symbol)
                    if confirmed:
                        bump("confirmation", True, symbol)
                        active.update(status="confirmed", confirmation_time=timestamp,
                                      confirmation_close=float(row["close"]), retrace=retrace,
                                      peak=active["peak"], trough=active["trough"],
                                      bars_waited=active["bars_waited"],
                                      rejection_reason=None, primary_rejection_reason=None,
                                      all_rejection_reasons=[])
                        pending_entries.append({"symbol": symbol,
                                                "direction": _trade_direction(active["direction"], trade_style),
                                                "event": active, "signal_row": row, "peak": active["peak"],
                                                "trough": active["trough"], "confirmation_index": global_index,
                                                "signal_timestamp": timestamp})
                        events.append(active)
                        active_events[symbol] = None

        # 4) New Layer A event at bar close.
        if active_events[symbol] is None and current_date not in triggered_days[symbol] and global_index >= cooldown_until[symbol]:
            if event_cfg.get("mode", "score") == "score":
                direction, primary, all_reasons, score, components, hard_ok = detect_event_score(
                    row, config, False)
            else:
                direction, primary = detect_event(row, config, False)
                all_reasons, score, components, hard_ok = [primary], 0, {}, False
            if direction:
                event_counter += 1
                bump("extreme_event_formed", True, symbol)
                trade_style = event_cfg.get("trade_style", "mean_reversion")
                new_event = _new_event(row, symbol, direction,
                                       f"E{event_counter:04d}", config,
                                       score=score, score_components=components)
                if config["confirmation"].get("mode", "original") == "direct":
                    trade_dir = _trade_direction(direction, trade_style)
                    direct_reasons = []
                    if _regime_blocks(row, trade_dir, config):
                        direct_reasons.append("regime_prohibition")
                    if _time_tuple(timestamp) > tuple(map(int, event_cfg["latest_confirmation_time"].split(":"))):
                        direct_reasons.append("after_latest_confirmation")
                    if direct_reasons:
                        mark_rejected(new_event, direct_reasons[0], direct_reasons)
                        events.append(new_event)
                    else:
                        bump("confirmation", True, symbol)
                        new_event.update(status="confirmed", confirmation_time=timestamp,
                                         confirmation_close=float(row["close"]), retrace=0.0,
                                         bars_waited=0, rejection_reason=None,
                                         primary_rejection_reason=None, all_rejection_reasons=[])
                        pending_entries.append({"symbol": symbol,
                                                "direction": trade_dir,
                                                "event": new_event, "signal_row": row,
                                                "peak": new_event["peak"], "trough": new_event["trough"],
                                                "confirmation_index": global_index,
                                                "signal_timestamp": timestamp})
                        events.append(new_event)
                else:
                    active_events[symbol] = new_event

        portfolio.record_equity(timestamp, price_map)
        global_index += 1

    # Close any event that remained armed at the end of the test window.
    for symbol, event in active_events.items():
        if event is not None:
            mark_rejected(event, "end_of_test_window", ["end_of_test_window"], status="cancelled")
            events.append(event)

    trades = portfolio.final_trade_rows(trade_meta)
    equity_frame = pd.DataFrame(portfolio.equity_curve)
    return {"config": config, "data": data, "frames": frames, "expected_hold": expected_hold,
            "hold_details": hold_details, "events": events, "trades": trades, "trade_meta": trade_meta,
            "portfolio": portfolio, "equity_curve": equity_frame, "test_start": test_start,
            "funnel": {key: dict(value) for key, value in funnel.items()}}
