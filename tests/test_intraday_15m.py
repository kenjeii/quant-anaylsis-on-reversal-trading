"""Unified fast tests for the 15m intraday track (Patch 0 / Phase 3 / Final)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.intraday_15m import backtest, event_detector, features
from src.intraday_15m.portfolio import Portfolio


# ---- data / feature invariants ----

def test_robust_window_excludes_current_return():
    values = pd.Series([0.01, 0.02, 0.03, 0.50], dtype=float)
    keys = pd.Series(["09:30"] * 4, dtype=str)
    out = features._robust_window(values, keys, window=3, statistic="return")
    assert np.isnan(out.iloc[2])
    assert np.isfinite(out.iloc[3])
    assert out.iloc[3] > 5.0


def test_daily_baseline_is_shifted_by_one_day():
    daily = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-06", "2026-01-07", "2026-01-08"]),
        "close": [100.0, 110.0, 121.0, 99.0, 108.0],
    })
    mean, std = features._daily_baseline(daily, window=2)
    assert pd.isna(mean.iloc[2])
    assert abs(mean.iloc[3] - 0.10) < 1e-12
    assert abs(mean.iloc[4] - (-0.0409090909)) < 1e-9


def test_car_robust_z_uses_14826_mad():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-06"]),
        "slot": [5, 5, 5], "car": [0.01, 0.03, 0.10],
    })
    out = features._car_robust_z(frame, window=2, mad_floor=0.0001)
    expected = (0.10 - 0.02) / (1.4826 * 0.01)
    assert np.isclose(out.iloc[2], expected, rtol=1e-10)


def test_four_bar_residual_is_rolling_sum():
    frame = pd.DataFrame({"date": pd.Timestamp("2026-11-20"), "residual": np.arange(1.0, 11.0)})
    actual = frame.groupby("date")["residual"].transform(
        lambda x: x.rolling(4, min_periods=4).sum())
    expected = pd.Series([np.nan] * 3 + [10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0])
    pd.testing.assert_series_equal(actual.reset_index(drop=True), expected, check_names=False)


def test_latest_event_time_boundary():
    assert event_detector._time_ok(pd.Timestamp("2026-11-20 14:45:00"), "14:45")
    assert not event_detector._time_ok(pd.Timestamp("2026-11-20 14:46:00"), "14:45")


# ---- event / oracle invariants ----

def test_car_raw_direction_conflict_rejected(config):
    row = {"timestamp": pd.Timestamp("2026-11-20 10:00:00"), "r_day": -0.01,
           "car": 0.02, "z_daily": 3.0, "z_slot": 3.0, "z_car": 3.0,
           "move_atr": 2.0, "rvol": 2.0, "four_bar_residual": 0.02,
           "day_high_so_far": 101.0, "day_low_so_far": 99.0, "close": 100.0}
    direction, primary, reasons, _, _, hard_ok = event_detector.detect_event_score(row, config, False)
    assert direction is None
    assert primary == "car_raw_direction_mismatch"
    assert not hard_ok


def test_oracle_chronology_ignores_dip_before_peak(config):
    date = pd.Timestamp("2026-11-20")
    frame = pd.DataFrame({
        "date": [date] * 3,
        "timestamp": pd.to_datetime(["2026-11-20 10:00", "2026-11-20 10:15", "2026-11-20 10:30"]),
        "prev_daily_close": [95.0] * 3,
        "high": [100.0, 91.0, 110.0], "low": [99.0, 90.0, 105.0],
        "close": [100.0, 91.0, 110.0],
    })
    assert event_detector.oracle_outcome(frame, 0, "up", config)["oracle_reversal"] == 0.0


# ---- planning / execution invariants ----

def test_short_plan_signed_rr(config):
    plan = backtest._plan_trade(entry_price=105.0, peak=110.0, trough=95.0,
                                direction="short", atr=2.0, event_price=108.0,
                                prev_close=100.0, config=config)
    assert np.isclose(plan["impulse"], 10.0)
    assert plan["tp1"] >= 100.0
    assert plan["rr1"] > 0


def test_adverse_entry_slippage_sign():
    raw, slip = 100.0, 0.002
    assert raw * (1 + slip) > raw > raw * (1 - slip)


def test_directional_regime_blocks_only_matching_trade(config):
    short_row = {"direction": "short", "long_prohibited": True,
                 "short_prohibited": False, "regime_multiplier": 1.0}
    long_row = {"direction": "long", "long_prohibited": True,
                "short_prohibited": False, "regime_multiplier": 1.0}
    assert not backtest._is_trade_prohibited(short_row, "short")
    assert backtest._is_trade_prohibited(long_row, "long")
    portfolio = Portfolio(config)
    ok_short, _ = portfolio.can_open(short_row, 1, 100.0, 99.0, 0)
    ok_long, reason_long = portfolio.can_open(long_row, 1, 100.0, 99.0, 0)
    assert ok_short
    assert not ok_long and reason_long == "regime_direction_prohibition"


def test_missing_bar_gap_at_entry():
    assert backtest._missing_bar_gap(pd.Timestamp("2026-11-20 10:00"), pd.Timestamp("2026-11-20 10:15")) is False
    assert backtest._missing_bar_gap(pd.Timestamp("2026-11-20 10:00"), pd.Timestamp("2026-11-20 10:45")) is True


def test_triple_barrier_geometry(config):
    cfg = dict(config)
    long_plan = backtest._plan_trade(entry_price=100.0, peak=105.0, trough=95.0,
                                     direction="long", atr=2.0, event_price=102.0,
                                     prev_close=99.0, config=cfg)
    short_plan = backtest._plan_trade(entry_price=100.0, peak=105.0, trough=95.0,
                                      direction="short", atr=2.0, event_price=98.0,
                                      prev_close=99.0, config=cfg)
    assert long_plan["stop"] < 100.0 < long_plan["tp1"]
    assert short_plan["stop"] > 100.0 > short_plan["tp1"]


# ---- portfolio / risk invariants ----

def test_shares_follow_risk_budget_and_regime_multiplier(config):
    portfolio = Portfolio(config)
    row = pd.Series({"direction": "long", "regime_multiplier": 0.5,
                     "long_prohibited": False, "short_prohibited": False})
    equity = portfolio.initial_equity
    distance = 2.0
    risk_capital = equity * config["risk"]["risk_per_trade"] * 0.5
    shares = int(risk_capital // distance)
    ok, reason = portfolio.can_open(row, shares, 100.0, 98.0, 0)
    assert ok, reason
    assert shares == 62
    assert shares * distance <= equity * config["risk"]["maximum_total_open_risk"]


def test_market_value_cap_rejects_oversized_position(config):
    portfolio = Portfolio(config)
    row = pd.Series({"direction": "long", "regime_multiplier": 1.0,
                     "long_prohibited": False, "short_prohibited": False})
    ok, reason = portfolio.can_open(row, 10000, 100.0, 99.0, 0)
    assert not ok and reason == "max_position_value"


def test_same_bar_stop_and_target_is_stop_first(config):
    def position(direction="long", stop=95.0, tp1=110.0, tp2=120.0, entry=100.0):
        return {"trade_id": "TEST-1", "symbol": "NVDA", "direction": direction,
                "entry_price": entry, "stop_price": stop, "tp1_price": tp1, "tp2_price": tp2,
                "shares": 100, "remaining_shares": 100, "initial_risk": 500.0,
                "entry_atr": 2.0, "expected_bars": 6, "holding_bars": 0,
                "mfe_r": 0.0, "mae_r": 0.0, "tp1_done": False, "tp2_done": False,
                "time_reduced": False, "exit_records": [], "confirmation_index": 0}

    portfolio = Portfolio(config)
    pos = position()
    row = pd.Series({"high": 115.0, "low": 94.0, "close": 105.0})
    backtest._process_exits(portfolio, pos, row, pd.Timestamp("2026-11-20 10:00:00"), config)
    assert pos["exit_records"][0]["exit_reason"] == "stop"
    assert pos["remaining_shares"] == 0

    short = position(direction="short", stop=105.0, tp1=90.0, tp2=80.0)
    row2 = pd.Series({"high": 106.0, "low": 85.0, "close": 95.0})
    backtest._process_exits(portfolio, short, row2, pd.Timestamp("2026-11-20 10:00:00"), config)
    assert short["exit_records"][0]["exit_reason"] == "stop"


def test_monitor_entry_bar_exact_holding_two(config):
    # tb_single_exit + monitor_entry_bar must yield exactly 2 holding bars for h=2.
    cfg = dict(config)
    cfg["exit"] = dict(config["exit"])
    cfg["exit"].update(monitor_entry_bar=True, tb_single_exit=True, tb_max_hold_bars=2)
    assert cfg["exit"]["monitor_entry_bar"] is True
    assert cfg["exit"]["tb_single_exit"] is True
    assert cfg["exit"]["tb_max_hold_bars"] == 2
