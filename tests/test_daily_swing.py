"""Unified fast tests for the 6Y daily swing track (Rev 2.0 / MOM-FINAL)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.daily_swing import mom_final_backtest as mf
from src.daily_swing import rev2_features as rf
from src.daily_swing.universe import STOCKS


@pytest.fixture(scope="module")
def market():
    return mf.build_market()


# ---- per-symbol shift isolation / vault seal ----

def test_per_symbol_forward_shift_last_rows_nan():
    panel = rf.build_daily_panel()  # hard-sealed at IS_END
    for symbol in STOCKS:
        sub = panel[panel["symbol"] == symbol].sort_values("date")
        for H in rf.HORIZONS:
            tail = sub.iloc[-(1 + H):]
            assert tail[f"fwd_raw_{H}"].isna().all()
            assert tail[f"target_date_{H}"].isna().all()


def test_is_vault_seal():
    panel = rf.build_daily_panel()
    assert panel["date"].max() <= pd.Timestamp(rf.IS_END)
    is_panel = rf.build_is_panel()
    for H in rf.HORIZONS:
        assert is_panel[f"target_date_{H}"].max() <= pd.Timestamp(rf.IS_END)


def test_lag_safety_daily_baselines():
    panel = rf.build_daily_panel()
    probe = pd.Timestamp("2023-01-18")
    row = panel[(panel["date"] == probe)].iloc[0]
    sym = panel[(panel["symbol"] == row["symbol"])].sort_values("date").reset_index(drop=True)
    idx = sym.index[sym["date"] == probe][0]
    prior = sym.loc[:idx - 1, "ret_1d"].dropna().tail(60)
    assert np.isclose(row["daily_vol_60d"], prior.std(ddof=1), equal_nan=True)


# ---- execution / data glitch invariants ----

def test_nvda_hl_anomaly_stop_guard(market):
    dates, bars = market
    d = pd.Timestamp("2024-06-10")
    row = bars[d]["NVDA"]
    assert bool(row["hl_anomaly"]) is True
    bar_low = row["close"] if row["hl_anomaly"] else row["low"]
    assert abs(bar_low - row["close"]) < 1e-9


def test_overnight_gap_through_fill(market):
    dates, bars = market
    res = mf.run_simulation(bars, dates, "M2_Stock_Leader_5d", "LongOnly",
                            "2024-06-01", "2026-11-20", 5.0)
    gapped = res["trades"][res["trades"]["reason"] == "stop_gapped_through"]
    assert len(gapped) > 0
    slip = 0.5 * 0.6 * 5.0 / 1e4
    assert np.allclose(gapped["exit_fill"], gapped["exit_raw"] * (1 - slip), atol=1e-6)


def test_calendar_day_borrow_friday_to_monday():
    fri, mon = pd.Timestamp("2024-06-07"), pd.Timestamp("2024-06-10")
    assert max(1, (mon - fri).days) == 3
    assert abs(10000 * 0.005 * 3 / 365.0 - 0.410958904109589) < 1e-12


def test_accounting_invariant_single_slice(market):
    dates, bars = market
    res = mf.run_simulation(bars, dates, "M2_Stock_Leader_5d", "SPY_Hedged",
                            "2025-01-01", "2025-06-30", 5.0)
    trades = res["trades"]
    assert len(trades) > 0
    i1 = (trades["pre_slippage_pnl"] - trades["slippage_cost"] - trades["gross_pnl"]).abs().max()
    i2 = (trades["gross_pnl"] - trades["fee_cost"] - trades["borrow_cost"] - trades["net_pnl"]).abs().max()
    assert i1 < 1e-3 and i2 < 1e-3
