"""Phase 6B — Reversal 2.0 lag-safe daily panel builder.

STRICT OOS VAULT SEAL: this module NEVER reads any row with date >= 2024-06-01.
All CSVs are cut to date <= IS_END immediately on load, before any computation,
aggregation, or memory allocation. Phase 6B code cannot inspect the holdout.

ZERO network, ZERO synthetic prices. Reads ONLY data/rev2_processed/.

Every .shift() / .rolling() is computed strictly WITHIN each symbol via
groupby("symbol") — no values can leak across concatenated symbol boundaries.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.daily_swing.universe import ALPACA_SYMBOLS, STOCKS, STOCK_TO_SECTOR, BENCHMARK

from .paths import ROOT, REV2_PROCESSED, REV2_RAW, OUTPUTS  # noqa: E402
PROCESSED = ROOT / "data" / "rev2_processed"

IS_START = "2021-06-01"
IS_END = "2024-05-31"
HORIZONS = (1, 2, 3, 5)

# Warm-up begins 2020-11-01; 60d windows need ~60 trading days before IS_START.
# We keep all rows <= IS_END (including 2020-11..2021-05 warm-up) so rolling
# windows are fully seeded, but the study filters decisions to IS_START..IS_END.
VAULT_CUTOFF = pd.Timestamp(IS_END)


def _load_daily(symbol: str) -> pd.DataFrame:
    """Load one symbol's daily CSV, hard-sealed at VAULT_CUTOFF on read."""
    frame = pd.read_csv(PROCESSED / f"{symbol}_daily.csv", parse_dates=["date"])
    # HARD SEAL: drop every OOS-vault row before anything else touches the data.
    frame = frame[frame["date"] <= VAULT_CUTOFF].copy()
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame


def _close_series(symbol: str) -> pd.DataFrame:
    frame = _load_daily(symbol)[["date", "open", "close"]].rename(
        columns={"open": "px_open", "close": "px_close"}
    )
    return frame


def build_daily_panel() -> pd.DataFrame:
    """Build the unified (date, symbol) daily panel for all 48 stocks.

    Every shifted/rolling quantity is computed per symbol. ETF/benchmark series
    (sector ETF returns, SPY returns) are joined by date so they are shared
    across stocks but themselves derived only from sealed daily bars.
    """
    # --- Benchmark & sector ETF daily returns (per ETF, by date) ---
    etf_close: dict[str, pd.DataFrame] = {}
    for etf in (BENCHMARK,) + tuple(sorted(set(STOCK_TO_SECTOR.values()))):
        etf_close[etf] = _close_series(etf)

    spy_ret = etf_close[BENCHMARK][["date", "px_close"]].copy()
    spy_ret["spy_ret_1d"] = spy_ret["px_close"].pct_change()  # single series, sorted by date
    spy_ret = spy_ret[["date", "spy_ret_1d"]]
    spy_open = etf_close[BENCHMARK][["date", "px_open"]].rename(columns={"px_open": "spy_open"})

    sec_ret_map: dict[str, pd.DataFrame] = {}
    sec_open_map: dict[str, pd.DataFrame] = {}
    for etf, frame in etf_close.items():
        r = frame[["date", "px_close"]].copy()
        r[f"sec_{etf}_ret"] = r["px_close"].pct_change()
        sec_ret_map[etf] = r[["date", f"sec_{etf}_ret"]]
        o = frame[["date", "px_open"]].rename(columns={"px_open": f"sec_{etf}_open"})
        sec_open_map[etf] = o

    # --- VIX term structure: ffill(limit=2), NEVER backfill, sealed ---
    vix = _load_daily("INDEX_VIX")[["date", "close"]].rename(columns={"close": "vix_close"})
    vix3m = _load_daily("INDEX_VIX3M")[["date", "close"]].rename(columns={"close": "vix3m_close"})
    vix_term = vix.merge(vix3m, on="date", how="outer").sort_values("date").reset_index(drop=True)
    vix_term["vix_close"] = vix_term["vix_close"].ffill(limit=2)
    vix_term["vix3m_close"] = vix_term["vix3m_close"].ffill(limit=2)
    vix_term["vix_ratio"] = vix_term["vix_close"] / vix_term["vix3m_close"]
    vix_term = vix_term[["date", "vix_ratio"]]

    # --- Per-stock features, strictly grouped by symbol ---
    pieces: list[pd.DataFrame] = []
    for symbol in STOCKS:
        etf = STOCK_TO_SECTOR[symbol]
        frame = _load_daily(symbol).copy()

        g = frame.groupby("symbol", sort=False)
        frame["ret_1d"] = g["close"].pct_change()
        frame["overnight_gap"] = frame["open"] / g["close"].shift(1) - 1.0
        # daily_vol_60d strictly lagged by 1 day
        frame["daily_vol_60d"] = g["ret_1d"].transform(
            lambda s: s.rolling(60, min_periods=60).std(ddof=1).shift(1)
        )
        # rvol: volume / prior-60-day median of volume (shifted baseline)
        frame["rvol_1d"] = frame["volume"] / g["volume"].transform(
            lambda s: s.shift(1).rolling(60, min_periods=60).median()
        )

        # Join sector ETF return & SPY return by date (shared market factors)
        frame = frame.merge(sec_ret_map[etf], on="date", how="left")
        frame = frame.merge(spy_ret, on="date", how="left")
        frame = frame.merge(sec_open_map[etf], on="date", how="left")
        frame = frame.merge(spy_open, on="date", how="left")
        frame = frame.merge(vix_term, on="date", how="left")

        sec_col = f"sec_{etf}_ret"
        sec_open_col = f"sec_{etf}_open"
        frame = frame.rename(columns={sec_col: "sec_ret_1d", sec_open_col: "sec_open"})

        # STRICT per-symbol isolation: `frame` contains exactly ONE symbol here
        # (loaded via _load_daily(symbol)); the cross-symbol concat happens only
        # AFTER every rolling/shift below is finished. Operating directly on
        # this single-symbol frame therefore cannot leak across symbol boundaries.
        # Beta = rolling 60d cov / var, shifted 1 trading day, clipped.
        cov_sec = frame["ret_1d"].rolling(60, min_periods=60).cov(frame["sec_ret_1d"])
        var_sec = frame["sec_ret_1d"].rolling(60, min_periods=60).var(ddof=1)
        frame["beta_sec_60d"] = (cov_sec / var_sec).shift(1).clip(0.25, 2.50)

        cov_spy = frame["ret_1d"].rolling(60, min_periods=60).cov(frame["spy_ret_1d"])
        var_spy = frame["spy_ret_1d"].rolling(60, min_periods=60).var(ddof=1)
        frame["beta_spy_60d"] = (cov_spy / var_spy).shift(1).clip(0.25, 2.50)

        frame["res_sec_1d"] = frame["ret_1d"] - frame["beta_sec_60d"] * frame["sec_ret_1d"]
        frame["res_spy_1d"] = frame["ret_1d"] - frame["beta_spy_60d"] * frame["spy_ret_1d"]

        frame["res_sec_vol_60d"] = frame["res_sec_1d"].rolling(60, min_periods=60).std(ddof=1).shift(1)
        frame["res_spy_vol_60d"] = frame["res_spy_1d"].rolling(60, min_periods=60).std(ddof=1).shift(1)

        frame["z_res_sec_3d"] = (
            frame["res_sec_1d"].rolling(3, min_periods=3).sum()
            / (frame["res_sec_vol_60d"] * np.sqrt(3))
        )
        frame["z_res_sec_5d"] = (
            frame["res_sec_1d"].rolling(5, min_periods=5).sum()
            / (frame["res_sec_vol_60d"] * np.sqrt(5))
        )
        frame["z_res_spy_3d"] = (
            frame["res_spy_1d"].rolling(3, min_periods=3).sum()
            / (frame["res_spy_vol_60d"] * np.sqrt(3))
        )

        # Pure price/volume no-news proxy (rolling within symbol)
        frame["max_abs_gap_3d"] = frame["overnight_gap"].abs().rolling(3, min_periods=3).max()
        frame["max_rvol_3d"] = frame["rvol_1d"].rolling(3, min_periods=3).max()
        frame["no_news_proxy_ok"] = (
            (frame["max_abs_gap_3d"] <= 0.05) & (frame["max_rvol_3d"] <= 3.0)
        )

        frame["vix_regime_mult"] = np.select(
            [frame["vix_ratio"] < 0.95,
             (frame["vix_ratio"] >= 0.95) & (frame["vix_ratio"] < 1.05),
             frame["vix_ratio"] >= 1.05],
            [1.0, 0.5, 0.0],
            default=np.nan,
        )

        # --- Tradable forward returns: entry open[D+1], exit open[D+1+H] ---
        # Same single-symbol frame => shift cannot cross symbol boundaries.
        for H in HORIZONS:
            exit_open = frame["open"].shift(-(1 + H))
            entry_open = frame["open"].shift(-1)
            sec_exit = frame["sec_open"].shift(-(1 + H))
            sec_entry = frame["sec_open"].shift(-1)
            frame[f"fwd_raw_{H}"] = exit_open / entry_open - 1.0
            fwd_sec = sec_exit / sec_entry - 1.0
            frame[f"fwd_hedged_{H}"] = frame[f"fwd_raw_{H}"] - frame["beta_sec_60d"] * fwd_sec
            frame[f"target_date_{H}"] = frame["date"].shift(-(1 + H))

        frame["sector_etf"] = etf
        pieces.append(frame)

    panel = pd.concat(pieces, ignore_index=True)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    return panel


def build_is_panel() -> pd.DataFrame:
    """Return the panel with decisions hard-filtered to the IS window AND
    every forward target_date_H also <= IS_END (zero boundary leakage)."""
    panel = build_daily_panel()
    is_mask = (panel["date"] >= pd.Timestamp(IS_START)) & (panel["date"] <= pd.Timestamp(IS_END))
    out = panel[is_mask].copy()
    for H in HORIZONS:
        out = out[out[f"target_date_{H}"].isna() | (out[f"target_date_{H}"] <= pd.Timestamp(IS_END))]
    return out.reset_index(drop=True)


if __name__ == "__main__":
    panel = build_daily_panel()
    print(f"panel rows={len(panel)} max_date={panel['date'].max().date()}")
    is_panel = build_is_panel()
    print(f"IS rows={len(is_panel)} "
          f"date=[{is_panel['date'].min().date()},{is_panel['date'].max().date()}]")
    for H in HORIZONS:
        print(f"target_date_{H} max={is_panel[f'target_date_{H}'].max()}")
