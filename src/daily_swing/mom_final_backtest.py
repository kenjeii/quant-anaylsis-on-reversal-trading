"""Phase 7 — MOM-FINAL clean multi-day daily portfolio simulator.

100% OFFLINE. Reads ONLY data/rev2_processed/. ZERO parameter tuning.
Pre-registered frozen rules M1/M2/M3 × LongOnly/SPY_Hedged.

All rolling/shift features are computed on each single-symbol frame BEFORE
concat, so no value can leak across symbol boundaries. Volatility is computed
strictly close-to-close; hl_anomaly bars use close instead of low/high for
intraday stop checks so a single bad tick never triggers a phantom stop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.daily_swing.universe import SECTOR_ETFS, SECTOR_MAP, STOCKS, STOCK_TO_SECTOR, BENCHMARK

from .paths import ROOT, REV2_PROCESSED, REV2_RAW, OUTPUTS  # noqa: E402
PROCESSED = ROOT / "data" / "rev2_processed"

FULL_START = "2020-11-02"
FULL_END = "2026-11-20"
H_BARS = 5
STOP_VOL_MULT = 3.0
INITIAL_EQUITY = 100_000.0
SHORT_BORROW_ANNUAL = 0.005

# Frozen pre-registered strategies (ZERO tuning).
STRATEGIES: dict[str, dict] = {
    "M1_Sector_ETF_5d": dict(
        universe="etf", z=1.0, sector_conf=False, rvol_conf=False,
        max_new=2, max_pos=4, max_sector=999, max_weight=0.25, risk=0.010),
    "M2_Stock_Leader_5d": dict(
        universe="stock", z=1.5, sector_conf=False, rvol_conf=False,
        max_new=3, max_pos=8, max_sector=2, max_weight=0.125, risk=0.0075),
    "M3_Synth_Sector_Vol_5d": dict(
        universe="stock", z=1.5, sector_conf=True, rvol_conf=True,
        max_new=3, max_pos=8, max_sector=2, max_weight=0.125, risk=0.0075),
}
MODES = ("LongOnly", "SPY_Hedged")


def _load(symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED / f"{symbol}_daily.csv", parse_dates=["date"])
    frame = frame[(frame["date"] >= FULL_START) & (frame["date"] <= FULL_END)]
    return frame.sort_values("date").reset_index(drop=True)


def _build_symbol_frame(symbol: str, spy_ret1: pd.Series, spy_ret3: pd.Series,
                        sec_ret3_map: dict[str, pd.Series]) -> pd.DataFrame:
    """Compute all features strictly on this one symbol's frame."""
    frame = _load(symbol).copy()
    frame["ret_1d"] = frame["close"].pct_change()
    frame["ret_3d"] = frame["close"].pct_change(3)
    frame["daily_vol_60d"] = frame["ret_1d"].rolling(60, min_periods=60).std(ddof=1).shift(1)
    frame["rvol_1d"] = frame["volume"] / frame["volume"].shift(1).rolling(60, min_periods=60).median()
    frame["max_rvol_3d"] = frame["rvol_1d"].rolling(3, min_periods=3).max()
    frame["hl_anomaly"] = (
        (frame["high"] > frame[["open", "close"]].max(axis=1) * 1.25)
        | (frame["low"] < frame[["open", "close"]].min(axis=1) * 0.75)
    )
    frame = frame.set_index("date")
    frame["spy_ret_1d"] = spy_ret1
    frame["spy_ret_3d"] = spy_ret3
    beta = (frame["ret_1d"].rolling(60, min_periods=60).cov(frame["spy_ret_1d"])
            / frame["spy_ret_1d"].rolling(60, min_periods=60).var(ddof=1)).shift(1).clip(0.25, 2.50)
    frame["beta_spy_60d"] = beta
    frame["res_spy_1d"] = frame["ret_1d"] - beta * frame["spy_ret_1d"]
    res_vol = frame["res_spy_1d"].rolling(60, min_periods=60).std(ddof=1).shift(1)
    frame["res_spy_vol_60d"] = res_vol
    frame["cum_res_spy_3d"] = frame["res_spy_1d"].rolling(3, min_periods=3).sum()
    frame["z_res_spy_3d"] = frame["cum_res_spy_3d"] / (res_vol * math.sqrt(3.0))
    sec3 = sec_ret3_map.get(symbol)
    frame["sec_ret_3d"] = sec3 if sec3 is not None else np.nan
    frame["sector_rel_3d"] = frame["sec_ret_3d"] - frame["spy_ret_3d"]
    frame["sector_etf"] = STOCK_TO_SECTOR.get(symbol, symbol)
    return frame.reset_index()


def build_market() -> tuple[list[pd.Timestamp], dict[pd.Timestamp, dict[str, dict]]]:
    """Return (trading_dates, bars) where bars[date][symbol] is a row dict."""
    spy = _load(BENCHMARK).set_index("date")
    spy_ret1 = spy["close"].pct_change()
    spy_ret3 = spy["close"].pct_change(3)

    sec_ret3_map: dict[str, pd.Series] = {}
    for etf, members in SECTOR_MAP.items():
        e = _load(etf).set_index("date")["close"].pct_change(3)
        for sym in members:
            sec_ret3_map[sym] = e
    for etf in SECTOR_ETFS:
        sec_ret3_map[etf] = spy["close"].pct_change(3) * 0  # ETFs: sector_rel not used

    vix = _load("INDEX_VIX").set_index("date")["close"]
    vix3m = _load("INDEX_VIX3M").set_index("date")["close"]
    vix_ratio = (vix / vix3m).reindex(spy.index).ffill(limit=2)

    all_symbols = [BENCHMARK, *SECTOR_ETFS, *STOCKS]
    frames = []
    for sym in all_symbols:
        f = _build_symbol_frame(sym, spy_ret1, spy_ret3, sec_ret3_map)
        f["vix_ratio"] = f["date"].map(vix_ratio)
        f["vix_mult"] = np.select(
            [f["vix_ratio"] < 0.95, f["vix_ratio"] < 1.05, f["vix_ratio"] >= 1.05],
            [1.0, 0.5, 0.0], default=np.nan)
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)

    dates = sorted(panel["date"].unique())
    bars: dict[pd.Timestamp, dict[str, dict]] = {}
    for date, group in panel.groupby("date"):
        bars[date] = {row["symbol"]: row.to_dict() for _, row in group.iterrows()}
    return dates, bars


@dataclass
class Position:
    symbol: str
    sector: str
    shares: int
    entry_date: pd.Timestamp
    entry_raw: float
    entry_fill: float
    stop_price: float
    holding_days: int
    beta: float
    hedge_shares: int = 0
    hedge_entry_raw: float = 0.0
    hedge_entry_fill: float = 0.0


def run_simulation(bars: dict, dates: list[pd.Timestamp], strategy: str, mode: str,
                   start: str, end: str, total_rt_bps: float) -> dict:
    """Run one frozen simulation. Returns trades, equity curve, and diagnostics."""
    cfg = STRATEGIES[strategy]
    slip = 0.5 * 0.6 * total_rt_bps / 1e4   # per side
    fee = 0.5 * 0.4 * total_rt_bps / 1e4    # per side
    hedged = mode == "SPY_Hedged"
    universe = SECTOR_ETFS if cfg["universe"] == "etf" else STOCKS

    window = [d for d in dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    cash = INITIAL_EQUITY
    equity = INITIAL_EQUITY
    positions: list[Position] = []
    pending: list[dict] = []
    trades: list[dict] = []
    curve: list[dict] = []

    def close_position(p: Position, exit_raw: float, reason: str,
                       exit_date: pd.Timestamp, hedge_exit_raw: float | None = None) -> None:
        nonlocal cash
        exit_fill = exit_raw * (1.0 - slip)
        pre_long = (exit_raw - p.entry_raw) * p.shares
        slip_long = (p.entry_raw * slip + exit_raw * slip) * p.shares
        gross_long = pre_long - slip_long
        fee_long = (p.entry_fill + exit_fill) * p.shares * fee
        cash += p.shares * exit_fill - p.shares * exit_fill * fee

        pre_short = slip_short = gross_short = fee_short = borrow = 0.0
        if hedged and p.hedge_shares:
            hraw = hedge_exit_raw if hedge_exit_raw is not None else bars[exit_date][BENCHMARK]["open"]
            hfill = hraw * (1.0 + slip)
            pre_short = (p.hedge_entry_raw - hraw) * p.hedge_shares
            slip_short = (p.hedge_entry_raw * slip + hraw * slip) * p.hedge_shares
            gross_short = pre_short - slip_short
            fee_short = (p.hedge_entry_fill + hfill) * p.hedge_shares * fee
            cal_days = max(1, (exit_date - p.entry_date).days)
            borrow = p.hedge_entry_fill * p.hedge_shares * SHORT_BORROW_ANNUAL * cal_days / 365.0
            cash -= p.hedge_shares * hfill + p.hedge_shares * hfill * fee + borrow

        pre_slip = pre_long + pre_short
        slip_cost = slip_long + slip_short
        gross = gross_long + gross_short
        fees = fee_long + fee_short
        net = gross - fees - borrow
        assert abs((pre_slip - slip_cost) - gross) < 1e-3, (strategy, mode, reason, pre_slip, slip_cost, gross)
        assert abs((gross - fees - borrow) - net) < 1e-3, (strategy, mode, reason)
        trades.append({
            "strategy": strategy, "mode": mode, "symbol": p.symbol, "sector": p.sector,
            "entry_date": p.entry_date, "exit_date": exit_date,
            "holding_days": p.holding_days, "reason": reason,
            "entry_raw": p.entry_raw, "entry_fill": p.entry_fill,
            "exit_raw": exit_raw, "exit_fill": exit_fill,
            "shares": p.shares, "hedge_shares": p.hedge_shares, "beta": p.beta,
            "pre_slippage_pnl": pre_slip, "slippage_cost": slip_cost,
            "gross_pnl": gross, "fee_cost": fees, "borrow_cost": borrow, "net_pnl": net,
            "long_notional": p.entry_raw * p.shares,
        })

    for t in window:
        today = bars[t]

        # 1) EXITS AT OPEN
        for p in list(positions):
            op = today[p.symbol]["open"]
            reason = None
            exit_raw = None
            if p.holding_days >= H_BARS:
                reason, exit_raw = "time_barrier", op
            elif op <= p.stop_price:
                reason, exit_raw = "stop_gapped_through", op
            if reason:
                close_position(p, exit_raw, reason, t, today[BENCHMARK]["open"])
                positions.remove(p)

        # 2) NEW ENTRIES AT OPEN (from signals queued at prior close)
        entered_today = 0
        for sig in pending:
            if entered_today >= cfg["max_new"] or len(positions) >= cfg["max_pos"]:
                break
            sym = sig["symbol"]
            if any(p.symbol == sym for p in positions):
                continue
            if sym not in today:
                continue
            sec = sig["sector"]
            if sum(1 for p in positions if p.sector == sec) >= cfg["max_sector"]:
                continue
            srow = sig["sig_row"]
            vol, scl, vmult, beta = (srow["daily_vol_60d"], srow["close"],
                                     srow["vix_mult"], srow["beta_spy_60d"])
            if not np.isfinite(vol) or vol <= 0 or not np.isfinite(scl) or scl <= 0:
                continue
            if not np.isfinite(vmult) or vmult <= 0 or not np.isfinite(beta):
                continue
            stop_dist = STOP_VOL_MULT * vol * scl
            if stop_dist <= 0:
                continue
            target_risk = equity * cfg["risk"] * vmult
            shares = math.floor(min(target_risk / stop_dist,
                                    (equity * cfg["max_weight"] * vmult) / scl))
            entry_raw = today[sym]["open"]
            if shares <= 0 or entry_raw <= 0:
                continue
            if shares * entry_raw > equity * cfg["max_weight"]:
                shares = math.floor(equity * cfg["max_weight"] / entry_raw)
            gross_long_now = sum(p.shares * today[p.symbol]["open"] for p in positions)
            remaining = equity - gross_long_now
            if remaining < 1000:
                continue
            while shares > 0 and shares * entry_raw > remaining:
                shares -= 1
            if shares <= 0:
                continue

            entry_fill = entry_raw * (1.0 + slip)
            cash -= shares * entry_fill + shares * entry_fill * fee
            hs = heraw = hefill = 0
            if hedged:
                spy_open = today[BENCHMARK]["open"]
                hs = int(round(shares * entry_raw * beta / spy_open)) if spy_open > 0 else 0
                if hs > 0:
                    heraw = spy_open
                    hefill = heraw * (1.0 - slip)
                    cash += hs * hefill - hs * hefill * fee
            pos = Position(symbol=sym, sector=sec, shares=shares, entry_date=t,
                           entry_raw=entry_raw, entry_fill=entry_fill,
                           stop_price=entry_raw - STOP_VOL_MULT * vol * entry_raw,
                           holding_days=0, beta=float(beta), hedge_shares=hs,
                           hedge_entry_raw=heraw, hedge_entry_fill=hefill)
            positions.append(pos)
            entered_today += 1

        # 3) INTRADAY STOP CHECK (including newly entered, conservative)
        for p in list(positions):
            bar = today[p.symbol]
            bad = bool(bar["hl_anomaly"])
            bar_low = bar["close"] if bad else bar["low"]
            if np.isfinite(bar_low) and bar_low <= p.stop_price:
                close_position(p, p.stop_price, "stop_intraday", t,
                               today[BENCHMARK]["close"])
                positions.remove(p)
            else:
                p.holding_days += 1

        # 4) MARK-TO-MARKET AT CLOSE
        mtm = cash
        for p in positions:
            mtm += p.shares * today[p.symbol]["close"]
            if hedged and p.hedge_shares:
                mtm -= p.hedge_shares * today[BENCHMARK]["close"]
        equity = mtm if np.isfinite(mtm) else equity
        gross_exp = (sum(p.shares * today[p.symbol]["close"] for p in positions)
                     / equity * 100.0) if equity > 0 else 0.0
        curve.append({"date": t, "equity": equity, "gross_exposure_pct": gross_exp,
                      "n_open": len(positions)})

        # 5) QUEUE SIGNALS AT CLOSE for next open
        if t < window[-1]:
            cands = []
            for sym in universe:
                row = today.get(sym)
                if row is None:
                    continue
                z = row["z_res_spy_3d"]
                if not np.isfinite(z) or z < cfg["z"]:
                    continue
                if not (row["vix_mult"] > 0):
                    continue
                if any(p.symbol == sym for p in positions):
                    continue
                if cfg["sector_conf"] and not (np.isfinite(row["sector_rel_3d"])
                                               and row["sector_rel_3d"] > 0):
                    continue
                if cfg["rvol_conf"] and not (np.isfinite(row["max_rvol_3d"])
                                             and row["max_rvol_3d"] >= 1.3):
                    continue
                cands.append({"symbol": sym, "sector": row["sector_etf"],
                              "z": float(z), "sig_row": row, "sig_date": t})
            cands.sort(key=lambda x: -x["z"])
            pending = cands

    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(curve)
    return {"trades": trades_df, "curve": curve_df, "strategy": strategy,
            "mode": mode, "start": start, "end": end, "total_rt_bps": total_rt_bps}


def spy_buy_hold(bars: dict, dates: list[pd.Timestamp], start: str, end: str) -> pd.DataFrame:
    window = [d for d in dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    entry = bars[window[0]][BENCHMARK]["open"]
    shares = INITIAL_EQUITY / entry
    return pd.DataFrame({"date": window,
                         "equity": [shares * bars[d][BENCHMARK]["close"] for d in window]})
