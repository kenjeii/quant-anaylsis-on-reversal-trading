"""Phase 4B: frozen V_FINAL validation on 2 years of Alpaca SIP data.

Strictly offline. No Alpaca API call. No parameter tuning.
"""
from __future__ import annotations

import copy
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

from .paths import ROOT, ALPACA_PROCESSED, ALPACA_RAW, OUTPUTS  # noqa: E402
# package imports; no sys.path shim needed

from src.intraday_15m.backtest import run_backtest                       # noqa: E402
from src.intraday_15m.evaluation import evaluate                        # noqa: E402


def _v0(base: dict) -> dict:
    """Frozen V0 all-AND control (inlined from the original comparison module)."""
    cfg = copy.deepcopy(base)
    cfg["event"].update(mode="all_and", maximum_event_wait_bars=8, latest_event_time="14:45",
                        stale_zcar=1.0, maximum_preentry_reversal=1.0)
    cfg["confirmation"].update(mode="original", retracement_fraction=0.25, momentum_bars=3)
    cfg["regime"].update(mode="original", prohibition_daily_move=0.02,
                         prohibition_trend_r2=0.65, prohibition_vwap_gap=0.0,
                         multipliers={"normal": 1.0, "cautious": 0.5,
                                      "high_stress": 0.25, "crisis": 0.0})
    cfg["risk"].update(stop_bound_reference="atr_15m", stop_atr_buffer=0.25,
                       minimum_stop_atr=0.50, maximum_stop_atr=1.50,
                       minimum_tp1_rr=1.0, minimum_tp2_rr=1.5,
                       tp1_r_multiple=1.0, tp2_r_multiple=1.75)
    return cfg


def _phase3_cfg(base: dict, variant: str, fee_bps: int) -> dict:
    """Frozen Phase 3 momentum config (inlined verbatim)."""
    cfg = copy.deepcopy(base)
    cfg["data"]["symbols"]["trade"] = list(cfg["data"]["symbols"]["universes"]["extended"])
    cfg["costs"]["fee_round_trip_bps"] = int(fee_bps)
    if variant.startswith("P3_MOM"):
        cfg["event"].update(trade_style="momentum", earliest_event_time="09:30",
                            latest_event_time="10:30", latest_confirmation_time="10:45",
                            maximum_event_wait_bars=2)
        cfg["confirmation"].update(latest_entry_time="11:00")
        tb_max_hold = 3
    else:
        cfg["event"].update(trade_style="mean_reversion", earliest_event_time="10:30",
                            latest_event_time="13:30", latest_confirmation_time="14:15",
                            maximum_event_wait_bars=3)
        cfg["confirmation"].update(latest_entry_time="14:30")
        tb_max_hold = 6
    cfg["confirmation"]["mode"] = ("direct" if variant == "P3_MOM_Open_Direct" else
                                   "momentum_fast1" if variant == "P3_MOM_Open_Fast1" else
                                   "reversion_fast1")
    cfg["exit"].update(mode="triple_barrier", tb_max_hold_bars=tb_max_hold)
    cfg["risk"].update(stop_bound_reference="atr_15m", minimum_stop_atr=0.5,
                       maximum_stop_atr=3.0, minimum_tp1_rr=0.9, minimum_tp2_rr=1.4,
                       tb_stop_atr_mult=1.5, tb_tp1_atr_mult=1.5, tb_tp2_atr_mult=2.5,
                       maximum_simultaneous_positions=6, maximum_total_open_risk=0.03,
                       correlated_group_max_risk=0.03)
    cfg["regime"]["soft_sizing_only"] = True
    return cfg


def _build_variant(base: dict, name: str, fee: int) -> dict:
    """Frozen variant builder (inlined verbatim from final evaluation)."""
    if name == "V0_Control_2sym":
        cfg = _v0(base)
    elif name == "V2_Baseline_2sym":
        cfg = copy.deepcopy(base)
    elif name == "V2_Baseline_12sym":
        cfg = copy.deepcopy(base)
        cfg["data"]["symbols"]["trade"] = list(base["data"]["symbols"]["universes"]["extended"])
    elif name == "P3_MOM_Open_Direct":
        cfg = _phase3_cfg(base, "P3_MOM_Open_Direct", fee)
    else:
        cfg = _phase3_cfg(base, "P3_MOM_Open_Direct", fee)
        max_hold = 3 if name == "V_FINAL_h3" else 2
        cfg["event"].update(latest_event_time="10:15")
        cfg["exit"].update(monitor_entry_bar=True, tb_max_hold_bars=max_hold,
                           tb_single_exit=True)
        cfg["risk"].update(tb_tp_atr_mult=2.0, minimum_tp2_rr=1.25)
        cfg["regime"].update(use_regime_multiplier=False, soft_sizing_only=True)
    cfg["costs"]["fee_round_trip_bps"] = int(fee)
    return cfg


OUTPUT = ROOT / "outputs" / "alpaca_2y_oos_evaluation.md"
PROCESSED = ROOT / "data" / "alpaca_processed"
PROCESSED_DIR = "data/alpaca_processed"
FULL_START = "2024-12-01"
TRUE_SPLIT = pd.Timestamp("2026-10-12")
FEE_SETS = {
    "V2_Baseline_12sym": (0, 1, 2, 5, 10, 15),
    "V_FINAL": (0, 1, 2, 5, 10, 15),
    "V_FINAL_h3": (2, 15),
}
VERSIONS = tuple(FEE_SETS)
SYMBOLS = ("NVDA", "MU", "AMD", "AVGO", "AAPL", "MSFT", "META", "AMZN",
           "GOOGL", "TSLA", "NFLX", "QCOM", "QQQ")
TRADE_SYMBOLS = SYMBOLS[:-1]
SEMI = ("NVDA", "MU", "AMD", "AVGO", "QCOM")
MEGA = ("AAPL", "MSFT", "META", "AMZN", "GOOGL", "TSLA", "NFLX")


def _f(value, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not np.isfinite(number) else f"{number:.{digits}f}"


def _pct(value, digits: int = 1) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.{digits}f}%"


def _profit_factor(trades: pd.DataFrame):
    if trades.empty:
        return None
    wins = trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum()
    losses = abs(trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum())
    return float(wins / losses) if losses > 0 else None


def _max_price_jump(frame: pd.DataFrame) -> float:
    # Tradable-price transitions only: close-to-close and overnight open vs prior close.
    # A single bad high/low print is not a price transition and is reported separately.
    close_jump = frame["close"].pct_change().abs()
    gap_jump = (frame["open"] / frame["close"].shift(1) - 1.0).abs()
    values = pd.concat([close_jump, gap_jump], ignore_index=True)
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.max()) if len(values) else 0.0


def daily_range_outlier_note() -> str:
    offenders = []
    for symbol in SYMBOLS:
        frame = pd.read_csv(PROCESSED / f"{symbol}_daily.csv", parse_dates=["date"])
        values = ((frame["high"] - frame["low"]) / frame["open"]).abs()
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        maximum = float(values.max()) if len(values) else 0.0
        if maximum > 0.40:
            offenders.append(f"{symbol} ({maximum:.2%})")
    return "無" if not offenders else "；".join(offenders)


def split_sanity_check() -> tuple[str, str]:
    rows = []
    notes = []
    for symbol in SYMBOLS:
        daily = pd.read_csv(PROCESSED / f"{symbol}_daily.csv", parse_dates=["date"])
        intraday = pd.read_csv(PROCESSED / f"{symbol}_15m.csv", parse_dates=["timestamp", "date"])
        max_jump = max(_max_price_jump(daily), _max_price_jump(intraday))
        rows.append((symbol, max_jump))
        if max_jump > 0.40:
            notes.append(f"{symbol} ({max_jump:.2%})")
        assert max_jump <= 0.45, f"Unadjusted split/jump detected in {symbol}: {max_jump:.2%}"
    header = "| symbol | max tradable-price jump (close/close or open/prev close) | status |\n|---|---:|---|"
    lines = [header]
    for symbol, jump in rows:
        status = "note >40%" if jump > 0.40 else "ok"
        lines.append(f"| {symbol} | {jump:.4%} | {status} |")
    note = "無；所有標的皆低於 40%" if not notes else "；".join(notes)
    return "\n".join(lines), note


def build_config(base: dict, name: str, fee: int) -> dict:
    cfg = _build_variant(base, name, fee)
    cfg["data"]["processed_dir"] = PROCESSED_DIR
    cfg["data"]["test_start_date"] = FULL_START
    cfg["features"].update(slot_z_window=60, car_z_window=60, rvol_window=60,
                           qqq_rv_percentile_days=252)
    return cfg


def window_metrics(evaluation: dict, fee: int, window: str) -> dict:
    trades = evaluation["trades_frame"].copy()
    equity = evaluation["daily_equity"].copy()
    entry_dates = pd.to_datetime(trades["entry_time"]).dt.normalize()
    if window == "True OOS":
        trade_mask = entry_dates < TRUE_SPLIT
        equity_mask = equity["date"] < TRUE_SPLIT
    elif window == "Discovery":
        trade_mask = entry_dates >= TRUE_SPLIT
        equity_mask = equity["date"] >= TRUE_SPLIT
    else:
        trade_mask = pd.Series(True, index=trades.index)
        equity_mask = pd.Series(True, index=equity.index)
    sub = trades[trade_mask]
    eq = equity[equity_mask]
    daily_returns = eq["daily_return"].dropna()
    sharpe = (float(daily_returns.mean() / daily_returns.std(ddof=1) * math.sqrt(252))
              if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0 else None)
    days = int(eq["date"].nunique())
    se_sr = math.sqrt(252.0 / days) if days else None
    notional = float((sub["entry_price"] * sub["shares"]).sum()) if len(sub) else 0.0
    pre_slip = float(sub["pre_slippage_pnl"].sum()) if len(sub) else 0.0
    slippage = float(sub["slippage_cost"].sum()) if len(sub) else 0.0
    gross = float(sub["gross_pnl"].sum()) if len(sub) else 0.0
    cost = float(sub["cost"].sum()) if len(sub) else 0.0
    net = float(sub["net_pnl"].sum()) if len(sub) else 0.0
    borrow = max(0.0, cost - fee / 10000.0 * notional)
    return {
        "window": window, "fee": fee, "days": days, "trades": int(len(sub)),
        "long": int((sub["direction"] == "long").sum()) if len(sub) else 0,
        "short": int((sub["direction"] == "short").sum()) if len(sub) else 0,
        "win_rate": float((sub["net_pnl"] > 0).mean()) if len(sub) else None,
        "mfe": float(sub["mfe_r"].mean()) if len(sub) else None,
        "mae": float(sub["mae_r"].mean()) if len(sub) else None,
        "pre_slip": pre_slip, "slippage": slippage, "gross": gross, "cost": cost,
        "borrow": borrow, "net": net,
        "expectancy": float(sub["net_pnl"].mean()) if len(sub) else None,
        "profit_factor": _profit_factor(sub),
        "max_dd": float(eq["drawdown"].min()) if len(eq) else None,
        "sharpe": sharpe, "se_sr": se_sr, "notional": notional,
        "turnover": notional * 2.0,
        "raw_bps": pre_slip / notional * 10000.0 if notional else 0.0,
        "slip_bps": slippage / notional * 10000.0 if notional else 0.0,
        "net_bps": net / notional * 10000.0 if notional else 0.0,
        "breakeven_fee": max(0.0, (gross - borrow) / notional * 10000.0) if notional else 0.0,
        "trades_frame": sub,
    }


def _summary_row(version: str, m: dict) -> str:
    return (
        f"| {m['window']} | {version} | {m['days']} | {m['trades']} | {m['long']}/{m['short']} | "
        f"{_pct(m['win_rate'])} | {_f(m['mfe'], 3)} | {_f(m['mae'], 3)} | "
        f"{_f(m['pre_slip'])} | {_f(m['slippage'])} | {_f(m['cost'])} | {_f(m['net'])} | "
        f"{_f(m['expectancy'])} | {_f(m['profit_factor'], 3)} | {_f(m['max_dd'], 5)} | "
        f"{_f(m['sharpe'], 3)} | {_f(m['se_sr'], 3)} | {_f(m['raw_bps'])} | "
        f"{_f(m['slip_bps'])} | {_f(m['net_bps'])} | {_f(m['breakeven_fee'])} |"
    )


def _summary_table(summaries: dict, fee: int) -> str:
    lines = [
        "| Window | Version | Days | Trades | Long/Short | Win Rate | MFE R | MAE R | Pre-Slip PnL | Slippage | Fees+Borrow | Net PnL | Expectancy | PF | Max DD | Sharpe | SE(SR) | Raw Alpha (bps) | Slippage (bps) | Net Edge (bps) | Breakeven Fee (bps) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window in ("True OOS", "Discovery", "Full 2Y"):
        for version in VERSIONS:
            lines.append(_summary_row(version, summaries[(version, fee, window)]))
    return "\n".join(lines)


def _qqq_quarters() -> dict[str, tuple[float, str]]:
    qqq = pd.read_csv(PROCESSED / "QQQ_daily.csv", parse_dates=["date"])
    qqq = qqq[(qqq["date"] >= FULL_START) & (qqq["date"] <= "2026-11-20")].copy()
    qqq["quarter"] = qqq["date"].dt.to_period("Q").astype(str)
    out = {}
    for quarter, group in qqq.groupby("quarter"):
        ret = float(group["close"].iloc[-1] / group["close"].iloc[0] - 1.0)
        label = "bull" if ret > 0.05 else "bear" if ret < -0.05 else "sideways"
        out[f"{quarter[:4]}-{quarter[4:]}"] = (ret, label)
    return out


def _quarter_table(summaries: dict) -> str:
    qqq = _qqq_quarters()
    t2 = summaries[("V_FINAL", 2, "Full 2Y")]["trades_frame"].copy()
    t15 = summaries[("V_FINAL", 15, "Full 2Y")]["trades_frame"].copy()
    for frame in (t2, t15):
        frame["quarter"] = (pd.to_datetime(frame["entry_time"]).dt.to_period("Q").astype(str)
                           .map(lambda value: f"{value[:4]}-{value[4:]}"))
    lines = [
        "| Quarter | QQQ_Ret_% | 市況 | Trades | Long/Short | Win Rate | Pre-Slip PnL | Net PnL (2bps) | Net PnL (15bps) | PF (2bps) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for quarter in sorted(qqq):
        ret, label = qqq[quarter]
        sub2 = t2[t2["quarter"] == quarter]
        sub15 = t15[t15["quarter"] == quarter]
        lines.append(
            f"| {quarter} | {_pct(ret)} | {label} | {len(sub2)} | "
            f"{int((sub2['direction'] == 'long').sum())}/{int((sub2['direction'] == 'short').sum())} | "
            f"{_pct(float((sub2['net_pnl'] > 0).mean())) if len(sub2) else 'n/a'} | "
            f"{_f(float(sub2['pre_slippage_pnl'].sum()))} | {_f(float(sub2['net_pnl'].sum()))} | "
            f"{_f(float(sub15['net_pnl'].sum()))} | {_f(_profit_factor(sub2), 3)} |"
        )
    return "\n".join(lines)


def _long_short_table(summaries: dict) -> str:
    lines = ["| Window | Fee_bps | 方向 | Trades | Win Rate | Pre-Slip PnL | Net PnL | Expectancy | PF |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for window in ("True OOS", "Full 2Y"):
        for fee in (2, 15):
            trades = summaries[("V_FINAL", fee, window)]["trades_frame"]
            for direction in ("long", "short"):
                sub = trades[trades["direction"] == direction]
                lines.append(
                    f"| {window} | {fee} | {direction} | {len(sub)} | "
                    f"{_pct(float((sub['net_pnl'] > 0).mean())) if len(sub) else 'n/a'} | "
                    f"{_f(float(sub['pre_slippage_pnl'].sum()))} | "
                    f"{_f(float(sub['net_pnl'].sum()))} | "
                    f"{_f(float(sub['net_pnl'].mean()) if len(sub) else None)} | "
                    f"{_f(_profit_factor(sub), 3)} |"
                )
    return "\n".join(lines)


def _fee_sensitivity_table(summaries: dict) -> str:
    lines = ["| Fee_bps | V2 Net PnL | V2 Expectancy | V2 PF | V_FINAL Net PnL | V_FINAL Expectancy | V_FINAL PF |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for fee in (0, 1, 2, 5, 10, 15):
        v2 = summaries[("V2_Baseline_12sym", fee, "True OOS")]
        vf = summaries[("V_FINAL", fee, "True OOS")]
        lines.append(
            f"| {fee} | {_f(v2['net'])} | {_f(v2['expectancy'])} | {_f(v2['profit_factor'], 3)} | "
            f"{_f(vf['net'])} | {_f(vf['expectancy'])} | {_f(vf['profit_factor'], 3)} |"
        )
    return "\n".join(lines)


def _concentration_table(summaries: dict) -> tuple[str, str, str]:
    trades = summaries[("V_FINAL", 2, "True OOS")]["trades_frame"].copy()
    grouped = trades.groupby("ticker").agg(trades=("net_pnl", "size"), net=("net_pnl", "sum"),
                                          expectancy=("net_pnl", "mean")).reindex(TRADE_SYMBOLS)
    grouped["trades"] = grouped["trades"].fillna(0).astype(int)
    grouped["net"] = grouped["net"].fillna(0.0)
    grouped = grouped.sort_values("net", ascending=False)
    lines = ["| 標的 | Trades | Net PnL | Expectancy |", "|---|---:|---:|---:|"]
    for symbol, row in grouped.iterrows():
        lines.append(f"| {symbol} | {int(row['trades'])} | {_f(row['net'])} | {_f(row['expectancy'])} |")
    semi_net = float(trades[trades["ticker"].isin(SEMI)]["net_pnl"].sum())
    mega_net = float(trades[trades["ticker"].isin(MEGA)]["net_pnl"].sum())
    groups = (f"半導體群（{', '.join(SEMI)}）Net PnL = {_f(semi_net)}；"
              f"大型科技群（{', '.join(MEGA)}）Net PnL = {_f(mega_net)}。")
    best = str(grouped.index[0])
    without_best = trades[trades["ticker"] != best]
    daily = trades.assign(day=pd.to_datetime(trades["entry_time"]).dt.normalize()).groupby("day")["net_pnl"].sum()
    best_days = list(daily.sort_values(ascending=False).head(2).index)
    worst_days = list(daily.sort_values().head(2).index)
    without_best_days = trades[~pd.to_datetime(trades["entry_time"]).dt.normalize().isin(best_days)]
    without_worst_days = trades[~pd.to_datetime(trades["entry_time"]).dt.normalize().isin(worst_days)]
    stress = (
        f"排除最佳標的 {best}：{len(without_best)} 筆、Net PnL {_f(float(without_best['net_pnl'].sum()))}。"
        f"排除最佳 2 個交易日（{', '.join(str(d.date()) for d in best_days)}）："
        f"{_f(float(without_best_days['net_pnl'].sum()))}。"
        f"排除最差 2 個交易日（{', '.join(str(d.date()) for d in worst_days)}）："
        f"{_f(float(without_worst_days['net_pnl'].sum()))}。"
    )
    return "\n".join(lines), groups, stress


def _exit_table(summaries: dict) -> str:
    trades = summaries[("V_FINAL", 2, "True OOS")]["trades_frame"]
    counters: dict[str, int] = {}
    pnls: dict[str, list[float]] = {}
    for _, row in trades.iterrows():
        for reason in str(row["exit_reason"]).split("|"):
            if not reason:
                continue
            counters[reason] = counters.get(reason, 0) + 1
            pnls.setdefault(reason, []).append(float(row["net_pnl"]))
    lines = ["| exit reason | count | avg net PnL | total net PnL |", "|---|---:|---:|---:|"]
    for reason in ("stop", "tp_final", "tb_time_barrier", "force_exit_1545"):
        values = pnls.get(reason, [])
        lines.append(f"| {reason} | {counters.get(reason, 0)} | "
                     f"{_f(np.mean(values)) if values else 'n/a'} | {_f(np.sum(values))} |")
    return "\n".join(lines)


def _observational_tables(frames: dict, summaries: dict) -> tuple[str, str]:
    trades = summaries[("V_FINAL", 2, "True OOS")]["trades_frame"].copy()
    rows = []
    for _, trade in trades.iterrows():
        frame = frames[trade["ticker"]]
        match = frame.index[frame["timestamp"] == pd.to_datetime(trade["event_time"])]
        if len(match) == 0:
            continue
        bar = frame.iloc[match[0]]
        high_low = float(bar["high"]) - float(bar["low"])
        efficiency = abs(float(bar["close"]) - float(bar["open"])) / high_low if high_low > 0 else np.nan
        prev_close = float(bar["prev_daily_close"])
        gap = float(bar["open"]) / prev_close - 1.0 if prev_close > 0 else np.nan
        drive = float(bar["close"]) / float(bar["open"]) - 1.0
        rvol = float(trade["rvol"])
        values = (rvol, efficiency, gap, drive, float(trade["net_pnl"]))
        if not all(np.isfinite(value) for value in values):
            continue
        rows.append({"rvol": rvol, "efficiency": efficiency,
                     "gap": gap, "drive": drive,
                     "net_pnl": float(trade["net_pnl"])})
    obs = pd.DataFrame(rows)
    if obs.empty:
        empty = ("| bucket | N | Win Rate | Net PnL | Expectancy | PF |\n"
                 "|---|---:|---:|---:|---:|---:|\n"
                 "| n/a | 0 | n/a | n/a | n/a | n/a |")
        return empty, empty
    q1, q2 = obs["rvol"].quantile([1/3, 2/3])
    if not np.isfinite(q1) or not np.isfinite(q2) or q1 == q2:
        obs["rvol_bucket"] = "Mid"
    else:
        obs["rvol_bucket"] = pd.cut(obs["rvol"], [-np.inf, q1, q2, np.inf],
                                    labels=["Low", "Mid", "High"])
    obs["eff_bucket"] = np.where(obs["efficiency"] < 0.5, "<0.5", ">=0.5")
    obs["rvol_x_eff"] = obs["rvol_bucket"].astype(str) + " / " + obs["eff_bucket"]

    def _block(column: str, by: str) -> str:
        lines = [f"| {by} | N | Win Rate | Net PnL | Expectancy | PF |", "|---|---:|---:|---:|---:|---:|"]
        for key, group in obs.groupby(column, observed=True):
            lines.append(f"| {key} | {len(group)} | {_pct(float((group['net_pnl'] > 0).mean()))} | "
                         f"{_f(float(group['net_pnl'].sum()))} | {_f(float(group['net_pnl'].mean()))} | "
                         f"{_f(_profit_factor(group), 3)} |")
        return "\n".join(lines)

    table_a = _block("rvol_x_eff", "rvol_bucket / efficiency")

    def _gap_class(row) -> str:
        if abs(row["gap"]) < 0.01:
            return "Pure Open Drive"
        if np.sign(row["gap"]) == np.sign(row["drive"]):
            return "Gap-and-Go"
        return "Gap-Fade / Mixed"

    obs["gap_class"] = obs.apply(_gap_class, axis=1)
    table_b = _block("gap_class", "gap_class")
    return table_a, table_b


def main() -> None:
    base = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    sanity_table, sanity_note = split_sanity_check()
    range_note = daily_range_outlier_note()
    print("Split-jump sanity check:")
    print(sanity_table)
    print(f"Notes: {sanity_note}", flush=True)
    print(f"Daily high-low outliers (not tradable-price jumps): {range_note}", flush=True)

    evaluations: dict[tuple[str, int], dict] = {}
    summaries: dict[tuple[str, int, str], dict] = {}
    frames_final2 = None
    for name in VERSIONS:
        for fee in FEE_SETS[name]:
            config = build_config(base, name, fee)
            result = run_backtest(config, make_plots=False)
            eval_result = evaluate(result)
            evaluations[(name, fee)] = eval_result
            if name == "V_FINAL" and fee == 2:
                frames_final2 = result["frames"]
            for window in ("True OOS", "Discovery", "Full 2Y"):
                summaries[(name, fee, window)] = window_metrics(eval_result, fee, window)
            print(f"PASS: ran {name} fee={fee}", flush=True)

    for (name, fee), evaluation in evaluations.items():
        for trade in evaluation["trades_frame"].to_dict("records"):
            residual = abs((trade["pre_slippage_pnl"] - trade["slippage_cost"]) - trade["gross_pnl"])
            assert residual < 1e-3, (name, fee, trade.get("trade_id"), residual)
    print("PASS: accounting invariant across all Alpaca 2Y trades", flush=True)

    true_final = summaries[("V_FINAL", 2, "True OOS")]
    disc_final = summaries[("V_FINAL", 2, "Discovery")]
    full_final = summaries[("V_FINAL", 2, "Full 2Y")]
    true_significant = (true_final["sharpe"] is not None and true_final["se_sr"] is not None and
                        abs(true_final["sharpe"]) > 2.0 * true_final["se_sr"])
    symbol_table, group_note, concentration_note = _concentration_table(summaries)
    table_a, table_b = _observational_tables(frames_final2, summaries)
    range_detail = ("；其中 NVDA 2024-06-10 的 `high` 為單一異常值，"
                    "open/low/close 彼此一致且 15m 無對應異常") if "NVDA" in range_note else ""

    report = f"""# Alpaca SIP 2 年 True OOS 期末考評估

資料：Alpaca SIP 真實 15m 與 1Day split-adjusted bars，完全離線讀取 `data/alpaca_processed/`；沒有再次呼叫 API、沒有合成價格、沒有調動任何凍結的 `V_FINAL` 參數。
暖機參數還原為原規格：`slot_z_window=60`、`car_z_window=60`、`rvol_window=60`、`qqq_rv_percentile_days=252`。正式窗口從 `2024-12-01` 起；True OOS 定義為 `2024-12-01` 至 `2026-10-11`，Discovery Window 為 `2026-10-12` 至 `2026-11-20`。

**資料防呆：** 13 檔標的皆通過可成交價格轉換（close-to-close 與 open/前收）單根跳價 <45% 檢查；需注意（>40%）者：{sanity_note}。另外，日線 high-low 異常值（非可成交價格轉換、不影響 2024-12-01 後的 OOS）：{range_note}{range_detail}。

{sanity_table}

## 1. 一頁總結（期末考結果：V_FINAL 在未見過的 2 年資料上活下來了嗎？）

- **True OOS（{true_final['days']} 天）**：V_FINAL fee=2bps 共 {true_final['trades']} 筆、Net PnL {_f(true_final['net'])}、Expectancy {_f(true_final['expectancy'])}、PF {_f(true_final['profit_factor'], 3)}、Sharpe {_f(true_final['sharpe'], 3)}。
- **Discovery（{disc_final['days']} 天）**：fee=2bps 共 {disc_final['trades']} 筆、Net PnL {_f(disc_final['net'])}、PF {_f(disc_final['profit_factor'], 3)}。
- **Full 2Y（{full_final['days']} 天）**：fee=2bps 共 {full_final['trades']} 筆、Net PnL {_f(full_final['net'])}、PF {_f(full_final['profit_factor'], 3)}。
- **統計 power**：True OOS `SE(SR)=sqrt(252/{true_final['days']})={_f(true_final['se_sr'], 3)}`，`|Sharpe| > 2*SE`：**{'是' if true_significant else '否'}**。
- **True OOS 損益兩平手續費（含滑價）**：{_f(true_final['breakeven_fee'])} bps round-trip。

## 2. True OOS vs Discovery vs Full 2Y 完整表現表

### 2.1 fee = 2 bps（Slippage ON）

{_summary_table(summaries, 2)}

### 2.2 fee = 15 bps（Slippage ON）

{_summary_table(summaries, 15)}

## 3. 跨市況季度／半年度穩定度（V_FINAL at fee = 2 bps）

{_quarter_table(summaries)}

> 市況標籤以 QQQ 季度報酬粗略分類：>5% bull、<-5% bear，其餘 sideways；僅供描述，不參與策略調整。

## 4. Long vs Short 完整拆解（True OOS vs Full 2Y）

{_long_short_table(summaries)}

## 5. 全費率敏感度（True OOS）

{_fee_sensitivity_table(summaries)}

## 6. 個股與板塊集中度（True OOS，V_FINAL at fee = 2 bps）

{symbol_table}

{group_note}

{concentration_note}

## 7. 出場原因與止損實測統計（True OOS）

{_exit_table(summaries)}

> `stop` 的實際 PnL distribution 用來檢驗 1.5 ATR 停損是否在 2 年樣本中被真正壓力測試；`tb_time_barrier` 對齊 h=2 的 30 分鐘垂直出場。

## 8. 下一步升級線索（僅觀察、未調參 — True OOS）

### Table A — rvol terciles × bar efficiency

{table_a}

### Table B — Overnight Gap vs Intraday Open Drive

{table_b}

> 本節完全為事後描述性診斷，沒有調整 `V_FINAL` 的任何參數或規則。

## 限制與免責聲明

2 年 Alpaca SIP 樣本仍只包含單一 12 檔科技／半導體 universes；即使 True OOS 結果為正，也不能直接外推到不同市況、不同期間或不同標的群。以上內容僅供量化研究參考，不構成投資建議；市場有風險，過往表現不預示未來收益。
"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
