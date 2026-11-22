"""Separate reversal-detection evaluation from P&L evaluation."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.intraday_15m.event_detector import oracle_outcome

from .paths import ROOT, ALPACA_PROCESSED, ALPACA_RAW, OUTPUTS  # noqa: E402
OUTPUTS = ROOT / "outputs"


def _safe_float(value, digits: int = 6):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if np.isfinite(number) else None


def evaluate(result: dict) -> dict:
    config = result["config"]
    events = result["events"]
    trades = result["trades"]
    frames = result["frames"]
    equity_curve = result["equity_curve"].copy()
    portfolio = result["portfolio"]
    test_start = result["test_start"]

    enriched_events = []
    for event in events:
        frame = frames[event["ticker"]]
        matches = frame.index[frame["timestamp"] == event["event_time"]]
        if len(matches) == 0:
            outcome = {"oracle_reversal": None, "reversal_label": None,
                       "oracle_peak": None, "oracle_trough": None}
        else:
            outcome = oracle_outcome(frame, int(matches[0]), event["direction"], config)
        entered = event.get("status") == "entered"
        enriched_events.append({**event, **outcome, "entered": entered,
                                "reversal_captured": bool(entered and outcome["reversal_label"] == "true")})
    events_frame = pd.DataFrame(enriched_events)
    trades_frame = pd.DataFrame(trades)

    equity_curve["date"] = pd.to_datetime(equity_curve["timestamp"]).dt.normalize()
    daily_equity = equity_curve.groupby("date", as_index=False).agg(
        equity=("equity", "last"), bars=("equity", "size"),
        exposure_bars=("open_positions", lambda x: int((x > 0).sum())))
    daily_equity["daily_return"] = daily_equity["equity"].pct_change()
    daily_equity["drawdown"] = daily_equity["equity"] / daily_equity["equity"].cummax() - 1.0

    initial = float(config["project"]["initial_equity"])
    final = float(daily_equity["equity"].iloc[-1]) if len(daily_equity) else initial
    net_return = final / initial - 1.0
    test_days = int(daily_equity["date"].nunique()) if len(daily_equity) else 0
    daily_returns = daily_equity["daily_return"].dropna()
    sharpe = (daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252)) \
        if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0 else None
    downside = daily_returns[daily_returns < 0]
    sortino = (daily_returns.mean() / downside.std(ddof=1) * np.sqrt(252)) \
        if len(downside) > 1 and downside.std(ddof=1) > 0 else None
    max_dd = float(daily_equity["drawdown"].min()) if len(daily_equity) else None

    if len(trades_frame):
        wins = trades_frame.loc[trades_frame["net_pnl"] > 0, "net_pnl"]
        losses = trades_frame.loc[trades_frame["net_pnl"] < 0, "net_pnl"]
        win_rate = float(len(wins) / len(trades_frame))
        avg_win = float(wins.mean()) if len(wins) else None
        avg_loss = float(losses.mean()) if len(losses) else None
        expectancy = float(trades_frame["net_pnl"].mean())
        profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else None
        avg_hold = float(trades_frame["holding_bars"].mean())
        median_hold = float(trades_frame["holding_bars"].median())
        long_trades = int((trades_frame["direction"] == "long").sum())
        short_trades = int((trades_frame["direction"] == "short").sum())
        avg_mfe = float(trades_frame["mfe_r"].mean())
        avg_mae = float(trades_frame["mae_r"].mean())
    else:
        win_rate = avg_win = avg_loss = expectancy = profit_factor = avg_hold = median_hold = avg_mfe = avg_mae = None
        long_trades = short_trades = 0

    true_events = events_frame[events_frame["reversal_label"] == "true"] if len(events_frame) else events_frame
    entered_events = events_frame[events_frame["entered"]] if len(events_frame) else events_frame
    true_entered = entered_events[entered_events["reversal_label"] == "true"] if len(entered_events) else entered_events
    precision = len(true_entered) / len(entered_events) if len(entered_events) else None
    recall = len(true_entered) / len(true_events) if len(true_events) else None
    false_entries = int(len(entered_events) - len(true_entered)) if len(entered_events) else 0
    missed = int(len(true_events) - len(true_entered)) if len(true_events) else 0
    avg_delay = float(events_frame["bars_waited"].mean()) if len(events_frame) else None
    avg_preentry_retrace = float(events_frame["retrace"].mean()) if "retrace" in events_frame and len(events_frame) else None
    if len(entered_events) and "retrace" in entered_events:
        late_entries = int(((entered_events["retrace"].notna()) &
                            (entered_events["oracle_reversal"].notna()) &
                            (entered_events["retrace"] >= entered_events["oracle_reversal"])).sum())
    else:
        late_entries = 0

    quality_dir = Path(config["data"].get("processed_dir", ROOT / "data" / "processed"))
    if not quality_dir.is_absolute():
        quality_dir = ROOT / quality_dir
    quality = json.loads((quality_dir / "data_quality.json").read_text(encoding="utf-8"))
    missing_bars = sum(v.get("missing_rth_bars", 0) for k, v in quality.items() if k.endswith("_15m"))

    r = float(config["confirmation"]["retracement_fraction"])
    f = float(config["exit"]["tp1_retracement"])
    k = float(config["event"]["move_atr"])
    buffer = float(config["risk"]["stop_atr_buffer"])
    rr1_upper = (f - r) / (1 - r) if 1 - r > 0 else None
    rr1_at_min_move = (f - r) * k / ((1 - r) * k + buffer) if (1 - r) * k + buffer > 0 else None

    metrics = {
        "report_type": "30_trading_day_smoke_test_not_profit_conclusion",
        "data_source": "Yahoo Finance chart v8; preserved raw responses",
        "data_window": {"first_trading_day": str(quality["window"]["first_trading_day"])[:10],
                        "last_trading_day": str(quality["window"]["last_trading_day"])[:10],
                        "test_start": str(test_start)[:10], "test_days": test_days,
                        "warmup_days_available": quality["window"]["warmup_days_available"],
                        "warmup_days_original_required": quality["window"]["original_warmup_days_required"],
                        "warmup_shortfall_days": quality["window"]["shortfall_days"]},
        "bars": {"test_15m_bars": int(len(equity_curve)),
                 "missing_rth_bars_in_download": int(missing_bars),
                 "note": "Missing bars retained as gaps; no synthetic prices inserted."},
        "events": {"extreme_events": int(len(events_frame)),
                   "armed_events": int((events_frame["status"] == "armed").sum()) if len(events_frame) else 0,
                   "confirmed_events": int(events_frame["confirmation_time"].notna().sum()) if len(events_frame) else 0,
                   "entered_events": int(len(entered_events)),
                   "cancelled_events": int((events_frame["status"] == "cancelled").sum()) if len(events_frame) else 0,
                   "rejected_events": int((events_frame["status"] == "rejected").sum()) if len(events_frame) else 0},
        "trades": {"confirmed_trades": int(len(trades_frame)), "long_trades": long_trades,
                   "short_trades": short_trades, "win_rate": _safe_float(win_rate),
                   "average_win": _safe_float(avg_win), "average_loss": _safe_float(avg_loss),
                   "expectancy": _safe_float(expectancy),
                   "average_holding_bars": _safe_float(avg_hold),
                   "median_holding_bars": _safe_float(median_hold),
                   "average_mfe_r": _safe_float(avg_mfe),
                   "average_mae_r": _safe_float(avg_mae)},
        "performance": {"initial_equity": initial, "final_equity": round(final, 2),
                        "net_return": _safe_float(net_return),
                        "annualized_return_display_only": _safe_float(
                            (1 + net_return) ** (252 / test_days) - 1 if test_days else None),
                        "sharpe": _safe_float(sharpe), "sortino": _safe_float(sortino),
                        "maximum_drawdown": _safe_float(max_dd),
                        "profit_factor": _safe_float(profit_factor),
                        "total_turnover": _safe_float(portfolio.total_turnover, 2),
                        "transaction_costs": _safe_float(portfolio.total_costs, 2),
                        "exposure": _safe_float(
                            float((equity_curve["open_positions"] > 0).mean()) if len(equity_curve) else None)},
        "reversal_detection": {"true_reversal_events": int(len(true_events)),
                               "true_reversals_entered": int(len(true_entered)),
                               "precision": _safe_float(precision), "recall": _safe_float(recall),
                               "false_reversal_entries": false_entries,
                               "missed_reversals": missed,
                               "late_entries": late_entries,
                               "average_entry_delay_bars": _safe_float(avg_delay),
                               "average_reversal_completed_before_entry": _safe_float(avg_preentry_retrace)},
        "expected_hold_model": result["expected_hold"],
        "structural_diagnostic": {
            "rr1_upper_bound_ignoring_atr_buffer": _safe_float(rr1_upper),
            "rr1_at_minimum_move_atr": _safe_float(rr1_at_min_move),
            "required_minimum_rr1": config["risk"]["minimum_tp1_rr"],
            "interpretation": ("With a 25% retracement entry and TP1 at a 50% retracement, "
                               "the structural stop at the event extreme gives RR1 <= about 1/3. "
                               "The specified RR1 >= 1.0 gate is therefore incompatible with the "
                               "specified MoveATR >= 1.75 event filter; this is a specification "
                               "issue, not a data failure."),
        },
        "limitations": [
            "Yahoo 15m history is limited to the latest 60 trading days; 30 days remain after the test window.",
            "Same-slot Z and CAR baselines use 20 preceding sessions instead of the specified 60.",
            "QQQ realized-volatility percentile uses available warm-up sessions instead of 252 sessions.",
            "Daily ATR and VIX are lagged to the previous completed session.",
            "The 30-day period is a smoke test and cannot establish long-term efficacy.",
        ],
        "disclaimer": "以上內容基於公開數據和量化分析，僅供參考，不構成投資建議。市場有風險，投資需謹慎。任何投資決策應結合個人風險承受能力、資金狀況和投資目標獨立判斷，必要時諮詢持牌專業機構。過往表現不預示未來收益。",
    }
    return {"metrics": metrics, "events_frame": events_frame, "trades_frame": trades_frame,
            "daily_equity": daily_equity, "funnel": result.get("funnel", {})}


def export_outputs(result: dict, evaluation: dict, outputs: Path = OUTPUTS) -> dict:
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "trades").mkdir(exist_ok=True)
    (outputs / "trade_details").mkdir(exist_ok=True)
    events_path = outputs / "events.csv"
    trades_path = outputs / "trades.csv"
    equity_path = outputs / "daily_equity.csv"
    metrics_path = outputs / "metrics.json"
    evaluation["events_frame"].to_csv(events_path, index=False)
    evaluation["trades_frame"].to_csv(trades_path, index=False)
    evaluation["daily_equity"].to_csv(equity_path, index=False)
    metrics_path.write_text(json.dumps(evaluation["metrics"], indent=2, ensure_ascii=False), encoding="utf-8")
    return {"events_csv": events_path, "trades_csv": trades_path,
            "daily_equity_csv": equity_path, "metrics_json": metrics_path}
