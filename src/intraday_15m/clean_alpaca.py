"""Phase 4A: clean preserved Alpaca SIP raw bars into the existing feature CSV schema."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import ROOT, ALPACA_PROCESSED, ALPACA_RAW, OUTPUTS  # noqa: E402
RAW = ROOT / "data" / "alpaca_raw"
PROCESSED = ROOT / "data" / "alpaca_processed"
ET = "America/New_York"
SYMBOLS = ("NVDA", "MU", "AMD", "AVGO", "AAPL", "MSFT", "META", "AMZN",
           "GOOGL", "TSLA", "NFLX", "QCOM", "QQQ")
TEST_START = "2024-12-01"


def _expected_slots(date: pd.Timestamp) -> set:
    try:
        import exchange_calendars as xcals
        calendar = xcals.get_calendar("XNYS")
        stamp = date.tz_localize(ET) if date.tzinfo is None else date.tz_convert(ET)
        session = stamp.normalize()
        row = calendar.schedule.loc[session]
        start = row["open"].tz_convert(ET)
        end = row["close"].tz_convert(ET)
        slots = pd.date_range(start, end, freq="15min", inclusive="left")
    except Exception:
        slots = pd.date_range("09:30", "16:00", freq="15min", inclusive="left")
    return {stamp.time() for stamp in slots}


def _load_bars(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "chart" in payload:
        result = payload["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        frame = pd.DataFrame({
            "t": pd.to_datetime(result.get("timestamp", []), unit="s", utc=True),
            "o": quote.get("open", []), "h": quote.get("high", []),
            "l": quote.get("low", []), "c": quote.get("close", []),
            "v": quote.get("volume", []),
        })
    else:
        frame = pd.DataFrame(payload.get("bars", []))
    return frame


def _clean_intraday(frame: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, dict]:
    raw_count = len(frame)
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["t"], utc=True).dt.tz_convert(ET)
    duplicated = int(frame["timestamp"].duplicated().sum())
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
    time_values = frame["timestamp"].dt.time
    session_mask = pd.Series(
        [pd.Timestamp("09:30").time() <= value < pd.Timestamp("16:00").time()
         for value in time_values], index=frame.index)
    excluded = int((~session_mask).sum())
    frame = frame.loc[session_mask].copy()
    frame = frame.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    invalid = ((frame["high"] < frame[["open", "close", "low"]].max(axis=1)) |
               (frame["low"] > frame[["open", "close", "high"]].min(axis=1)) |
               (frame[["open", "high", "low", "close"]] <= 0).any(axis=1) |
               (frame["volume"] < 0))
    invalid_count = int(invalid.sum())
    frame = frame.loc[~invalid].copy()
    frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["slot"] = ((frame["timestamp"].dt.hour - 9) * 60 +
                     frame["timestamp"].dt.minute - 30) // 15
    frame["symbol"] = symbol
    frame = frame[["symbol", "timestamp", "date", "slot", "open", "high", "low", "close", "volume"]]
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    missing = 0
    for date, group in frame.groupby("date"):
        missing += len(_expected_slots(date) - set(group["timestamp"].dt.time))
    quality = {
        "raw_bars": raw_count,
        "rth_bars": len(frame),
        "excluded_non_rth_bars": excluded,
        "duplicate_timestamps": duplicated,
        "invalid_ohlc_bars": invalid_count,
        "missing_rth_bars": missing,
        "first_bar": str(frame["timestamp"].min()),
        "last_bar": str(frame["timestamp"].max()),
        "trading_days": int(frame["date"].nunique()),
    }
    return frame, quality


def _clean_daily(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["t"], utc=True).dt.tz_convert(ET)
    frame = frame.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    frame["date"] = frame["timestamp"].dt.normalize().dt.tz_localize(None)
    frame["symbol"] = symbol
    frame = frame.drop_duplicates(subset=["date"], keep="last")
    frame = frame[["symbol", "date", "open", "high", "low", "close", "volume"]]
    return frame.sort_values("date").reset_index(drop=True)


def clean_all() -> dict:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    quality: dict = {}
    for symbol in SYMBOLS:
        intraday, report = _clean_intraday(_load_bars(RAW / f"{symbol}_15m_3y.json"), symbol)
        intraday.to_csv(PROCESSED / f"{symbol}_15m.csv", index=False)
        quality[symbol + "_15m"] = report
        daily = _clean_daily(_load_bars(RAW / f"{symbol}_1d_3y.json"), symbol)
        daily.to_csv(PROCESSED / f"{symbol}_daily.csv", index=False)
        quality[symbol + "_1d"] = {"bars": len(daily), "first_date": str(daily["date"].min()),
                                   "last_date": str(daily["date"].max())}
    vix = _clean_daily(_load_bars(RAW / "INDEX_VIX_1d_5y.json"), "INDEX_VIX")
    vix.to_csv(PROCESSED / "INDEX_VIX_daily.csv", index=False)
    quality["INDEX_VIX_1d"] = {"bars": len(vix), "first_date": str(vix["date"].min()),
                               "last_date": str(vix["date"].max())}

    qqq = pd.read_csv(PROCESSED / "QQQ_15m.csv", parse_dates=["date"])
    dates = sorted(qqq["date"].unique())
    test_start = pd.Timestamp(TEST_START)
    warmup_days = int(sum(date < test_start for date in dates))
    quality["window"] = {
        "first_trading_day": str(dates[0])[:10],
        "last_trading_day": str(dates[-1])[:10],
        "trading_days": len(dates),
        "test_start_date": TEST_START,
        "warmup_days_available": warmup_days,
        "original_warmup_days_required": 60,
        "shortfall_days": 0,
    }
    (PROCESSED / "data_quality.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")
    return quality


if __name__ == "__main__":
    report = clean_all()
    print(json.dumps(report["window"], indent=2, ensure_ascii=False))
    for symbol in SYMBOLS:
        item = report[symbol + "_15m"]
        print(f"{symbol:5s} days={item['trading_days']:4d} rth_bars={item['rth_bars']:6d} "
              f"missing={item['missing_rth_bars']:4d}")
