"""Phase 6A — clean raw Reversal 2.0 daily JSON into data/rev2_processed/.

Writes ONLY to data/rev2_processed/. Legacy data dirs are never touched.
Prices are NEVER silently altered: anomalies are flagged/counted in
data_quality.json, not patched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.daily_swing.universe import ALPACA_SYMBOLS

from .paths import ROOT, REV2_PROCESSED, REV2_RAW, OUTPUTS  # noqa: E402
RAW = ROOT / "data" / "rev2_raw"
OUT = ROOT / "data" / "rev2_processed"
COLS = ["symbol", "date", "open", "high", "low", "close", "volume"]

YAHOO_NAMES = {
    "INDEX_VIX_1d_10y": "INDEX_VIX",
    "INDEX_VIX3M_1d_10y": "INDEX_VIX3M",
}


def _clean_alpaca(symbol: str) -> tuple[pd.DataFrame, dict]:
    payload = json.loads((RAW / f"{symbol}_1d_6y.json").read_text(encoding="utf-8"))
    bars = payload.get("bars", [])
    frame = pd.DataFrame(bars)
    if frame.empty:
        raise RuntimeError(f"{symbol}: no bars in raw file")
    # t is RFC3339 UTC; normalize to America/New_York calendar date.
    ts = pd.to_datetime(frame["t"], utc=True, format="mixed").dt.tz_convert("America/New_York")
    frame["date"] = ts.dt.tz_localize(None).dt.normalize()
    frame = frame.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    frame["symbol"] = symbol
    frame = frame[COLS]

    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
    frame = frame[(frame[["open", "high", "low", "close"]] > 0).all(axis=1)]
    frame = frame[frame["volume"] >= 0]

    frame = frame.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)

    # Anomaly FLAGS ONLY — do not change any price.
    high_anom = frame["high"] > frame[["open", "close"]].max(axis=1) * 1.25
    low_anom = frame["low"] < frame[["open", "close"]].min(axis=1) * 0.75
    hl_anomaly = int((high_anom | low_anom).sum())

    close_jump = (frame["close"] / frame["close"].shift(1) - 1.0).abs()
    open_jump = (frame["open"] / frame["close"].shift(1) - 1.0).abs()
    max_jump = float(pd.concat([close_jump, open_jump], axis=1).replace([np.inf, -np.inf], np.nan).max(axis=1).max())

    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    frame = frame[COLS]
    frame.to_csv(OUT / f"{symbol}_daily.csv", index=False)

    quality = {
        "bars": int(len(frame)),
        "first_date": str(frame["date"].iloc[0]),
        "last_date": str(frame["date"].iloc[-1]),
        "max_tradable_jump": round(max_jump, 6),
        "hl_anomaly_bars": hl_anomaly,
    }
    return frame, quality


def _clean_yahoo(stem: str, symbol: str) -> tuple[pd.DataFrame, dict]:
    payload = json.loads((RAW / f"{stem}.json").read_text(encoding="utf-8"))
    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp", [])
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame({
        "date_unix": timestamps,
        "open": quote.get("open", []),
        "high": quote.get("high", []),
        "low": quote.get("low", []),
        "close": quote.get("close", []),
        "volume": quote.get("volume", [np.nan] * len(timestamps)),
    })
    # Yahoo daily timestamps are ~14:30 UTC => NY calendar date.
    frame["date"] = (
        pd.to_datetime(frame["date_unix"], unit="s", utc=True)
        .dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    )
    frame["symbol"] = symbol
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame["volume"] = frame["volume"].fillna(0)
    frame = frame[(frame[["open", "high", "low", "close"]] > 0).all(axis=1)]
    frame = frame.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    frame = frame[COLS]
    frame.to_csv(OUT / f"{symbol}_daily.csv", index=False)

    quality = {
        "bars": int(len(frame)),
        "first_date": str(frame["date"].iloc[0]),
        "last_date": str(frame["date"].iloc[-1]),
    }
    return frame, quality


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    quality: dict = {"symbols": {}, "indices": {}, "master_calendar": {}}

    frames: dict[str, pd.DataFrame] = {}
    for symbol in ALPACA_SYMBOLS:
        frame, info = _clean_alpaca(symbol)
        frames[symbol] = frame
        quality["symbols"][symbol] = info
        print(f"{symbol:5s} bars={info['bars']:4d} {info['first_date']}..{info['last_date']} "
              f"jump={info['max_tradable_jump']:.4%} hl_anom={info['hl_anomaly_bars']}", flush=True)

    spy_dates = set(frames["SPY"]["date"])
    quality["master_calendar"] = {
        "source": "SPY_daily.csv",
        "bars": len(spy_dates),
        "first_date": min(spy_dates),
        "last_date": max(spy_dates),
    }

    for stem, symbol in YAHOO_NAMES.items():
        frame, info = _clean_yahoo(stem, symbol)
        # VIX series is 10y; SPY is 6y. Correct alignment = fraction of SPY
        # trading dates that have a VIX observation (not VIX dates in SPY).
        index_dates = set(frame["date"])
        aligned = sum(1 for d in spy_dates if d in index_dates) / len(spy_dates)
        info["alignment_to_spy"] = round(float(aligned), 6)
        quality["indices"][symbol] = info
        print(f"{symbol:12s} bars={info['bars']:4d} {info['first_date']}..{info['last_date']} "
              f"spy_align={aligned:.4%}", flush=True)

    (OUT / "data_quality.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("DONE: cleaned 57 Alpaca symbols + VIX + VIX3M", flush=True)


if __name__ == "__main__":
    main()
