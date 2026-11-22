"""Phase 6A — download 6 years of Alpaca SIP daily bars (57 symbols) and
Yahoo Finance ^VIX / ^VIX3M daily term-structure series.

Credentials are read ONLY from environment variables or the local .env file.
No key or secret is written to source, manifest, raw JSON, or any output file.

All writes go to data/rev2_raw/ — legacy data directories are never touched.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from src.daily_swing.universe import ALPACA_SYMBOLS

from .paths import ROOT, REV2_PROCESSED, REV2_RAW, OUTPUTS  # noqa: E402
RAW = ROOT / "data" / "rev2_raw"

BASE_URL = "https://data.alpaca.markets/v2/stocks/bars"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"

START = "2020-11-01T00:00:00Z"
END = "2026-11-20T23:59:59Z"
SLEEP_BETWEEN_REQUESTS = 0.30
LIMIT = 10000

# Yahoo indices: (local filename stem, URL-encoded symbol)
VIX_SERIES = (
    ("INDEX_VIX_1d_10y", "%5EVIX"),
    ("INDEX_VIX3M_1d_10y", "%5EVIX3M"),
)
YAHOO_PARAMS = {"interval": "1d", "range": "10y", "includePrePost": "false"}


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def _credentials() -> tuple[str, str]:
    env = _load_env(ROOT / ".env")
    key = os.environ.get("APCA_API_KEY_ID") or env.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY") or env.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in environment or .env")
    return key, secret


def _get_json(url: str, params: dict | None = None, headers: dict | None = None,
              timeout: int = 90, attempts: int = 4) -> dict:
    query = urllib.parse.urlencode(params or {})
    full_url = f"{url}?{query}" if query else url
    request = urllib.request.Request(full_url, headers=headers or {})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if retryable and attempt < attempts - 1:
                time.sleep(2.0 * (2 ** attempt))  # 2,4,8s exponential backoff
                continue
            raise RuntimeError(f"HTTP {exc.code} while fetching {urllib.parse.urlsplit(url).path}") from exc
        except urllib.error.URLError as exc:
            if attempt < attempts - 1:
                time.sleep(2.0 * (2 ** attempt))
                continue
            raise RuntimeError(f"Network error fetching {urllib.parse.urlsplit(url).path}: {exc.reason}") from exc
    raise RuntimeError(f"Unreachable state fetching {url}")


def _max_tradable_jump(bars: list[dict]) -> float:
    """Max |close/prev_close-1| or |open/prev_close-1| from raw daily bars."""
    if len(bars) < 2:
        return 0.0
    closes = [b["c"] for b in bars]
    max_jump = 0.0
    for i in range(1, len(bars)):
        prev = closes[i - 1]
        if not prev or prev <= 0:
            continue
        max_jump = max(max_jump, abs(closes[i] / prev - 1.0),
                       abs(bars[i]["o"] / prev - 1.0))
    return float(max_jump)


def _fetch_symbol_daily(symbol: str, headers: dict, adjustment: str = "split") -> list[dict]:
    params = {
        "symbols": symbol,
        "timeframe": "1Day",
        "start": START,
        "end": END,
        "feed": "sip",
        "adjustment": adjustment,
        "limit": LIMIT,
        "sort": "asc",
    }
    bars: list[dict] = []
    page_token: str | None = None
    while True:
        request_params = dict(params)
        if page_token:
            request_params["page_token"] = page_token
        payload = _get_json(BASE_URL, request_params, headers)
        bars.extend(payload.get("bars", {}).get(symbol, []))
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    return bars


def _write_bars(path: Path, symbol: str, adjustment: str, bars: list[dict]) -> None:
    path.write_text(json.dumps({
        "symbol": symbol,
        "timeframe": "1Day",
        "feed": "sip",
        "adjustment": adjustment,
        "start": START,
        "end": END,
        "bars": bars,
    }), encoding="utf-8")


def _cached_bars(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    bars = payload.get("bars", [])
    return bars if bars else None


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    api_key, api_secret = _credentials()
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}

    manifest: dict = {
        "source": "Alpaca SIP 1Day split-adjusted + Yahoo Finance chart-v8",
        "start": START,
        "end": END,
        "feed": "sip",
        "adjustment": "split",
        "timeframe": "1Day",
        "alpaca_symbols": list(ALPACA_SYMBOLS),
        "yahoo_series": [name for name, _ in VIX_SERIES],
        "split_override": {},  # symbols whose `split` feed had an unadjusted jump
        "files": {},
        # Deliberately NO credentials here.
    }

    for symbol in ALPACA_SYMBOLS:
        path = RAW / f"{symbol}_1d_6y.json"
        cached = _cached_bars(path)
        adjustment_used = "split"
        if cached is not None:
            bars = cached
            print(f"{symbol:5s} 1Day bars={len(bars)} (cached)", flush=True)
        else:
            bars = _fetch_symbol_daily(symbol, headers, "split")
            print(f"{symbol:5s} 1Day bars={len(bars)} (downloaded split)", flush=True)

        # Institutional split-adjustment guard: if the `split` feed shows an
        # implausible >=45% single-bar tradable jump, re-fetch with `all`
        # (a genuine Alpaca adjustment, NEVER a fabricated/synthetic price).
        # Record the override transparently in the manifest and raw metadata.
        jump = _max_tradable_jump(bars)
        if jump >= 0.45:
            print(f"{symbol:5s} split feed jump={jump:.2%} >=45%; re-fetching adjustment=all", flush=True)
            bars_all = _fetch_symbol_daily(symbol, headers, "all")
            if _max_tradable_jump(bars_all) < jump:
                bars = bars_all
                adjustment_used = "all"
                manifest["split_override"][symbol] = {
                    "split_jump": round(jump, 6),
                    "all_jump": round(_max_tradable_jump(bars_all), 6),
                    "reason": "Alpaca split feed showed >=45% implausible jump; used genuine adjustment=all bars",
                }
                print(f"{symbol:5s} override to adjustment=all, jump now "
                      f"{_max_tradable_jump(bars):.2%}", flush=True)
            else:
                raise RuntimeError(
                    f"{symbol}: both split and all feeds show >=45% jump "
                    f"({jump:.2%} / {_max_tradable_jump(bars_all):.2%}); refusing to fabricate prices"
                )

        _write_bars(path, symbol, adjustment_used, bars)
        manifest["files"][path.name] = {
            "symbol": symbol, "bars": len(bars), "adjustment": adjustment_used,
            "max_tradable_jump": round(_max_tradable_jump(bars), 6),
        }

    for name, encoded in VIX_SERIES:
        path = RAW / f"{name}.json"
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("chart", {}).get("result"):
                    print(f"{name} (cached)", flush=True)
                    manifest["files"][path.name] = {"series": name, "cached": True}
                    continue
            except (json.JSONDecodeError, OSError):
                pass
        payload = _get_json(YAHOO_CHART.format(sym=encoded), YAHOO_PARAMS,
                            {"User-Agent": "Mozilla/5.0"}, timeout=90)
        if not payload.get("chart", {}).get("result"):
            raise RuntimeError(f"Yahoo returned no result for {name}: {str(payload)[:200]}")
        path.write_text(json.dumps(payload), encoding="utf-8")
        print(f"{name} (downloaded)", flush=True)
        manifest["files"][path.name] = {"series": name, "cached": False}

    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"DONE: {len(ALPACA_SYMBOLS)} Alpaca symbols + {len(VIX_SERIES)} Yahoo series", flush=True)


if __name__ == "__main__":
    main()
