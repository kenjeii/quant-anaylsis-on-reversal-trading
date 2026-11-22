"""Phase 4A: download Alpaca SIP 15m/1Day bars and Yahoo VIX daily bars.

Credentials are read only from environment variables or the local .env file.
No API key or secret is ever written to source, manifest, or output files.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .paths import ROOT, ALPACA_PROCESSED, ALPACA_RAW, OUTPUTS  # noqa: E402
RAW = ROOT / "data" / "alpaca_raw"
BASE_URL = "https://data.alpaca.markets/v2/stocks/bars"
VIX_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"

SYMBOLS = ("NVDA", "MU", "AMD", "AVGO", "AAPL", "MSFT", "META", "AMZN",
           "GOOGL", "TSLA", "NFLX", "QCOM", "QQQ")
START = "2023-11-01T00:00:00Z"
END = "2026-11-20T23:59:59Z"
SLEEP_BETWEEN_REQUESTS = 0.35


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
        raise RuntimeError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY in environment or .env")
    return key, secret


def _get_json(url: str, params: dict | None = None, headers: dict | None = None) -> dict:
    query = urllib.parse.urlencode(params or {})
    full_url = f"{url}?{query}" if query else url
    request = urllib.request.Request(full_url, headers=headers or {})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if retryable and attempt == 0:
                time.sleep(2.0)
                continue
            raise RuntimeError(f"HTTP {exc.code} while fetching {urllib.parse.urlsplit(url).path}") from exc
        except urllib.error.URLError as exc:
            if attempt == 0:
                time.sleep(2.0)
                continue
            raise RuntimeError(f"Network error while fetching {urllib.parse.urlsplit(url).path}: {exc.reason}") from exc


def _fetch_symbol_bars(symbol: str, timeframe: str, headers: dict) -> list[dict]:
    params = {
        "symbols": symbol,
        "timeframe": timeframe,
        "start": START,
        "end": END,
        "feed": "sip",
        "adjustment": "split",
        "limit": 10000,
        "sort": "asc",
    }
    bars: list[dict] = []
    page_token = None
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


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    api_key, api_secret = _credentials()
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
    records = []

    for symbol in SYMBOLS:
        for timeframe, suffix in (("15Min", "15m_3y"), ("1Day", "1d_3y")):
            path = RAW / f"{symbol}_{suffix}.json"
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                bars = existing.get("bars", [])
                if bars:
                    records.append({"symbol": symbol, "timeframe": timeframe, "file": path.name,
                                    "bars": len(bars), "feed": "sip", "adjustment": "split"})
                    print(f"{symbol:5s} {timeframe:5s} bars={len(bars)} (cached)", flush=True)
                    continue
            bars = _fetch_symbol_bars(symbol, timeframe, headers)
            path.write_text(json.dumps({"symbol": symbol, "timeframe": timeframe,
                                        "feed": "sip", "adjustment": "split",
                                        "start": START, "end": END, "bars": bars}),
                            encoding="utf-8")
            records.append({"symbol": symbol, "timeframe": timeframe, "file": path.name,
                            "bars": len(bars), "feed": "sip", "adjustment": "split"})
            print(f"{symbol:5s} {timeframe:5s} bars={len(bars)}", flush=True)

    vix_payload = _get_json(VIX_URL, {"interval": "1d", "range": "5y", "includePrePost": "false"},
                            {"User-Agent": "Mozilla/5.0"})
    vix_path = RAW / "INDEX_VIX_1d_5y.json"
    vix_path.write_text(json.dumps(vix_payload), encoding="utf-8")
    print(f"VIX  1Day bars={len(vix_payload.get('chart', {}).get('result', [{}])[0].get('timestamp', []))}")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "Alpaca SIP bars API + Yahoo Finance chart v8 for ^VIX",
        "start": START,
        "end": END,
        "symbols": list(SYMBOLS),
        "records": records,
        "vix_file": vix_path.name,
        "note": "No credentials are stored in this manifest.",
    }
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {RAW / 'manifest.json'}")


if __name__ == "__main__":
    main()
