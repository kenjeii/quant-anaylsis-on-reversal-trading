# Data Provenance

All price data is vendor OHLCV only. No synthetic prices, news, fundamentals, or NLP inputs were used.

## Track 1: intraday 15m

- Source: Alpaca SIP `15Min` and `1Day` bars, split-adjusted; `^VIX` from Yahoo Finance chart-v8.
- Requested window: `2023-11-01` through `2026-11-20`.
- Universe: 12 equities (`NVDA`, `MU`, `AMD`, `AVGO`, `AAPL`, `MSFT`, `META`, `AMZN`, `GOOGL`, `TSLA`, `NFLX`, `QCOM`), benchmark `QQQ`, and `INDEX_VIX`.
- Processed coverage: 767 trading days, 15m bars from `2023-11-01 09:30` to `2026-11-20 15:45`; daily bars from `2023-11-01` to `2026-11-20`.
- `INDEX_VIX_daily.csv`: 1305 bars, `2021-11-22` to `2026-11-20`.
- Warm-up: `test_start_date = 2024-12-01`; 272 warm-up trading days available, 60 required, shortfall 0.
- Missing RTH 15m bars are gaps, never forward-filled. Counts by symbol are recorded in `data/alpaca_processed/data_quality.json`.
- Raw JSON caches were removed after this note was written. Clean files under `data/alpaca_processed/` remain available for offline reruns.

## Track 2: daily cross-section

- Source: Alpaca SIP `1Day` bars, split-adjusted, plus Yahoo Finance chart-v8 for `^VIX` and `^VIX3M`.
- Requested window: `2020-11-01` through `2026-11-20`.
- Universe: 48 large-cap equities, 8 sector ETFs, and `SPY` (57 Alpaca symbols total), plus two Yahoo volatility indices.
- Processed coverage: 1521 trading days per Alpaca symbol, `2020-11-02` to `2026-11-20`; master calendar from `SPY_daily.csv`.
- Yahoo index coverage: `INDEX_VIX` 2516 bars and `INDEX_VIX3M` 2514 bars, both `2016-11-21` to `2026-11-20`, with 100% alignment to SPY trading dates after intersection.
- Vendor split anomaly: Alpaca `adjustment=split` produced a 50.9496% implausible jump for `HON`. The downloader refetched genuine `adjustment=all` bars, reducing the maximum tradable jump to 11.1940%. This override is recorded in the original manifest and `data/rev2_processed/data_quality.json`.
- High/low bad ticks were flagged, never altered. The three flagged bars were:
  - `NVDA 2024-06-10`: high `195.95` versus open `120.37` and close `121.79`. This date is inside LOCKED OOS.
  - `TSLA 2021-03-04`: high `291.31` versus open `218.60` and close `207.15`.
  - `MRK 2021-06-11`: low `15.32` versus open `76.71` and close `76.27`.
- These flags feed the `hl_anomaly` stop guard in the daily engine.
- Raw JSON caches were removed after this note was written. Clean files under `data/rev2_processed/` remain available for offline reruns.

## Retention

Kept locally: `data/alpaca_processed/` and `data/rev2_processed/`.
Removed: `data/raw/`, `data/processed/`, `data/alpaca_raw/`, and `data/rev2_raw/`.
Large processed CSVs and `.env` are ignored by Git; the two `data_quality.json` manifests remain tracked as provenance records.
