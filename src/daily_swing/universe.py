"""Phase 6A — Reversal 2.0 universe definitions.

Pure constant module. No network, no side effects.

48 stocks across 8 GICS-like sectors + 9 ETFs (8 sector ETFs + SPY) = 57
Alpaca symbols. Sector ETF is the single-factor hedge/regression basis for
each stock.
"""
from __future__ import annotations

SECTOR_MAP: dict[str, tuple[str, ...]] = {
    "SMH": ("NVDA", "AMD", "AVGO", "QCOM", "MU", "TXN"),
    "XLK": ("AAPL", "MSFT", "ORCL", "CRM", "ADBE", "CSCO"),
    "XLC": ("META", "GOOGL", "NFLX", "DIS", "CMCSA", "TMUS"),
    "XLY": ("AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX"),
    "XLF": ("JPM", "BAC", "GS", "MS", "BLK", "V"),
    "XLV": ("UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO"),
    "XLI": ("CAT", "HON", "UNP", "LMT", "DE", "RTX"),
    "XLE": ("XOM", "CVX", "COP", "SLB", "EOG", "MPC"),
}

SECTOR_ETFS: tuple[str, ...] = tuple(SECTOR_MAP.keys())  # 8 ETFs
BENCHMARK: str = "SPY"
ALL_ETFS: tuple[str, ...] = (BENCHMARK,) + SECTOR_ETFS  # 9 ETFs

STOCKS: tuple[str, ...] = tuple(
    sym for symbols in SECTOR_MAP.values() for sym in symbols
)  # 48 stocks

STOCK_TO_SECTOR: dict[str, str] = {
    sym: etf for etf, symbols in SECTOR_MAP.items() for sym in symbols
}

ALPACA_SYMBOLS: tuple[str, ...] = STOCKS + ALL_ETFS  # 57 total

assert len(STOCKS) == 48
assert len(ALL_ETFS) == 9
assert len(ALPACA_SYMBOLS) == 57
assert len(STOCK_TO_SECTOR) == 48
