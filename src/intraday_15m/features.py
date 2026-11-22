"""Lag-safe numerical features. Every rolling statistic excludes the current bar/day."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .paths import ROOT, ALPACA_PROCESSED, ALPACA_RAW, OUTPUTS  # noqa: E402
PROCESSED = ROOT / "data" / "processed"


def _resolve_processed_dir(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def _robust_window(series: pd.Series, keys: pd.Series, window: int, statistic: str = "return") -> pd.Series:
    """Previous-window median/MAD by key; the current observation is excluded."""
    result = pd.Series(np.nan, index=series.index, dtype=float)
    for key, idx in keys.groupby(keys).groups.items():
        values = series.loc[idx].astype(float)
        past = values.shift(1)
        if statistic == "return":
            med = past.rolling(window, min_periods=window).median()
            mad = past.rolling(window, min_periods=window).apply(_mad, raw=True)
            result.loc[idx] = np.where(mad > 0, (values - med) / (1.4826 * mad), np.nan)
        elif statistic == "volume":
            result.loc[idx] = past.rolling(window, min_periods=window).median()
        else:
            raise ValueError(statistic)
    return result


def _rolling_slope_r2(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    slopes = pd.Series(np.nan, index=series.index, dtype=float)
    r2s = pd.Series(np.nan, index=series.index, dtype=float)
    values = series.to_numpy(dtype=float)
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denom = float((x_centered ** 2).sum())
    for i in range(window - 1, len(values)):
        y = values[i - window + 1:i + 1]
        if not np.isfinite(y).all():
            continue
        y_centered = y - y.mean()
        slope = float((x_centered * y_centered).sum() / denom)
        intercept = float(y.mean() - slope * x.mean())
        fitted = intercept + slope * x
        ss_res = float(((y - fitted) ** 2).sum())
        ss_tot = float((y_centered ** 2).sum())
        slopes.iloc[i] = slope
        r2s.iloc[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slopes, r2s


def _load_intraday(symbol: str, processed_dir: Path = PROCESSED) -> pd.DataFrame:
    frame = pd.read_csv(Path(processed_dir) / f"{symbol}_15m.csv", parse_dates=["timestamp", "date"])
    return frame.sort_values("timestamp").reset_index(drop=True)


def _load_daily(name: str, processed_dir: Path = PROCESSED) -> pd.DataFrame:
    frame = pd.read_csv(Path(processed_dir) / f"{name}_daily.csv", parse_dates=["date"])
    return frame.sort_values("date").reset_index(drop=True)


def _previous_daily_close(daily: pd.DataFrame) -> pd.Series:
    return daily.set_index("date")["close"].shift(1)


def _daily_baseline(daily: pd.DataFrame, window: int) -> tuple[pd.Series, pd.Series]:
    """Mean/std of daily returns strictly before each date."""
    prices = daily.set_index("date")["close"].astype(float)
    returns = prices.pct_change()
    mean = returns.shift(1).rolling(window, min_periods=window).mean()
    std = returns.shift(1).rolling(window, min_periods=window).std(ddof=1)
    return mean, std


def _daily_atr(daily: pd.DataFrame, window: int) -> pd.Series:
    frame = daily.set_index("date")
    prev = frame["close"].shift(1)
    tr = pd.concat([frame["high"] - frame["low"],
                    (frame["high"] - prev).abs(), (frame["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(span=window, adjust=False, min_periods=window).mean().shift(1)


def _car_matrix(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    matrix = frame.pivot_table(index="date", columns="slot", values="car", aggfunc="last")
    return matrix.sort_index()


def _car_robust_z(frame: pd.DataFrame, window: int, mad_floor: float) -> pd.Series:
    matrix = _car_matrix(frame, window)
    medians = pd.DataFrame(np.nan, index=matrix.index, columns=matrix.columns, dtype=float)
    mads = medians.copy()
    for slot in matrix.columns:
        past = matrix[slot].shift(1)
        medians[slot] = past.rolling(window, min_periods=window).median()
        mads[slot] = past.rolling(window, min_periods=window).apply(_mad, raw=True).clip(lower=mad_floor)
    z = (matrix - medians) / (1.4826 * mads)
    stacked = z.stack().rename("z_car").reset_index()
    stacked.columns = ["date", "slot", "z_car"]
    merged = frame[["date", "slot"]].merge(stacked, on=["date", "slot"], how="left")
    return merged["z_car"].set_axis(frame.index)


def build_qqq_regime(qqq: pd.DataFrame, daily_qqq: pd.DataFrame, vix: pd.DataFrame,
                     config: dict) -> pd.DataFrame:
    q = qqq[["timestamp", "date", "slot", "open", "high", "low", "close", "volume"]].copy()
    q = q.sort_values("timestamp").reset_index(drop=True)
    prev_close = _previous_daily_close(daily_qqq)
    q["prev_daily_close"] = q["date"].map(prev_close)
    q["qqq_daily_return"] = q["close"] / q["prev_daily_close"] - 1.0

    typical = (q["high"] + q["low"] + q["close"]) / 3.0
    grouped = q.groupby("date", sort=False)
    q["vwap"] = (typical * q["volume"]).groupby(q["date"]).cumsum() / q["volume"].groupby(q["date"]).cumsum()
    q["qqq_vwap_gap"] = (q["close"] - q["vwap"]) / q["vwap"]

    log_close = np.log(q["close"])
    trend_bars = int(config["features"]["qqq_trend_bars"])
    q["trend_slope"], q["trend_r2"] = _rolling_slope_r2(log_close, trend_bars)
    q["qqq_8bar_return"] = grouped["close"].transform(lambda x: x / x.shift(trend_bars - 1) - 1.0)

    rb = int(config["features"]["qqq_rv_bars"])
    q["r_15"] = np.log(grouped["close"].transform(lambda x: x / x.shift(1)))
    q.loc[grouped.head(1).index, "r_15"] = np.log(q.loc[grouped.head(1).index, "close"] /
                                                  q.loc[grouped.head(1).index, "prev_daily_close"])
    q["qqq_rv"] = np.sqrt(252.0 * (q["r_15"] ** 2).rolling(rb, min_periods=rb).sum())

    rv_daily = q.groupby("date")["qqq_rv"].last().dropna()
    percentile_days = int(config["features"]["qqq_rv_percentile_days"])
    q90 = rv_daily.shift(1).rolling(percentile_days, min_periods=20).quantile(
        config["regime"]["qqq_rv_crisis_percentile"])
    q["qqq_rv_q90_prev"] = q["date"].map(q90)

    vix_index = vix.set_index("date")["close"].sort_index()
    dates = pd.Index(sorted(q["date"].unique()))
    vix_prev = {}
    for date in dates:
        available = vix_index[vix_index.index < date]
        if len(available):
            vix_prev[date] = float(available.iloc[-1])
    q["vix_prev_close"] = q["date"].map(vix_prev)

    regime = config["regime"]
    q["regime_daily_move"] = q["qqq_daily_return"].abs() >= regime["qqq_daily_move"]
    q["regime_vwap_gap"] = q["qqq_vwap_gap"].abs() >= regime["qqq_vwap_gap"]
    trend_strong = (q["trend_r2"] >= regime["qqq_trend_r2"]) & \
                   (q["qqq_8bar_return"].abs() >= regime["qqq_trend_return"])
    q["regime_trend"] = trend_strong.fillna(False)
    q["regime_rv"] = (q["qqq_rv"] >= q["qqq_rv_q90_prev"]).fillna(False)
    q["regime_vix"] = (q["vix_prev_close"] >= regime["vix_stressed"]).fillna(False)
    score_cols = ["regime_daily_move", "regime_vwap_gap", "regime_trend", "regime_rv", "regime_vix"]
    q["regime_score"] = q[score_cols].sum(axis=1).astype(int)
    if regime.get("mode", "revised") == "original":
        q["regime_state"] = pd.cut(q["regime_score"], bins=[-1, 1, 2, 3, 5],
                                   labels=["normal", "cautious", "high_stress", "crisis"]).astype(str)
        limit = regime["prohibition_daily_move"]
        prohibition_trend = trend_strong
        vwap_limit = 0.0
    else:
        q["regime_state"] = pd.cut(q["regime_score"], bins=[-1, 1, 2, 3, 4, 5],
                                   labels=["normal", "cautious", "high_stress", "severe", "crisis"]).astype(str)
        limit = regime["prohibition_daily_move"]
        prohibition_trend = (q["trend_r2"] >= regime["prohibition_trend_r2"]).fillna(False)
        vwap_limit = float(regime["prohibition_vwap_gap"])
    q["regime_multiplier"] = q["regime_state"].map(regime["multipliers"]).astype(float)

    q["long_prohibited"] = ((q["qqq_daily_return"] <= -limit) & prohibition_trend &
                            (q["qqq_vwap_gap"] <= -vwap_limit)).fillna(False)
    q["short_prohibited"] = ((q["qqq_daily_return"] >= limit) & prohibition_trend &
                             (q["qqq_vwap_gap"] >= vwap_limit)).fillna(False)
    return q


def build_symbol_features(symbol: str, qqq_features: pd.DataFrame, config: dict,
                          processed_dir: Path = PROCESSED) -> pd.DataFrame:
    frame = _load_intraday(symbol, processed_dir)
    daily = _load_daily(symbol, processed_dir)
    features_cfg = config["features"]
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    daily_mean, daily_std = _daily_baseline(daily, int(features_cfg["daily_z_window"]))
    prev_daily = _previous_daily_close(daily)
    daily_atr = _daily_atr(daily, int(features_cfg["atr_daily_window"]))

    frame["prev_daily_close"] = frame["date"].map(prev_daily)
    frame["daily_mean_prev"] = frame["date"].map(daily_mean)
    frame["daily_std_prev"] = frame["date"].map(daily_std)
    frame["daily_atr_prev"] = frame["date"].map(daily_atr)
    frame["r_day"] = frame["close"] / frame["prev_daily_close"] - 1.0
    denom = frame["daily_std_prev"].replace(0, np.nan)
    frame["z_daily"] = (frame["r_day"] - frame["daily_mean_prev"]) / denom

    grouped = frame.groupby("date", sort=False)
    prev_close = grouped["close"].shift(1)
    prev_close = prev_close.fillna(frame["prev_daily_close"])
    frame["r_15"] = np.log(frame["close"] / prev_close)
    frame["z_slot"] = _robust_window(frame["r_15"], frame["slot"],
                                    int(features_cfg["slot_z_window"]), "return")
    floor = float(features_cfg["robust_mad_floor"])
    frame["z_slot"] = frame["z_slot"].replace([np.inf, -np.inf], np.nan).clip(-30, 30)

    frame["four_bar_return"] = grouped["close"].transform(lambda x: x / x.shift(4) - 1.0)

    tr = pd.concat([frame["high"] - frame["low"],
                    (frame["high"] - prev_close).abs(),
                    (frame["low"] - prev_close).abs()], axis=1).max(axis=1)
    frame["tr"] = tr
    frame["atr_15m"] = tr.ewm(span=int(features_cfg["atr_15m_window"]),
                              adjust=False, min_periods=int(features_cfg["atr_15m_window"])).mean()
    frame["move_atr"] = (frame["close"] - frame["prev_daily_close"]).abs() / \
                        frame["daily_atr_prev"].replace(0, np.nan)

    frame["rvol"] = frame["volume"] / _robust_window(frame["volume"], frame["slot"],
                                                     int(features_cfg["rvol_window"]), "volume").replace(0, np.nan)

    beta_window = int(features_cfg["beta_window_bars"])
    pair = frame[["timestamp", "r_15"]].merge(
        qqq_features[["timestamp", "r_15"]].rename(columns={"r_15": "r_qqq"}),
        on="timestamp", how="left")
    cov = pair["r_15"].rolling(beta_window, min_periods=100).cov(pair["r_qqq"])
    var = pair["r_qqq"].rolling(beta_window, min_periods=100).var()
    beta = (cov / var.replace(0, np.nan)).shift(1).clip(0.25, 3.00)
    frame["beta"] = beta.set_axis(frame.index)
    frame["residual"] = frame["r_15"] - frame["beta"] * pair["r_qqq"].set_axis(frame.index)
    frame["car"] = frame.groupby("date")["residual"].cumsum()
    frame["four_bar_residual"] = frame.groupby("date")["residual"].transform(
        lambda x: x.rolling(4, min_periods=4).sum()
    )
    frame["day_high_so_far"] = frame.groupby("date")["high"].cummax()
    frame["day_low_so_far"] = frame.groupby("date")["low"].cummin()
    frame["z_car"] = _car_robust_z(frame, int(features_cfg["car_z_window"]), floor)
    frame["z_car"] = frame["z_car"].replace([np.inf, -np.inf], np.nan).clip(-30, 30)

    qcols = ["timestamp", "qqq_daily_return", "qqq_vwap_gap", "trend_r2", "qqq_8bar_return",
             "qqq_rv", "qqq_rv_q90_prev", "vix_prev_close", "regime_score", "regime_state",
             "regime_multiplier", "long_prohibited", "short_prohibited"]
    frame = frame.merge(qqq_features[qcols], on="timestamp", how="left", suffixes=("", "_qqq"))
    return frame


def build_all(config: dict) -> dict:
    processed_dir = _resolve_processed_dir(config["data"].get("processed_dir", PROCESSED))
    qqq_raw = _load_intraday("QQQ", processed_dir)
    qqq_daily = _load_daily("QQQ", processed_dir)
    vix = _load_daily("INDEX_VIX", processed_dir)
    qqq_features = build_qqq_regime(qqq_raw, qqq_daily, vix, config)
    out = {"QQQ": qqq_features}
    for symbol in config["data"]["symbols"]["trade"]:
        out[symbol] = build_symbol_features(symbol, qqq_features, config, processed_dir)
    dates = pd.Index(sorted(qqq_features["date"].unique()))
    test_start_date = config["data"].get("test_start_date")
    if test_start_date:
        candidates = dates[dates >= pd.Timestamp(test_start_date)]
        out["test_start"] = candidates[0] if len(candidates) else dates[0]
    else:
        out["test_start"] = dates[-int(config["data"]["test_trading_days"])]
    out["dates"] = list(dates)
    return out
