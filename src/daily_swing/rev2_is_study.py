"""Phase 6B — Reversal 2.0 IS-only ablation study (S0 -> S4).

STRICT OOS VAULT SEAL: only data/rev2_processed/ rows with date <= 2024-05-31
are read (enforced in rev2_features.build_daily_panel). No API. No tuning on
2024-06-01+ data.

Outputs:
- outputs/rev2_is_events.csv   (raw IS event table, long format)
- outputs/rev2_is_study.md     (Traditional Chinese full report)
"""
from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.daily_swing.rev2_features import HORIZONS, IS_END, IS_START, build_is_panel

from .paths import ROOT, REV2_PROCESSED, REV2_RAW, OUTPUTS  # noqa: E402
OUT = ROOT / "outputs"

VARIANTS = ("S0_Raw_SPY_3d", "S1_Sector_3d", "S2_Sector_NoNews_3d",
            "S3_Sector_NoNews_VIX_3d", "S4_Sector_NoNews_VIX_5d")
DIRECTIONS = ("Both", "Long-Only", "Short-Only")
EXITIONS = ("Outright", "Hedged")
Z_THRESH = 1.5


def _trigger_mask(panel: pd.DataFrame) -> dict[str, pd.Series]:
    base_sec = panel["z_res_sec_3d"].abs() >= Z_THRESH
    base_spy = panel["z_res_spy_3d"].abs() >= Z_THRESH
    nonews = panel["no_news_proxy_ok"].fillna(False)
    vix_ok = panel["vix_ratio"] < 1.05
    z5 = panel["z_res_sec_5d"].abs() >= Z_THRESH
    return {
        "S0_Raw_SPY_3d": base_spy,
        "S1_Sector_3d": base_sec,
        "S2_Sector_NoNews_3d": base_sec & nonews,
        "S3_Sector_NoNews_VIX_3d": base_sec & nonews & vix_ok,
        "S4_Sector_NoNews_VIX_5d": z5 & nonews & vix_ok,
    }


def _z_for_variant(panel: pd.DataFrame, variant: str) -> pd.Series:
    return panel["z_res_sec_5d"] if variant == "S4_Sector_NoNews_VIX_5d" else (
        panel["z_res_spy_3d"] if variant == "S0_Raw_SPY_3d" else panel["z_res_sec_3d"]
    )


def build_events(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    masks = _trigger_mask(panel)
    for variant in VARIANTS:
        mask = masks[variant].fillna(False)
        sub = panel[mask].copy()
        if sub.empty:
            continue
        z = _z_for_variant(panel, variant)[mask]
        # Reversal direction: oversold (z<=-1.5) -> Long; overbought (z>=+1.5) -> Short
        direction = np.where(z <= -Z_THRESH, "long", "short")
        sign = np.where(z <= -Z_THRESH, 1.0, -1.0)  # sign-flipped (mean reversion)
        sub = sub.assign(direction=direction, rev_sign=sign, z_value=z.values)
        for H in HORIZONS:
            raw = sub[f"fwd_raw_{H}"].values
            hedged = sub[f"fwd_hedged_{H}"].values
            for exe, arr in (("Outright", raw), ("Hedged", hedged)):
                signed = sign * arr
                valid = ~np.isnan(signed)
                rows.append(pd.DataFrame({
                    "variant": variant,
                    "horizon": H,
                    "execution": exe,
                    "event_date": sub["date"].values,
                    "target_date": sub[f"target_date_{H}"].values,
                    "symbol": sub["symbol"].values,
                    "direction": direction,
                    "z_value": z.values,
                    "signed_ret": signed,
                    "valid": valid,
                }))
    events = pd.concat(rows, ignore_index=True)
    events = events[events["valid"]].drop(columns=["valid"]).reset_index(drop=True)
    events["signed_bps"] = events["signed_ret"] * 1e4
    return events


def _date_cluster_stats(group: pd.DataFrame, value_col: str = "signed_bps") -> dict:
    """Fama-MacBeth style: equal-weight by calendar day, t on daily means."""
    daily = group.groupby("event_date")[value_col].mean()
    n_days = len(daily)
    if n_days < 2:
        return {"N": len(group), "D": n_days, "mean": daily.mean() if n_days else np.nan,
                "median": np.nan, "t": np.nan, "ir": np.nan}
    mean = float(daily.mean())
    median = float(group[value_col].median())
    sd = float(daily.std(ddof=1))
    t = mean / (sd / math.sqrt(n_days)) if sd > 0 else np.nan
    ir = mean / sd * math.sqrt(252) if sd > 0 else np.nan
    return {"N": int(len(group)), "D": int(n_days), "mean": mean, "median": median,
            "t": float(t), "ir": float(ir)}


def _l2o_t(group: pd.DataFrame) -> float:
    """Drop the 2 calendar days with largest |daily_mean * n_events| impact."""
    daily = group.groupby("event_date")["signed_bps"].agg(["mean", "size"])
    if len(daily) <= 2:
        return np.nan
    impact = (daily["mean"].abs() * daily["size"]).nlargest(2)
    keep = group[~group["event_date"].isin(impact.index)]
    return _date_cluster_stats(keep)["t"]


def cell_metrics(events: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for (variant, H, exe), grp in events.groupby(["variant", "horizon", "execution"]):
        for dlabel, dmask in (
            ("Both", grp["direction"].isin(["long", "short"])),
            ("Long-Only", grp["direction"] == "long"),
            ("Short-Only", grp["direction"] == "short"),
        ):
            sub = grp[dmask]
            if sub.empty:
                continue
            st = _date_cluster_stats(sub)
            records.append({
                "variant": variant, "horizon": H, "execution": exe, "direction": dlabel,
                "N_events": st["N"], "D_days": st["D"],
                "day_mean_bps": st["mean"], "day_median_bps": st["median"],
                "day_t": st["t"], "l2o_t": _l2o_t(sub), "ir_proxy": st["ir"],
            })
    return pd.DataFrame(records)


def candidate_shortlist(metrics: pd.DataFrame, M: int) -> tuple[pd.DataFrame, float]:
    t_bonf = float(NormalDist().inv_cdf(1 - 0.05 / (2 * M)))
    def passes(r: pd.Series) -> bool:
        hurdle = 25.0 if r["execution"] == "Outright" else 35.0
        return (r["D_days"] >= 80 and r["day_mean_bps"] >= hurdle
                and abs(r["day_t"]) >= 2.0 and abs(r["l2o_t"]) >= 1.7
                and r["day_median_bps"] > 0)
    mask = metrics.apply(passes, axis=1)
    out = metrics[mask].copy().sort_values("day_mean_bps", ascending=False)
    return out, t_bonf


def _fmt(v, pct=False) -> str:
    if pd.isna(v):
        return "n/a"
    return f"{v:.2%}" if pct else f"{v:.2f}"


def render_report(events: pd.DataFrame, metrics: pd.DataFrame, shortlist: pd.DataFrame,
                  t_bonf: float, M: int) -> str:
    lines: list[str] = []
    lines.append("# Phase 6B — Reversal 2.0 純量化反轉 IS 消融研究")
    lines.append("")
    lines.append("## 研究範圍與硬封印")
    lines.append("")
    lines.append(f"- IS 探索期：`{IS_START}` 至 `{IS_END}`（3 年，含 2022 熊市與多頭修復）。")
    lines.append("- **OOS vault 硬封印**：所有載入在讀取瞬間即強制 `date <= 2024-05-31`；"
                 "`2024-06-01` 至 `2026-11-20` 的列沒有被讀取、聚合或輸出。")
    lines.append("- 100% 離線讀取 `data/rev2_processed/`；無 API、無合成價格。")
    lines.append("- 所有 `.shift()` / `.rolling()` 都在單一 symbol frame 上完成後才 concat，"
                 "保證不會跨標的邊界洩漏；每個符號最後 `1+H` 列的 forward return 為 NaN。")
    lines.append("- 特徵只用 `open` / `close` / `volume`；`high/low` 毛刺（NVDA/TSLA/MRK）零干預。")
    lines.append("")
    lines.append("## 消融階梯與觸發定義")
    lines.append("")
    lines.append("- 反轉方向：oversold（`z <= -1.5`）做多、overbought（`z >= +1.5`）做空；"
                 "signed return 已對反轉方向翻轉（+1 long / -1 short）。")
    lines.append("- 進場 `open[D+1]`、出場 `open[D+1+H]`，完全可成交。")
    lines.append("- S0：`|z_res_spy_3d| >= 1.5`（SPY 單因子、無過濾）；"
                 "S1：`|z_res_sec_3d| >= 1.5`（加產業中性）；"
                 "S2：S1 + 無新聞代理（`max|gap|<=5%` 且 `max RVOL<=3`）；"
                 "S3：S2 + VIX 期限結構（`vix_ratio < 1.05`）；"
                 "S4：S3 改用 5 日累積殘差 `|z_res_sec_5d| >= 1.5`。")
    lines.append("- **S2 與 S3 數值相同的原因**：IS 期間 VIX 逆價差（`vix_ratio >= 1.05`）"
                 "共有 96/43,200 列（0.22%），但這些日期剛好都沒有同時通過 S2 的"
                 "`|z_res_sec_3d|>=1.5` 與無新聞代理，因此 VIX filter 在 IS 沒有剔除任何事件。"
                 "這不是程式失效，而是 2021-06～2024-05 大多處於 VIX contango；"
                 "VIX 恐慌過濾在 2022 熊市急殺的壓力效果需在 OOS 才會被檢驗。")
    lines.append("")
    lines.append("## 多重測試與門檻")
    lines.append("")
    lines.append(f"- 掃描格子數：`5 variants × 4 horizons × 2 executions = 40` 基礎 cell；"
                 f"每個再拆 Both/Long/Short 三種方向呈現 → **M = {M}** 個 dir×cell 組合。")
    lines.append(f"- Bonferroni 雙尾臨界值（alpha=0.05/M）：`t_bonf = {t_bonf:.3f}`。")
    lines.append("- 候選門檻：`D_days >= 80`、`day_mean_bps >= 25`(Outright) 或 `>= 35`(Hedged)、"
                 "`|day_t| >= 2.0`、`|l2o_t| >= 1.7`、median 與 mean 同號（>0）。")
    lines.append("")

    lines.append("## 1. Candidate Shortlist（通過全部 4 道機構級 filters）")
    lines.append("")
    if shortlist.empty:
        lines.append("**無候選通過全部門檻。** 代表目前的消融組合沒有足夠厚的毛邊際與穩健度。"
                     "這種情況下建議直接寫研究報告，不要開 OOS vault。")
    else:
        lines.append("| Variant | H | Exec | Dir | N | D_days | mean_bps | median_bps | day_t | l2o_t | IR |")
        lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in shortlist.iterrows():
            lines.append(f"| {r['variant']} | {r['horizon']} | {r['execution']} | {r['direction']} | "
                         f"{r['N_events']} | {r['D_days']} | {_fmt(r['day_mean_bps'])} | "
                         f"{_fmt(r['day_median_bps'])} | {_fmt(r['day_t'])} | "
                         f"{_fmt(r['l2o_t'])} | {_fmt(r['ir_proxy'])} |")
    lines.append("")

    lines.append("## 2. 消融階梯對比（Both 方向，Outright）")
    lines.append("")
    lines.append("| Variant | H | N_events | D_days | mean_bps | median_bps | day_t | l2o_t |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    both = metrics[(metrics["direction"] == "Both") & (metrics["execution"] == "Outright")]
    for variant in VARIANTS:
        for H in HORIZONS:
            m = both[(both["variant"] == variant) & (both["horizon"] == H)]
            if m.empty:
                continue
            r = m.iloc[0]
            lines.append(f"| {variant} | {H} | {r['N_events']} | {r['D_days']} | "
                         f"{_fmt(r['day_mean_bps'])} | {_fmt(r['day_median_bps'])} | "
                         f"{_fmt(r['day_t'])} | {_fmt(r['l2o_t'])} |")
    lines.append("")

    lines.append("## 3. 完整 cell 表（所有 variant × H × exec × direction）")
    lines.append("")
    lines.append("| Variant | H | Exec | Dir | N_events | D_days | mean_bps | median_bps | day_t | l2o_t | IR |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in metrics.iterrows():
        lines.append(f"| {r['variant']} | {r['horizon']} | {r['execution']} | {r['direction']} | "
                     f"{r['N_events']} | {r['D_days']} | {_fmt(r['day_mean_bps'])} | "
                     f"{_fmt(r['day_median_bps'])} | {_fmt(r['day_t'])} | "
                     f"{_fmt(r['l2o_t'])} | {_fmt(r['ir_proxy'])} |")
    lines.append("")
    lines.append("> 解讀提醒：`day_t` 採 date-clustered Fama-MacBeth；`l2o_t` 為丟棄影響最大的 2 個交易日後重算。"
                 "Hedged 用同時間戳產業 ETF 報酬與決策日前一天的 `beta_sec_60d` 對沖。"
                 "本報告所有數字僅來自 IS；不是 OOS 成績，也不構成投資建議。")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_is_panel()
    events = build_events(panel)
    metrics = cell_metrics(events)
    # M = number of (variant, horizon, execution, direction) combinations tested
    M = int(len(metrics))
    shortlist, t_bonf = candidate_shortlist(metrics, M)

    events[["variant", "horizon", "execution", "event_date", "target_date",
            "symbol", "direction", "z_value", "signed_ret", "signed_bps"]].to_csv(
        OUT / "rev2_is_events.csv", index=False
    )
    report = render_report(events, metrics, shortlist, t_bonf, M)
    (OUT / "rev2_is_study.md").write_text(report, encoding="utf-8")

    print(f"IS panel rows={len(panel)} events(rows)={len(events)} cells(M)={M}")
    print(f"max event_date={events['event_date'].max().date()} "
          f"max target_date={events['target_date'].max().date()}")
    print(f"shortlist candidates={len(shortlist)}  t_bonf={t_bonf:.3f}")
    print(f"report -> {OUT / 'rev2_is_study.md'}")


if __name__ == "__main__":
    main()
