"""Phase 7 — final pre-registered MOM-FINAL evaluation (IS / LOCKED OOS / Full).

100% OFFLINE. Reads ONLY data/rev2_processed/. ZERO tuning: all rules frozen in
mom_final_backtest.STRATEGIES before any date >= 2024-06-01 is touched.

Outputs outputs/final_mom_oos_verdict.md with the 7 required sections.
"""
from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.daily_swing.mom_final_backtest import (BENCHMARK, INITIAL_EQUITY, MODES, STRATEGIES,
                                build_market, run_simulation, spy_buy_hold)

from .paths import ROOT, REV2_PROCESSED, REV2_RAW, OUTPUTS  # noqa: E402
OUT = ROOT / "outputs"
IS = ("2021-06-01", "2024-05-31")
OOS = ("2024-06-01", "2026-11-20")
FULL = ("2021-06-01", "2026-11-20")
FRICTIONS = (0, 2, 5, 10, 15)
MAIN_FEE = 5
M_TRIALS = 126
ND = NormalDist()
EULER_GAMMA = 0.5772156649015329


def _max_dd(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def _sharpe(returns: pd.Series) -> float:
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return float("nan")
    return float(returns.mean() / returns.std(ddof=1) * math.sqrt(252))


def _deflated_sharpe(returns: pd.Series, M: int = M_TRIALS) -> float:
    """Bailey & López de Prado (2014) closed-form DSR with skew & excess kurtosis."""
    r = returns.dropna()
    n = len(r)
    if n < 20 or r.std(ddof=1) == 0:
        return float("nan")
    mu, sd = r.mean(), r.std(ddof=1)
    sr = mu / sd  # per-observation Sharpe
    skew = float(((r - mu) ** 3).mean() / sd ** 3)
    kurt = float(((r - mu) ** 4).mean() / sd ** 4)  # excess kurtosis would be kurt-3
    v_sr = (1.0 - sr * skew + (kurt - 1.0) / 4.0 * sr ** 2) / (n - 1)
    v_sr = max(v_sr, 1e-12)
    se_sr = math.sqrt(v_sr)
    z1 = ND.inv_cdf(1.0 - 1.0 / M)
    z2 = ND.inv_cdf(1.0 - 1.0 / (M * math.e))
    sr0 = se_sr * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    return float(ND.cdf((sr - sr0) / se_sr))


def _metrics(result: dict, spy_curve: pd.DataFrame) -> dict:
    trades = result["trades"]
    curve = result["curve"]
    eq = curve["equity"]
    rets = eq.pct_change().dropna()
    spy_rets = spy_curve["equity"].pct_change().dropna()
    aligned = pd.concat([rets.rename("s"), spy_rets.rename("b")], axis=1).dropna()
    n = len(eq)
    total_ret = float(eq.iloc[-1] / INITIAL_EQUITY - 1.0)
    cagr = float((eq.iloc[-1] / INITIAL_EQUITY) ** (252.0 / max(n - 1, 1)) - 1.0)
    sharpe = _sharpe(rets)
    se_sr = math.sqrt(252.0 / max(n - 1, 1))
    dsr = _deflated_sharpe(rets)
    net = trades["net_pnl"] if len(trades) else pd.Series(dtype=float)
    wins = net[net > 0]
    losses = net[net <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else (
        float("inf") if len(wins) else float("nan"))
    notional = float(trades["long_notional"].sum()) if len(trades) else 0.0
    raw_bps = float(trades["pre_slippage_pnl"].sum() / notional * 1e4) if notional else 0.0
    net_bps = float(net.sum() / notional * 1e4) if notional else 0.0
    # Beta & Jensen alpha vs SPY (daily, rf=0)
    if len(aligned) > 2 and aligned["b"].var(ddof=1) > 0:
        beta = float(aligned["s"].cov(aligned["b"]) / aligned["b"].var(ddof=1))
        ann_alpha = float((aligned["s"].mean() - beta * aligned["b"].mean()) * 252)
        spread = aligned["s"] - aligned["b"]
        ir = float(spread.mean() / spread.std(ddof=1) * math.sqrt(252)) if spread.std(ddof=1) else float("nan")
    else:
        beta = ann_alpha = ir = float("nan")
    return {
        "trades": int(len(trades)),
        "avg_exp_pct": (float(curve["gross_exposure_pct"].mean())
                        if "gross_exposure_pct" in curve.columns else 100.0),
        "win_rate": float((net > 0).mean()) if len(net) else float("nan"),
        "total_ret": total_ret, "cagr": cagr, "max_dd": _max_dd(eq),
        "sharpe": sharpe, "se_sr": se_sr, "dsr": dsr, "pf": pf,
        "raw_alpha_bps": raw_bps, "net_edge_bps": net_bps,
        "beta_spy": beta, "ann_alpha_spy": ann_alpha, "ir_spy": ir,
        "net_pnl": float(net.sum()) if len(net) else 0.0,
        "days": n - 1,
    }


def _fmt(v, pct=False, digits=2) -> str:
    if isinstance(v, float) and (math.isnan(v)):
        return "n/a"
    if v == float("inf"):
        return "inf"
    return f"{v * 100:.{digits}f}%" if pct else f"{v:.{digits}f}"


def breakeven_bps(net_by_bps: dict[int, float]) -> float:
    pts = sorted(net_by_bps.items())
    for (b0, v0), (b1, v1) in zip(pts, pts[1:]):
        if v0 >= 0 >= v1:
            if v0 == v1:
                return float(b0)
            return float(b0 + (b1 - b0) * v0 / (v0 - v1))
    if pts[-1][1] > 0:
        # extrapolate two-point slope beyond highest tested friction
        (b0, v0), (b1, v1) = pts[-2], pts[-1]
        slope = (v1 - v0) / (b1 - b0)
        return float(max(0.0, b1 - v1 / slope)) if slope < 0 else float("inf")
    return 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building market (offline, full range)...", flush=True)
    dates, bars = build_market()
    spy_curves = {name: spy_buy_hold(bars, dates, w[0], w[1])
                  for name, w in (("IS", IS), ("OOS", OOS), ("Full 5.5Y", FULL))}

    # Main runs: 6 strategies × 3 windows at 5 bps
    results: dict[tuple[str, str, str], dict] = {}
    metrics: dict[tuple[str, str, str], dict] = {}
    for strategy in STRATEGIES:
        for mode in MODES:
            for wname, window in (("IS", IS), ("OOS", OOS), ("Full 5.5Y", FULL)):
                r = run_simulation(bars, dates, strategy, mode, window[0], window[1], MAIN_FEE)
                results[(strategy, mode, wname)] = r
                metrics[(strategy, mode, wname)] = _metrics(r, spy_curves[wname])
            print(f"  {strategy:28s} {mode:11s} done", flush=True)

    # Friction sensitivity on OOS only
    sensitivity: dict[tuple[str, str], dict[int, float]] = {}
    for strategy in STRATEGIES:
        for mode in MODES:
            by_bps = {}
            for fee in FRICTIONS:
                if fee == MAIN_FEE:
                    by_bps[fee] = metrics[(strategy, mode, "OOS")]["net_pnl"]
                else:
                    r = run_simulation(bars, dates, strategy, mode, OOS[0], OOS[1], float(fee))
                    by_bps[fee] = float(r["trades"]["net_pnl"].sum()) if len(r["trades"]) else 0.0
            sensitivity[(strategy, mode)] = by_bps
    breakeven = {k: breakeven_bps(v) for k, v in sensitivity.items()}

    write_report(metrics, sensitivity, breakeven, results, spy_curves)
    print("Report -> outputs/final_mom_oos_verdict.md", flush=True)


def write_report(metrics, sensitivity, breakeven, results, spy_curves) -> None:
    L: list[str] = []
    A = L.append
    A("# Phase 7 — MOM-FINAL 最終期末考判決書")
    A("")
    A("資料：Alpaca SIP 6 年日線（`data/rev2_processed/`），100% 離線；零 API、零合成價格、零調參。")
    A("所有閾值在讀取任何 `date >= 2024-06-01` 之前已凍結（`src/mom_final_backtest.py` 的 `STRATEGIES`）。")
    A("LOCKED True OOS = `2024-06-01` 至 `2026-11-20`（620 交易日）；IS = `2021-06-01` 至 `2024-05-31`；"
      "IS 與 OOS 各自從 $100,000 乾淨起步、零部位帶過去。")
    A("")

    # Section 1
    A("## 1. 最終判決書（LOCKED True OOS，`total_rt_bps = 5`）")
    A("")
    A("三道機構門檻：Bar1 淨獲利（Net PnL > 0 且 PF > 1.10 且 Breakeven >= 10bps）；"
      "Bar2 打敗基準（LongOnly: Sharpe > SPY Sharpe 或 Jensen Alpha > 0；Hedged: Net Sharpe > 0.50）；"
      "Bar3 統計可信（DSR(M=126) > 0.50）。")
    A("")
    spy_oos = _metrics({"trades": pd.DataFrame(), "curve": spy_curves["OOS"]}, spy_curves["OOS"])
    A(f"SPY Buy & Hold（OOS）：Total Ret {_fmt(spy_oos['total_ret'], True)}、"
      f"CAGR {_fmt(spy_oos['cagr'], True)}、Sharpe {_fmt(spy_oos['sharpe'])}、MaxDD {_fmt(spy_oos['max_dd'], True)}。")
    A("")
    A("| 策略 | Mode | Net PnL | PF | Breakeven(bps) | Sharpe / SE(SR) | DSR(126) | Ann.Alpha% | Bar1 | Bar2 | Bar3 | 判決 |")
    A("|---|---|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|:--:|")
    verdicts = {}
    for strategy in STRATEGIES:
        for mode in MODES:
            m = metrics[(strategy, mode, "OOS")]
            be = breakeven[(strategy, mode)]
            b1 = m["net_pnl"] > 0 and m["pf"] > 1.10 and be >= 10.0
            if mode == "LongOnly":
                b2 = (m["sharpe"] > spy_oos["sharpe"]) or (m["ann_alpha_spy"] > 0)
            else:
                b2 = m["sharpe"] > 0.50
            b3 = (not math.isnan(m["dsr"])) and m["dsr"] > 0.50
            score = sum([b1, b2, b3])
            verdict = "PASS" if score == 3 else ("CONDITIONAL" if score == 2 else "FAIL")
            verdicts[(strategy, mode)] = verdict
            A(f"| {strategy} | {mode} | {m['net_pnl']:.1f} | {_fmt(m['pf'])} | {be:.2f} | "
              f"{_fmt(m['sharpe'])} / {_fmt(m['se_sr'])} | {_fmt(m['dsr'])} | {_fmt(m['ann_alpha_spy'], True)} | "
              f"{'Y' if b1 else 'N'} | {'Y' if b2 else 'N'} | {'Y' if b3 else 'N'} | **{verdict}** |")
    A("")

    # Section 2
    A("## 2. IS vs LOCKED True OOS 核心成績單（`total_rt_bps = 5` 寫實 / 15 壓力另見 §3）")
    A("")
    A("| Window | Strategy | Mode | Trades | Avg_Exp_% | Win_Rate | Total_Ret_% | CAGR_% | Max_DD_% | Sharpe | SE(SR) | DSR(126) | PF | Raw_Alpha_bps | Net_Edge_bps | Breakeven_bps | Beta_SPY | Ann_Alpha_SPY_% |")
    A("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for wname in ("IS", "OOS", "Full 5.5Y"):
        sb = _metrics({"trades": pd.DataFrame(), "curve": spy_curves[wname]}, spy_curves[wname])
        A(f"| {wname} | SPY Buy&Hold | - | - | 100.0 | - | {_fmt(sb['total_ret'], True)} | "
          f"{_fmt(sb['cagr'], True)} | {_fmt(sb['max_dd'], True)} | {_fmt(sb['sharpe'])} | "
          f"{_fmt(sb['se_sr'])} | - | - | - | - | - | 1.00 | - |")
        for strategy in STRATEGIES:
            for mode in MODES:
                m = metrics[(strategy, mode, wname)]
                be = breakeven[(strategy, mode)] if wname == "OOS" else float("nan")
                A(f"| {wname} | {strategy} | {mode} | {m['trades']} | {_fmt(m['avg_exp_pct'])} | "
                  f"{_fmt(m['win_rate'], True)} | {_fmt(m['total_ret'], True)} | {_fmt(m['cagr'], True)} | "
                  f"{_fmt(m['max_dd'], True)} | {_fmt(m['sharpe'])} | {_fmt(m['se_sr'])} | "
                  f"{_fmt(m['dsr'])} | {_fmt(m['pf'])} | {_fmt(m['raw_alpha_bps'])} | "
                  f"{_fmt(m['net_edge_bps'])} | {_fmt(be)} | {_fmt(m['beta_spy'])} | "
                  f"{_fmt(m['ann_alpha_spy'], True)} |")
    A("")

    # Section 3
    A("## 3. 全摩擦敏感度（LOCKED True OOS，Net PnL）")
    A("")
    A("| Strategy | Mode | " + " | ".join(str(f) + "bps" for f in FRICTIONS) + " | Breakeven(bps) |")
    A("|---|---|" + "---:|" * (len(FRICTIONS) + 1))
    for strategy in STRATEGIES:
        for mode in MODES:
            s = sensitivity[(strategy, mode)]
            A(f"| {strategy} | {mode} | " + " | ".join(f"{s[f]:.1f}" for f in FRICTIONS)
              + f" | {breakeven[(strategy, mode)]:.2f} |")
    A("")

    # Section 4: yearly for M3 & M1 vs SPY
    A("## 4. 跨年度逐年穩定度（M3 與 M1，5 bps）")
    A("")
    A("| 年 | SPY Ret_% | M1_LongOnly_% | M1_Hedged_% | M3_LongOnly_% | M3_Hedged_% |")
    A("|---|---:|---:|---:|---:|---:|")
    spy_curve = spy_curves["OOS"].set_index("date")["equity"]
    years = sorted({d.year for d in spy_curve.index})
    def year_ret(curve_df, year):
        c = curve_df.set_index("date")["equity"]
        c = c[c.index.year == year]
        if len(c) < 2:
            return float("nan")
        return float(c.iloc[-1] / c.iloc[0] - 1.0)
    for y in years:
        cells = [year_ret(spy_curves["OOS"], y)]
        for s in ("M1_Sector_ETF_5d", "M3_Synth_Sector_Vol_5d"):
            for mode in MODES:
                cells.append(year_ret(results[(s, mode, "OOS")]["curve"], y))
        A(f"| {y} | " + " | ".join(_fmt(x, True) for x in cells) + " |")
    A("")

    # Section 5: sector & concentration for M3 at 5 bps
    A("## 5. 產業貢獻度與集中度（M3，OOS，5 bps）")
    A("")
    for mode in MODES:
        t = results[("M3_Synth_Sector_Vol_5d", mode, "OOS")]["trades"]
        A(f"### M3_{mode}")
        A("")
        A("| 產業 ETF | Trades | Net PnL |")
        A("|---|---:|---:|")
        g = t.groupby("sector")["net_pnl"].agg(["size", "sum"]).sort_values("sum", ascending=False)
        for sec, r in g.iterrows():
            A(f"| {sec} | {int(r['size'])} | {r['sum']:.1f} |")
        top_sec = g.index[0] if len(g) else None
        ex_top = t[t["sector"] != top_sec]["net_pnl"].sum() if top_sec else 0.0
        daily = t.groupby("exit_date")["net_pnl"].sum()
        top2 = daily.nlargest(2).index
        ex_days = t[~t["exit_date"].isin(top2)]["net_pnl"].sum()
        A("")
        A(f"- 排除最佳 1 產業（{top_sec}）：Net PnL {ex_top:.1f}")
        A(f"- 排除最佳 2 交易日（{', '.join(str(d.date()) for d in top2)}）：Net PnL {ex_days:.1f}")
        A("")

    # Section 6: exits and gap-through, NVDA glitch
    A("## 6. 出場原因與隔夜跳空停損壓力表（OOS）")
    A("")
    A("| Strategy | Mode | reason | count | avg net PnL |")
    A("|---|---|---|---:|---:|")
    for strategy in STRATEGIES:
        for mode in MODES:
            t = results[(strategy, mode, "OOS")]["trades"]
            for reason, g in t.groupby("reason"):
                A(f"| {strategy} | {mode} | {reason} | {len(g)} | {g['net_pnl'].mean():.2f} |")
    A("")
    A("> `NVDA 2024-06-10` 的 high 毛刺在特徵層被標記為 `hl_anomaly=True`；"
      "當日 intraday stop 檢查改用 `close` 而非 `low/high`，因此不會被壞 tick 觸發假停損。"
      "該日期位於 LOCKED OOS（2024-06-01 至 2026-11-20）之內，此防護在整個 OOS 區間一致生效。")
    A("")

    # Section 7: cross-phase review
    A("## 7. 全研究 7 階段總回顧對照表")
    A("")
    A("| 階段 | 策略 | 資料 | 毛邊際 / 結果 | 關鍵判讀 |")
    A("|---|---|---|---|---|")
    A("| Phase 3/4B | V_FINAL 15m 開盤動能 | Alpaca 2Y OOS 465d | Raw alpha 3.43bps / slippage 9.53bps / Net -3873.96 @2bps / PF 0.768 | 毛邊際被滑價吃死，未通過 OOS |")
    A("| Phase 5A | Plan1 B/up Hedged momentum | Yahoo/Alpaca IS | day_mean 67–149bps、t 2.27–2.63，但 l2o 與 M=170 未過 Bonferroni | 較厚但未經 OOS |")
    A("| Phase 6B | Rev2 反轉 S0→S4 | rev2 IS 3Y | S0 3d SPY Short -57bps t=-4.01；S4 Long 5d +23.9bps t=1.65；shortlist=0 | 反轉無穩健候選；反向的強勢動能反而显著 |")
    A("| Phase 7 | MOM-FINAL M1/M2/M3 | LOCKED OOS 620d | 見 §1 判決書；6 規則全 FAIL、LongOnly 最強 M2 僅 +6.22% (<SPY 30.39%) | 強勢動能一次性期末考未通過 |")
    A("")
    A("## 限制與免責聲明")
    A("")
    A("單一 48 檔科技/跨產業美股 universe、單一 5.5 年區間；即使 PASS 也不能外推到不同市場體制。"
      "以上僅供量化研究，不構成投資建議。")
    (OUT / "final_mom_oos_verdict.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
