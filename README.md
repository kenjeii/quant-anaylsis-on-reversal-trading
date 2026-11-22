# Pure-OHLCV Quant Research — 7-Phase Falsification Study

Built two lag-safe backtest tracks (15m opening momentum and 6-year daily cross-sectional reversal/momentum) on real Alpaca SIP data, with date-sealed IS/Locked-OOS splits, per-symbol shift isolation, full friction/cost accounting, and multiple-testing controls. Falsified every pure price/volume short-horizon hypothesis across 7 phases: no strategy survived locked True OOS. Negative results matter here because they quantify exactly where raw edge is smaller than real execution friction, and separate beta drift from genuine alpha instead of reporting a tuned discovery-window Sharpe as success.

## Highlights

| 階段 | 資料 | 結果 | 判讀 |
|---|---|---|---|
| V0 -> V_FINAL (30d Discovery) | Yahoo 15m, 12 檔 | Net 1117.59 @15bps、Sharpe 5.66、raw 59.67bps | 看似極強，實為短窗口假象 |
| V_FINAL (465d Locked OOS) | Alpaca 15m, 12 檔 | Net -3873.96 @2bps、PF 0.768、raw 3.43bps vs slippage 9.53bps | 開盤微結構吃掉 edge |
| MOM-FINAL (620d Locked OOS) | Alpaca 6Y daily, 48 檔 | M1/M2/M3 × 2 mode 全 FAIL；最佳 M2 LongOnly +6.22% vs SPY +30.39%、DSR 0.02 | 無市場中性 alpha |

Full analysis: `results/RESEARCH_FINDINGS.md`. Bug/data audit and reference tables: `results/DIAGNOSTIC_APPENDIX.md`.

## Directory Map

```
results/                     2 份最終報告（精簡、zero-fluff）
  RESEARCH_FINDINGS.md       7-phase verdict、兩條 track 的 IS->OOS 衰減
  DIAGNOSTIC_APPENDIX.md     7 bug + 3 資料異常 audit、key reference tables
src/
  intraday_15m/              15m 開盤動能 V0->V_FINAL + 2Y OOS
    paths.py                 動態 ROOT（walking up to config.yaml）
    download_alpaca.py       Alpaca SIP raw bars（credentials 只從 .env）
    clean_alpaca.py          raw -> data/alpaca_processed/
    features.py              lag-safe 15m features
    event_detector.py        分層事件偵測 + oracle outcome
    regime.py                QQQ/VIX regime
    portfolio.py             部位、費用、會計
    backtest.py              engine（Patch 0 + Phase 3 + Final fixes）
    evaluation.py            metrics
    run_2y_oos.py            -> outputs/alpaca_2y_oos_evaluation.md
  daily_swing/               48 檔 6Y daily + MOM-FINAL Locked OOS
    paths.py                 動態 ROOT
    universe.py              SECTOR_MAP（8 sectors x 6）
    download.py / clean.py   -> data/rev2_processed/
    rev2_features.py         IS-only lag-safe daily panel
    rev2_is_study.py         S0->S4 ablation -> outputs/rev2_is_study.md
    mom_final_backtest.py    M1/M2/M3 x LongOnly/SPY_Hedged simulator
    run_mom_final_oos.py     -> outputs/final_mom_oos_verdict.md
skills/quant-recursive-research/SKILL.md
                             7-Gate 遞迴研究流程 + 實戰踩坑清單（可複用）
tests/                       2 unified fast pytest files + conftest.py
data/                        *_processed/ CSVs 保留供離線重跑；raw JSON 不進 repo
```

## Reproduce (offline; processed CSVs must exist locally)

```bash
# install
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

# unified fast tests (unit + small slices, no multi-minute sweep)
.venv/Scripts/python -m pytest -q

# Track 1 — 15m 2Y True OOS (reads data/alpaca_processed/)
.venv/Scripts/python -m src.intraday_15m.run_2y_oos

# Track 2 — daily IS ablation (hard-sealed at 2024-05-31)
.venv/Scripts/python -m src.daily_swing.rev2_is_study

# Track 2 — daily 2.5Y LOCKED OOS verdict
.venv/Scripts/python -m src.daily_swing.run_mom_final_oos
```

All run scripts are 100% offline. Only `download*.py` ever touches APIs, and only those read `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` from environment or local `.env` (never hardcoded, never committed).

## Research Workflow

The recursive debugging/validation protocol used across all 7 phases is extracted as a reusable agent skill at `skills/quant-recursive-research/SKILL.md`. It covers the 7 gates from accounting audit through one-shot OOS verdict and hard kill criteria.

> 免責聲明：以上為量化研究紀錄，僅供參考，不構成投資建議。過往表現不預示未來收益。
