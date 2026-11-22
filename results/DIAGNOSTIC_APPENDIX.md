# Diagnostic Appendix

純參考 tables，無贅字。來源：`outputs/*.md`、`tests/verify_*.py`。

## 1. Engine & Data Audit Log

### 1.1 已修 7 個 Code Bug

| # | Bug | 症狀 | 修法 |
|---|---|---|---|
| 1 | ATR timeframe scale mismatch | stop-bound 用 15m ATR 但風險用日線 scale | 統一 `stop_bound_reference="atr_15m"` |
| 2 | Directional regime OR bug | 不分多空一律禁令，momentum 被雙重處罰 | `_is_trade_prohibited` 帶 direction；`use_regime_multiplier=False` 可關 |
| 3 | CAR / raw 方向衝突 | 原始報酬與 abnormal return 反向仍進場 | 方向不一致直接 rejection |
| 4 | `_plan_trade` impulse / 短線 TP cap | signed RR 錯號、short TP 被 cap | signed RR 與 short TP 邊界修正 |
| 5 | Entry slippage 反號 | short entry 成本方向反了 | adverse sign 修正 |
| 6 | four-bar residual 用差分；MAD 少 1.4826 | 特徵分布錯、Z-score 被低估 | 改 rolling sum；乘 1.4826 |
| 7 | Off-by-one entry bar 未監控 | `holding_bars` 少一根，2-bar trade 被 TP1 干擾 | `monitor_entry_bar=True` + `tb_single_exit=True` |

同時修正 oracle chronology look-ahead、MTM equity 忽略現價、cooldown/day-lock 時點過早、extension 用前一 impulse、`retracement/retrace` 不一致等。

### 1.2 真實 Vendor 資料異常

| 標的 / 日期 | 異常 | 處理 |
|---|---|---|
| HON 2026-06-29 | Alpaca `adjustment=split` 50.95% 假 2:1 跳價（前收 464 -> 227） | manifest `split_override` 記錄，改用真實 `adjustment=all` bars（11.19%），零合成價格 |
| NVDA 2024-06-10 | 單根 daily `high=195.95`，open/low/close 約 117-122（65.58% HL 異常，位於 Locked OOS） | `hl_anomaly=True` 標記不竄改；intraday stop 改用 close，避免假停損 |
| TSLA 2021-03-04 | 單根 `high=291.31`，O/C 約 207-218 | 同上 |
| MRK 2021-06-11 | 單根 `low=15.32`，O/H/C 約 76.3 | 同上 |

其他 split 檢查：57 檔可成交價格轉換單根跳價全 <45%（最大 AMD 37.55%）。

## 2. Track 1 — 2Y True OOS Key Tables（2 bps / 15 bps）

### 2.1 核心成績（True OOS 465d）

| 版本 | Fee | Trades | Win | Net PnL | PF | Sharpe | Raw alpha | Slippage | Net edge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2_Baseline_12sym | 2 | 492 | 46.1% | -3916.60 | 0.795 | -1.330 | 2.76 | 7.53 | -6.98 |
| V_FINAL | 2 | 410 | 46.3% | -3873.96 | 0.768 | -1.402 | 3.43 | 9.53 | -8.14 |
| V_FINAL_h3 | 2 | 409 | 46.5% | -2401.16 | 0.870 | -0.714 | 6.56 | 9.54 | -5.04 |
| V2_Baseline_12sym | 15 | 494 | 38.9% | -10644.75 | 0.527 | -3.673 | 3.21 | 7.53 | -19.53 |
| V_FINAL | 15 | 409 | 39.1% | -9649.37 | 0.510 | -3.517 | 3.58 | 9.52 | -20.98 |
| V_FINAL_h3 | 15 | 408 | 40.9% | -8433.22 | 0.603 | -2.604 | 6.25 | 9.55 | -18.35 |

### 2.2 摩擦瀑布（V_FINAL，fee bps -> Net PnL）

| Fee | 0 | 1 | 2 | 5 | 10 | 15 |
|---|---:|---:|---:|---:|---:|---:|
| V2 Net | -2619.80 | -3369.13 | -3916.60 | -5371.01 | -8183.50 | -10644.75 |
| V_FINAL Net | -2937.24 | -3407.02 | -3873.96 | -5288.74 | -7549.06 | -9649.37 |

### 2.3 季度穩定度（V_FINAL，2 bps）

| Quarter | QQQ | 市況 | Trades | Net PnL | PF |
|---|---:|---|---:|---:|---:|
| 2024Q4 | -0.8% | sideways | 13 | -440.75 | 0.551 |
| 2025Q1 | -8.1% | bear | 62 | -379.88 | 0.838 |
| 2025Q2 | +16.7% | bull | 51 | +442.34 | 1.310 |
| 2025Q3 | +9.8% | bull | 45 | -952.50 | 0.582 |
| 2025Q4 | +1.8% | sideways | 56 | -895.95 | 0.661 |
| 2026Q1 | -5.9% | bear | 61 | -594.20 | 0.733 |
| 2026Q2 | +26.0% | bull | 69 | -487.27 | 0.823 |
| 2026Q3 | +2.0% | sideways | 50 | -424.17 | 0.775 |
| 2026Q4 | -18.0% | bear | 40 | +1460.35 | 2.706 |

### 2.4 Long / Short（True OOS，2 bps）

| 方向 | Trades | Win | Pre-Slip | Net PnL | PF |
|---|---:|---:|---:|---:|---:|
| long | 222 | 45.0% | 879.85 | -2078.41 | 0.777 |
| short | 188 | 47.9% | 751.49 | -1795.55 | 0.757 |

## 3. Track 2 — Locked 2.5Y OOS Key Tables（5 bps）

### 3.1 核心成績（OOS 620d）

| 策略 | Mode | Trades | Win | Total Ret | Sharpe | DSR(126) | PF | Net PnL | Breakeven |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SPY Buy&Hold | - | - | - | +30.39% | 0.74 | - | - | - | - |
| M1 | LongOnly | 301 | 52.16% | +2.73% | 0.18 | 0.01 | 1.04 | +2728.2 | 9.52 |
| M1 | Hedged | 301 | 49.50% | -2.99% | -0.21 | 0.00 | 0.94 | -2989.3 | 2.36 |
| M2 | LongOnly | 702 | 50.14% | +6.22% | 0.29 | 0.02 | 1.05 | +6432.5 | 13.70 |
| M2 | Hedged | 702 | 50.14% | -4.31% | -0.17 | 0.00 | 0.96 | -4259.5 | 1.73 |
| M3 | LongOnly | 507 | 48.32% | +2.53% | 0.16 | 0.01 | 1.03 | +2739.2 | 10.20 |
| M3 | Hedged | 507 | 47.34% | -5.81% | -0.30 | 0.00 | 0.93 | -5762.5 | 0.00 |

### 3.2 摩擦敏感度（Net PnL）

| 策略 / Mode | 0bps | 2bps | 5bps | 10bps | 15bps |
|---|---:|---:|---:|---:|---:|
| M1 LongOnly | 5917.6 | 4582.5 | 2728.2 | -292.4 | -3260.6 |
| M1 Hedged | 2764.1 | 408.8 | -2989.3 | -8573.5 | -13754.7 |
| M2 LongOnly | 9987.1 | 8768.2 | 6432.5 | 2466.3 | -862.1 |
| M2 Hedged | 2397.9 | -367.9 | -4259.5 | -10362.3 | -16164.5 |
| M3 LongOnly | 5378.2 | 4407.5 | 2739.2 | 104.9 | -2531.3 |
| M3 Hedged | -1204.5 | -3225.0 | -5762.5 | -10546.9 | -14475.5 |

### 3.3 出場與隔夜跳空停損

| 策略 / Mode | time_barrier | stop_intraday | stop_gapped_through | gap-through avg PnL |
|---|---:|---:|---:|---:|
| M1 LongOnly | 250 | 44 | 7 | -929.05 |
| M1 Hedged | 250 | 44 | 7 | -376.57 |
| M2 LongOnly | 569 | 106 | 27 | -927.16 |
| M2 Hedged | 569 | 106 | 27 | -711.33 |
| M3 LongOnly | 410 | 81 | 16 | -839.96 |
| M3 Hedged | 410 | 81 | 16 | -699.70 |

跳空成交：`open * (1 - slip)` 且 `reason=stop_gapped_through`（27 筆 M2 OOS 實測），非 theoretical stop price。融券按日曆天計息（週五至週一 = 3 天）。

### 3.4 產業集中度（M3 LongOnly，5 bps）

| ETF | Net PnL | ETF | Net PnL |
|---|---:|---|---:|
| XLE | +2939.5 | SMH | +94.6 |
| XLV | +2874.5 | XLI | -33.3 |
| XLC | +662.7 | XLK | -848.7 |
| XLY | +538.3 | XLF | -3488.6 |

排除最佳產業 XLE 後 Net `-200.3`；排除最佳 2 交易日後 `-2034.1`。
