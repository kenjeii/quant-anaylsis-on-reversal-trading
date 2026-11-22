# 7-Phase Quantitative Research Findings

全部數字摘錄自既有 `outputs/*.md`，零合成、零改寫。資料：Alpaca SIP 15m/1Day + Yahoo VIX，2016-11 至 2026-11。

## 1. Executive Verdict

- 7 個階段、2 至 48 檔美股、最長 6 年日線與 2 年 15m，**沒有任何純 OHLCV 短週期策略在 Locked True OOS 存活**。
- 天花板一（微結構）：15m 開盤動能 raw alpha `3.43 bps` 小於開盤滑價 `9.53 bps`，遲早被摩擦吃死。
- 天花板二（反轉）：48 檔 6 年日線 3 至 5 日反轉消融 `S0` 至 `S4` 得到 `0/120` 通過候選。
- 天花板三（動能 IS 至 OOS 衰減）：唯一通過 Bonferroni 的 IS 訊號（`t=4.01 > 3.529`）在 620 天 Locked OOS 衰減至 Jensen alpha `+0.03%`、`DSR <= 0.02`（`M2 LongOnly +6.22%` 對 `SPY +30.39%`）。
- 6 條 `MOM-FINAL` 凍結規則在 Locked OOS **6/6 FAIL**；LongOnly 正 PnL 完全是 `0.24` 的 SPY beta，SPY_Hedged 全數為負。

## 2. 7-Phase Progression Summary

| 階段 | 資料 / 窗口 | 核心結果 | 判讀 |
|---|---|---|---|
| V0 | Yahoo 30d，2 檔，15m | 15 bps 下 0 筆交易、Net PnL 0.00 | 所有 AND 濾網過嚴，無交易 |
| V_FINAL Discovery | Yahoo 30d，12 檔 | 15 bps Net 1117.59、2 bps Net 1576.43、breakeven 46.83 bps、Sharpe 5.66、raw 59.67 bps | 30 天看起來極強，屬假象 |
| V_FINAL 2Y OOS | Alpaca 465d，12 檔 | 2 bps Net -3873.96、PF 0.768、Sharpe -1.402、raw 3.43 bps / slippage 9.53 bps | Discovery illusion 崩潰 |
| Plan 1 IS | Yahoo/Alpaca 1.5Y | 最強 B/up Hedged D5 raw 149.37 bps、t 2.627、l2o 2.206；M=170、t_bonf 3.620 | 較厚但未過 Bonferroni，未經 OOS |
| Rev 2.0 IS | rev2 3Y，48 檔 | S0 強勢股做空 -57.05 bps / t -4.01；S4 Long 5d +23.92 bps / t 1.65；shortlist 0 | 反轉無穩健候選；反向動能才是真訊號 |
| MOM-FINAL OOS | rev2 620d Locked | M1/M2/M3 × 2 mode 全 FAIL；M2 LongOnly +6.22%、SPY +30.39%、DSR 0.02 | 強勢動能期末考未通過 |

## 3. Track 1 — Intraday 15m (`V0` -> `V_FINAL` -> 2Y OOS)

### 被遮蔽的 edge 與假 Sharpe

- 6 個沉默引擎 Bug（ATR scale mismatch、directional regime OR bug、CAR/raw 方向衝突、`_plan_trade` impulse/短線 TP cap、entry slippage 反號、four-bar residual 與 1.4826 MAD factor）加上 1 個 off-by-one（entry bar 未監控、`holding_bars` 少算一根）→ 真實 edge 被壓住，同時在 30 天 Discovery 製造 `Sharpe = 5.66` 的錯覺。
- Discovery raw `+59.67 bps` 對 True OOS raw `+3.43 bps`，差 17 倍；同一套凍結規則，唯一差別是資料窗口（此為 Discovery illusion）。

### 摩擦瀑布（V_FINAL，2 bps）

| 窗口 | Pre-Slip PnL | Slippage | Fees+Borrow | Net PnL | Raw alpha (bps) | Slippage (bps) | Net edge (bps) | Breakeven (bps) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Discovery 30d | 2102.77 | 428.55 | 72.27 | 1601.95 | 59.67 | 12.16 | 45.45 | 47.46 |
| True OOS 465d | 1631.34 | 4534.14 | 971.16 | -3873.96 | 3.43 | 9.53 | -8.14 | 0.00 |

- 0 bps 手續費下 OOS 仍為 `-2937.24`：raw alpha 3.43 bps 已被 9.53 bps 滑價吃光，問題不是佣金而是開盤微結構。
- 損益兩平手續費：Discovery `47.46 bps` -> True OOS `0.00 bps`。

### 微結構診斷與集中度

- High RVOL（開盤量能）在 OOS 造成 >60% 虧損：`High / <0.5 efficiency` N=53、Net -970.05、PF 0.643；`High / >=0.5` N=84、Net -1373.33、PF 0.673。
- Gap-and-Go 仍為負（N=275、Net -2166.74），Pure Open Drive 才勉强正（N=26、Net +118.43、PF 1.153），但樣本過小。
- 個股：GOOGL 唯一正（`+526.24`），AMD 最差（`-976.28`）；半導體群 `-2569.78`、大型科技 `-1304.18`。排除最佳 2 日後 `-4799.31`（比原始更差，代表不是少數極端日造成虧損）。

## 4. Track 2 — 6Y Daily Cross-Sectional (`Rev 2.0` IS -> `MOM-FINAL` OOS)

### 反轉消融為何失敗

- `S0` SPY 殘差 3d 做空強勢股：`-57.05 bps`、`t=-4.01`（即反向做多才是真訊號，但反轉框架把方向翻轉成做空）。
- `S4` 5d 做多超賣：outright `+23.92 bps`、`t=1.65`、`l2o=1.86`；但 sector-hedged 降至 `-4.64 bps`，證明所謂反彈其實是市場漂移，不是個股 alpha。
- `S0 -> S4` 任何 cell 都沒同時過 `D_days>=80`、`mean>=25/35 bps`、`|t|>=2.0`、`|l2o_t|>=1.7`，shortlist `0/120`。
- `S2` 與 `S3` 完全相同：VIX 逆價差（`ratio>=1.05`）96/43200 列從未與 S2 事件重疊；恐慌過濾在 IS 無作用。

### MOM-FINAL Locked OOS 為何失敗（620d，5 bps）

| 策略 | LongOnly Net | LongOnly Ret | Sharpe | DSR(126) | SPY_Hedged Net | Breakeven |
|---|---:|---:|---:|---:|---:|---:|
| M1 板塊 ETF | +2728.2 | +2.73% | 0.18 | 0.01 | -2989.3 | 9.52 bps |
| M2 個股冠軍 | +6432.5 | +6.22% | 0.29 | 0.02 | -4259.5 | 13.70 bps |
| M3 終極綜合 | +2739.2 | +2.53% | 0.16 | 0.01 | -5762.5 | 10.20 bps |

- 同期 SPY Buy&Hold `+30.39%`、Sharpe 0.74、MaxDD -19.00%；三條 LongOnly 的 beta 只有 `0.18` 至 `0.24`，即只吃到四分之一的市場報酬。
- SPY_Hedged 全數為負（`-2.99%` 至 `-5.81%`），證明 LongOnly 正 PnL 沒有市場中性 alpha；Jensen alpha M1 -1.52%、M2 +0.03%、M3 -0.87%。
- 加濾網反而更差：M3（板塊+放量+VIX）OOS `+2.53%` 低於單純 M2 `+6.22%`；M3_SPY_Hedged raw alpha 已 `-1.68 bps`。
- `3.0σ` 寬災難停損仍放血：M2 有 27 筆隔夜跳空停損（平均 `-$927.16`）與 106 筆盤中停損（平均 `-$541.02`），time_barrier 才 `+$156.09`。
- 產業集中：M3_LongOnly 的 XLF `-3488.6`；排除最佳產業 XLE 後 Net `-200.3`；排除最佳 2 日後 `-2034.1`。
- 逐年不穩：M3_LongOnly 2025 `-4.20%`、2026 `+6.45%`；M3_Hedged 2025 `-7.78%`。

### 摩擦敏感度（OOS Net PnL）

| 策略 / Mode | 0bps | 2bps | 5bps | 10bps | 15bps |
|---|---:|---:|---:|---:|---:|
| M2 LongOnly | 9987.1 | 8768.2 | 6432.5 | 2466.3 | -862.1 |
| M2 Hedged | 2397.9 | -367.9 | -4259.5 | -10362.3 | -16164.5 |
| M3 LongOnly | 5378.2 | 4407.5 | 2739.2 | 104.9 | -2531.3 |

## 5. Actionable Implications

- 純 OHLCV 的日內／3-5 日 cross-section 在大型科技股已被定價；下一條路應換 universe 而非加濾網：流動性足夠但較不擁擠的中型股或 maker-execution 環境。
- 若要抓數週至數月的不對稱，應換 horizon 與 asset class（cross-asset 1-6m 宏觀趨勢），而不是在 15m 至 5d 繼續微調。
- 要打穿摩擦必須引入非價格催化或選擇權 VRP 等二階資訊；純價格在 5 bps 來回成本下邊際太薄。
- 任何新假說都先跑 Gate 2 事件研究確認 raw vs hedged 方向與 `h*`，再談 backtest；`IS -> OOS` 一但衰減就結案，不要調參救。
