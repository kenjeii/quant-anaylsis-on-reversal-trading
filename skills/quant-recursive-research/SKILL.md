---
name: quant-recursive-research
description: Use when debugging a quantitative trading strategy, diagnosing zero or undertrading, iterating from results into a revised plan, validating an event study with date-clustered statistics, attributing PnL to frictions and beta, or auditing an out-of-sample backtest for leakage and overfitting. Applies to purely numerical OHLCV research and multi-phase strategy development where each result must feed the next frozen hypothesis.
---

# Quant Recursive Research

Run quantitative strategy work as a gated recursion where each result either eliminates a hypothesis or earns a stricter out-of-sample test. Never tune on data that will later be used to claim out-of-sample validity.

## When to Use

- A backtest produces zero trades, implausible Sharpe, or PnL that collapses out of sample.
- An event study must decide true direction and horizon before a backtest is written.
- Multiple variants, filters, or horizons are compared and multiple-testing bias must be controlled.
- A vendor feed shows suspicious prices or a symbol boundary is suspected of leaking features.
- A strategy must be accepted or killed by a single pre-registered out-of-sample verdict.

## Core Recursive Loop (7-Gate Protocol)

Each phase ends at a gate. Do not skip forward.

### Gate 0. Code and Accounting Audit Before Tuning

- Fix scale mismatches before interpreting any metric (indicator computed on one timeframe but applied to another).
- Verify every slippage sign against trade direction. Adverse fill means long entries pay up and short entries sell down.
- Replace non-directional regime blocks with direction-aware checks. A blanket block that punishes both long and short is a bug, not a filter.
- Enforce bar processing order explicitly: exits at open, then entries at open, then intraday stops, then mark to market, then queue signals at close.
- Enforce both accounting identities on every closed trade: `Pre-Slippage - Slippage == Gross` and `Gross - Fees - Borrow == Net`, each within `1e-3`.
- Verify oracle or label features use only information available at the decision timestamp. Move any forward-looking computation into a separate evaluation-only path.

### Gate 1. Data Hygiene and Vendor Audit

- Check tradable price jumps only on executable conversions: close to close and open to previous close. A split error appears as a roughly halved or doubled price while the underlying day is ordinary.
- Reject a feed with an implausible single-bar jump above a fixed threshold such as 45 percent. Re-fetch using a genuine alternative adjustment mode if available. Record the override in a manifest. Never edit prices by hand.
- Flag single-bar high/low bad ticks where the extreme deviates sharply from open and close. Keep the original values, and substitute close for high or low in stop checks on flagged bars.
- Reject entries when the signal timestamp and fill timestamp are separated by a gap larger than one bar. Missing bars remain gaps, never synthetic observations.
- Check timestamp continuity, duplicate dates, non-positive prices, negative volume, session boundaries, and slot ranges.
- Compute every shift and rolling statistic on a single-symbol frame before concatenation. Assert that the last `1 + H` rows of each symbol are NaN for forward targets.

### Gate 2. Unfiltered Conditional Event Study Before Backtesting

- Build decision snapshots at the exact bar close when the signal is knowable, and measure returns from the next tradable open.
- Compute both raw returns and benchmark-hedged residual returns over identical timestamps with a beta estimated strictly before the decision date.
- Scan multiple horizons. The half-life of the edge and the sign of its persistence distinguish momentum from mean reversion.
- Compare magnitude against a realistic round-trip friction hurdle. An edge smaller than friction is not a strategy even if statistically significant.
- Inspect subgroups only after the unconditional result. Report all cells tested, not only the winners.

### Gate 3. Anti-Inflation Statistical Audit

- Cluster by calendar date. Use Fama-MacBeth daily means and a t statistic on the daily series, not a t statistic on individual events.
- Drop the top two days by absolute impact and recompute. A result that collapses without two days is a date-concentration artifact.
- Require mean and median to share a sign. A positive mean with a negative median is a skew story.
- Count the total number of scanned cells `M`. Report the Bonferroni critical value `t_bonf` from the normal inverse CDF at `1 - 0.05/(2M)`.
- Treat a lone marginal cell as noise. Prefer adjacent horizons with consistent sign and a clear economic mechanism.

### Gate 4. Strict IS and Locked OOS Vault Split

- Hard-code the cutoff date in the loader. Drop rows beyond the cutoff at read time, before allocation, aggregation, or display.
- Apply the same filter to forward target dates. A decision inside IS with a target beyond the cutoff is leakage.
- Never print, summarize, or inspect a single vault row during exploration. If a diagnostic needs vault data, it belongs in the final one-shot phase.
- Start each window with clean equity and zero carried positions.

### Gate 5. Controlled Ablation Ladder

- Add one hypothesis dimension at a time: baseline, sector neutrality, no-news proxy, volatility regime, then horizon change.
- Report each layer with event count, clustered t, and leave-two-out t. Discard a filter that does not add incremental residual edge.
- Explain identical adjacent layers rather than ignoring them. A filter with no overlapping triggers is inactive on that sample, not proven effective.
- Pre-register and freeze at most one to three rules before unlocking the vault. Record the exact trigger, horizon, direction, execution mode, and all caps.

### Gate 6. Realistic Execution and One-Shot OOS Verdict

- Enter at the next open from a signal at close. Size with lagged information only.
- Fill a stop gap at the actual open with adverse slippage when price gaps through the stop. Record a `stop_gapped_through` flag. Never pretend a fill at the theoretical stop price.
- Accrue short borrow on calendar days, including weekends, not trading days.
- Run a long-only version against buy and hold with Jensen alpha, beta, and information ratio. Run a beta-hedged market-neutral twin. Long-only profit that is just market beta fails.
- Sweep round-trip friction from zero to fifteen basis points and compute the breakeven friction.
- Compute the Deflated Sharpe Ratio using the scanned `M`, sample skewness, and excess kurtosis of daily returns. Compare Sharpe against its standard error `sqrt(252/T)`.
- Compare to a buy-and-hold benchmark over the identical window before declaring success.

### Gate 7. Hard Kill Criteria

- Kill a hypothesis when no cell clears the clustered and leave-out gates in IS.
- Kill a hypothesis when IS edge decays to near-zero alpha in OOS, even if raw PnL is positive.
- Kill a hypothesis when long-only PnL is market beta and the hedged twin loses money.
- Kill a hypothesis when a wide catastrophe stop still produces heavy gap-through bleed or when results depend on one sector or two days.
- Document an honest falsification instead of tuning to rescue a failed OOS. Then change universe, horizon, or information set rather than parameters.

## Battle-Tested Gotchas and Pitfalls

- Indicator timeframe mismatch: a stop distance from a fast indicator cannot be sized against risk from a slow indicator without explicit conversion.
- Entry slippage sign inversion silently converts a positive gross edge into a negative net edge on the short side only.
- A directional regime filter implemented as a plain OR blocks both directions and destroys momentum strategies.
- Residual and raw returns with conflicting signs will kill one side of a trade if the direction test is missing.
- Exits processed after entries on the same bar create an off-by-one holding period. A two-bar trade can be wiped by a scale-out meant for longer holds.
- An unmonitored entry bar hides a stop or target hit that occurred at the entry bar itself.
- Two-bar scale-out and a partial first target interfere with short holds. Prefer a clean single-exit mode for short-horizon tests.
- Relative volume is a state switch, not an additive score. High relative volume can mean momentum continuation in one horizon and exhaustion in another.
- Mean reversion with a tight stop has a structural stop-loss paradox. Prefer wide volatility-scaled stops plus a time barrier, and verify gap-through loss explicitly.
- Long and short behavior is asymmetric in equity universes. Report both sides separately even if the final rule is single-sided.
- Short-window Sharpe has a standard error of `sqrt(252/T)`. A Sharpe above five on thirty days is not evidence.
- A discovery window used for rule selection is not out of sample. Re-label it as IS.
- Vendor split adjustment can be wrong for individual symbols. Trust an executable-price jump check over the adjustment flag alone.
- A daily high or low bad tick can trigger a phantom stop. Substitute close on flagged bars.
- Adding more filters can reduce out-of-sample performance. Verify each layer incrementally instead of assuming monotonic improvement.
- Concentration in one sector or a few days can manufacture a positive mean. Report exclusion stress tests.

## Verification Checklist

- Compile all modules before running any study.
- Assert accounting identities on every closed trade in every run.
- Assert zero OOS leakage: decision date and target date both within the cutoff.
- Assert per-symbol shift isolation with last rows NaN for every forward target.
- Assert baseline features on a date do not use that date's future data.
- Assert flagged high/low bars do not produce phantom stops.
- Assert gap-through fills equal the adverse open price.
- Assert calendar-day borrow counts Friday to Monday as three days.
- Assert sector and simultaneous-position caps are respected.
- Assert the unified fast test suite passes before any cleanup or packaging step.
- Run a secret scan over tracked text files and assert zero credential matches.
