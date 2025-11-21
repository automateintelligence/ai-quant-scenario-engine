# Feature Specification: Option Strategy Optimizer (009-option-optimizer)

**Feature Branch**: `009-option-optimizer`
**Created**: 2025-11-20
**Status**: Draft
**Input**: Modular, strategy-agnostic option-trade scoring framework that evaluates option structures under assumed price movement regimes

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Single-Command Strategy Optimization (Priority: P1)

A quantitative trader wants to identify the best option strategy given their market outlook for tomorrow. They provide a ticker symbol, regime assumption (e.g., "strong-bullish"), and trade horizon, then receive a ranked list of optimal option structures with complete risk/reward analysis.

**Why this priority**: Core value proposition of 009-option-optimizer—enables autonomous strategy discovery from market assumptions without manual candidate enumeration. Essential for MVP viability.

**Independent Test**: Can be fully tested by running `qse optimize-strategy --ticker NVDA --regime strong-bullish --trade-horizon 1` and verifying that Top-10 ranked strategies are returned with complete metrics (E[PnL], POP, ROC, Greeks, VaR/CVaR) and leg-by-leg breakdowns within <30 seconds.

**Acceptance Scenarios**:

1. **Given** user provides ticker "NVDA", regime "strong-bullish", and default config, **When** optimizer runs, **Then** system returns Top-10 option strategies ranked by composite score with all metrics computed over the specified trade horizon
2. **Given** user specifies trade_horizon=3, **When** optimizer runs, **Then** all metrics (E[PnL], POP, ROC) are computed over a 3-day holding period and horizon is clearly documented in output
3. **Given** user provides --override flags for config parameters, **When** optimizer runs, **Then** specified overrides (e.g., `--override "mc.num_paths=10000"`) take precedence over config.yml defaults
4. **Given** no candidates pass hard filters, **When** optimizer completes, **Then** system returns empty result with diagnostic summary explaining which filters rejected most candidates

---

### User Story 2 - Multi-Stage Candidate Filtering (Priority: P1)

The optimizer must efficiently search thousands of potential option structures by using a multi-stage filter process: (Stage 0) expiry selection, (Stage 1) strike filtering by moneyness/liquidity, (Stage 2) structure generation with width limits, (Stage 3) cheap analytic prefiltering, (Stage 4) full Monte Carlo scoring on survivors.

**Why this priority**: Computational tractability—without staged filtering, full MC on all candidates is infeasible. This is the architectural backbone that enables 009-option-optimizer to operate within <30 second targets.

**Independent Test**: Can be tested by seeding a synthetic option chain with 1000+ candidate structures, running optimizer with verbose diagnostics enabled, and verifying that: (1) Stage 0-3 reduce candidates from ~1000 to <200, (2) Stage 4 performs full MC on survivors only, (3) runtime remains <30 seconds for Top-10 computation.

**Acceptance Scenarios**:

1. **Given** optimizer processes a stock with 15 tradeable strikes across 4 expiries, **When** Stage 0 runs, **Then** 3-5 expiries with DTE in [7, 45] days are selected
2. **Given** Stage 1 moneyness filters are applied, **When** filtering completes, **Then** only strikes with K/S0 in regime-appropriate band (e.g., [0.85, 1.15]) and meeting liquidity thresholds (volume ≥100, OI ≥100, bid-ask ≤15% of mid) are retained
3. **Given** Stage 2 generates vertical spreads with width limits 1-3 strikes, **When** generation completes, **Then** candidate count is reduced from O(S²) to O(S·W) per side per expiry
4. **Given** Stage 3 applies cheap analytic prefiltering with hard filters (capital ≤$15k, MaxLoss/capital ≤5%, E[PnL]≥$500, POP≥60%), **When** filtering completes, **Then** top K=20-50 candidates per structure type advance to Stage 4, totaling 50-200 survivors
5. **Given** Stage 4 receives survivors from Stage 3, **When** full MC scoring runs, **Then** GARCH-t/ML distribution engine with Bjerksund-Stensland pricer computes all metrics and ranks by composite intraday-spreads score

---

### User Story 3 - Regime-Driven Distribution Selection (Priority: P1)

Users specify qualitative regime labels (neutral, strong-bullish, mild-bearish, etc.) that automatically map to Monte Carlo distribution parameters (mean, σ, skew, kurtosis) without manually providing statistical inputs. The system supports config-driven regime tables, historical calibration, and explicit parameter overrides.

**Why this priority**: Bridges qualitative trading assumptions to quantitative simulation—enables non-quant users to express market views naturally while maintaining rigorous statistical foundations.

**Independent Test**: Can be tested by specifying regime="strong-bullish" in CLI, confirming that system loads corresponding distribution parameters from config.yml (e.g., mean_daily_return=0.02, daily_vol=0.03, skew=0.5), runs MC simulation with these parameters over trade_horizon days, and produces results consistent with a bullish bias.

**Acceptance Scenarios**:

1. **Given** config.yml defines 8 regime labels with distribution parameters, **When** user specifies --regime neutral, **Then** system loads mean_daily_return=0.0, daily_vol=0.01, skew=0.0, kurtosis_excess=1.0 for MC simulation
2. **Given** user specifies trade_horizon=3 with regime="strong-bullish" (mean=2%/day), **When** MC paths are generated, **Then** regime parameters compound across 3 days via multi-step simulation (not a single 6% jump)
3. **Given** user sets regime_mode="calibrated", **When** optimizer initializes, **Then** system finds historical bars matching regime features (volatility quintile, trend state, VIX level), bootstraps return sequences, and generates MC paths from empirical distribution
4. **Given** user provides explicit overrides --override "regime.mean_return=0.03" --override "regime.vol=0.04", **When** optimizer runs, **Then** explicit parameters take precedence over config table and calibration

---

### User Story 4 - Pluggable Pricing Models (Priority: P2)

The system supports swapping option pricing models (Black-Scholes, Bjerksund-Stensland, Heston, SLV/SVI) via a shared OptionPricer interface. All pricers work in both entry scoring and intraday repricing contexts, and US1 (stock-vs-option comparison) shares this interface for consistency.

**Why this priority**: Future-proofs architecture for evolving pricing needs (American exercise, stochastic vol, local vol surfaces). Bjerksund-Stensland as default provides realism without Heston's calibration overhead, but modularity is essential.

**Independent Test**: Can be tested by running optimizer with --pricing-model black_scholes vs --pricing-model bjerksund_stensland on identical inputs, verifying that both complete successfully with different option valuations, and confirming that US1's MarketSimulator can instantiate the same pricers via shared interface.

**Acceptance Scenarios**:

1. **Given** user specifies --pricing-model bjerksund_stensland, **When** Stage 4 MC scoring runs, **Then** all option legs are priced using Bjerksund-Stensland approximation (American exercise) and Greeks are computed accordingly
2. **Given** user specifies --pricing-model heston with calibrated parameters, **When** MC paths are generated, **Then** Heston model evolves both S_t (underlying price) and v_t (stochastic volatility) jointly, and option prices reflect path-dependent IV dynamics
3. **Given** Bjerksund-Stensland pricer fails to converge for extreme parameters, **When** pricing error occurs, **Then** system automatically falls back to Black-Scholes for that candidate, emits a warning, and continues without halting
4. **Given** Black-Scholes pricer also fails (extremely rare), **When** both pricers fail, **Then** system skips that candidate entirely and logs the failure in diagnostics

---

### User Story 5 - Composite Strategy Scoring (Priority: P2)

The optimizer ranks candidates using a pluggable composite scoring function (default: intraday-spreads) that weighs POP (probability of profit), ROC (return on capital), Theta reward (time decay), and penalties for tail risk, delta misalignment, gamma instability, and vega exposure. Users can adjust weights via config or develop custom scorers.

**Why this priority**: Scoring is the differentiator between naive "find any spread" and intelligent trade selection aligned with user risk preferences. Plugin architecture enables extensibility to directional-bullish, volatility-play, gamma-scalping scorers.

**Independent Test**: Can be tested by running optimizer with default intraday-spreads weights, then re-running with adjusted weights (--override "scoring.w_pop=0.50" --override "scoring.w_theta=0.20"), and verifying that Top-10 rankings change to reflect new preferences (e.g., higher theta-reward weight surfaces more short-premium trades).

**Acceptance Scenarios**:

1. **Given** optimizer computes metrics for a candidate Iron Condor (E[PnL]=$600, POP=72%, ROC=4%, Theta=+$25/day, Delta=-0.02, Gamma=-0.05, Vega=-15), **When** intraday-spreads scorer runs, **Then** score = w_pop·POP_norm + w_roc·ROC_norm + w_theta·ThetaReward - w_tail·TailPenalty - w_delta·|Delta-0|/scale - w_gamma·|Gamma|/scale - w_vega·|Vega|/scale with default weights (0.35, 0.30, 0.10, 0.15, 0.05, 0.03, 0.02)
2. **Given** user adjusts weights to prioritize ROC over POP, **When** scoring runs, **Then** high-ROC trades with lower POP rank higher than high-POP trades with lower ROC
3. **Given** user develops a custom directional-bullish scorer rewarding positive delta, **When** custom scorer is loaded via plugin, **Then** optimizer uses custom scoring logic and ranks bullish structures (call spreads with +0.3 to +0.6 delta) higher
4. **Given** scorer computes components for diagnostics, **When** output is generated, **Then** score decomposition shows individual contributions of POP, ROC, Theta, and all penalties for each Top-10 trade

---

### User Story 6 - Transaction Cost Modeling (Priority: P2)

All P&L calculations account for realistic transaction costs: entry fills pay the spread (buy at ask, sell at bid), plus $0.65 per contract commission. Hard filters and scoring operate on net P&L after costs to prevent spurious profitability from optimistic fill assumptions.

**Why this priority**: Prevents optimizer from surfacing trades that appear profitable theoretically but fail after real-world costs. Critical for production use where execution quality determines actual returns.

**Independent Test**: Can be tested by comparing optimizer output with --costs realistic (pay spread + commission) vs --costs optimistic (mid-price fills, zero commission) on identical inputs, and verifying that realistic mode rejects more candidates and reports lower E[PnL] for survivors.

**Acceptance Scenarios**:

1. **Given** candidate trade has 4 legs (Iron Condor: sell call $150, buy call $155, sell put $140, buy put $135), **When** entry fill is simulated, **Then** short legs fill at bid price, long legs fill at ask price, and total commission = 4 × $0.65 = $2.60
2. **Given** Stage 3 prefilter applies E[PnL]≥$500 hard filter, **When** filtering runs, **Then** E[PnL] is computed after subtracting entry costs and expected exit costs (pay spread again at horizon)
3. **Given** Stage 4 MC scoring computes final metrics, **When** results are reported, **Then** all P&L distributions, POP thresholds, and ROC calculations include transaction costs, and diagnostics clearly state cost assumptions
4. **Given** user specifies --commission 0 for testing, **When** optimizer runs, **Then** commission charges are zeroed but spread costs remain unless also overridden

---

### User Story 7 - Confidence Intervals & Diagnostics (Priority: P3)

Monte Carlo estimates (E[PnL], POP, VaR, CVaR) include confidence intervals to quantify estimation uncertainty. Adaptive path increase triggers when variance is high. When no candidates pass filters, diagnostics explain which constraints rejected most trades.

**Why this priority**: Quantifies epistemic uncertainty in strategy selection—prevents false confidence in point estimates. Adaptive paths balance runtime vs accuracy. Diagnostic feedback accelerates user learning when searches fail.

**Independent Test**: Can be tested by running optimizer with fixed 5,000 paths, then with adaptive mode enabled (--adaptive-paths), and verifying that: (1) fixed mode reports confidence intervals for E[PnL] and POP, (2) adaptive mode increases paths when CI width exceeds threshold (e.g., E[PnL] ± $100) up to max path cap (e.g., 20,000), (3) empty results include diagnostic summary (e.g., "87% rejected by MaxLoss filter, 10% by POP filter").

**Acceptance Scenarios**:

1. **Given** Stage 4 MC runs with 5,000 paths per candidate, **When** metrics are computed, **Then** E[PnL] and POP are reported with 95% confidence intervals (e.g., E[PnL] = $650 [CI: $580–$720], POP = 73% [CI: 70%–76%])
2. **Given** adaptive paths mode is enabled with variance threshold, **When** a candidate's E[PnL] CI width exceeds $100 after 5k paths, **Then** system doubles paths to 10k, recomputes, and repeats until CI narrows or max path cap (20k) is reached
3. **Given** zero candidates pass hard filters after Stage 3, **When** optimizer completes, **Then** output includes diagnostic summary: "92% rejected by capital filter (>$15k), 5% by MaxLoss filter (>5%), 3% by E[PnL] filter (<$500), 0% by POP filter"
4. **Given** diagnostics suggest constraint relaxation, **When** user reviews output, **Then** system provides hints (e.g., "Consider increasing max_capital to $20k—87 additional candidates would qualify") without automatically relaxing filters

---

### User Story 8 - Live Position Monitoring (Priority: P3)

After selecting a trade, users can run continuous monitoring that periodically reprices the position using updated market data, recomputes remaining-horizon metrics (POP, E[PnL], tail risk), checks early-exit conditions, and alerts when thresholds are breached.

**Why this priority**: Bridges pre-trade optimization (009-option-optimizer) to intraday risk management—enables systematic profit-taking and stop-loss logic. Essential for live trading but not required for qse/research MVP.

**Independent Test**: Can be tested by: (1) running optimizer to select a trade, (2) exporting position to JSON, (3) launching `qse monitor --position my_trade.json --interval 5min`, (4) simulating underlying price movement, (5) verifying that system reprices position, updates metrics, and triggers alert when early-exit condition is met (e.g., PnL reaches profit target or stop-loss).

**Acceptance Scenarios**:

1. **Given** user selects Iron Condor from Top-10 and exports to position.json, **When** monitoring starts with --interval 5min, **Then** system fetches live market data every 5 minutes, reprices all legs using current underlying price and updated IV, and recalculates Greeks
2. **Given** position is 1 day into a 3-day trade (H_remaining = 2 days), **When** repricing runs, **Then** system computes PnL_Hremaining, POP, VaR, CVaR by simulating MC paths from current state over remaining 2-day horizon with updated regime parameters
3. **Given** early-exit rule: close if PnL ≥ profit_target ($500) or PnL ≤ stop_loss (-$150), **When** repricing detects PnL = $520, **Then** system triggers alert "Profit target reached: $520 > $500. Recommend close position." and optionally generates broker-ready exit order JSON
4. **Given** monitoring runs in daemon mode, **When** alert conditions are met, **Then** system logs event, sends notification (email/SMS/webhook if configured), and continues monitoring unless user manually stops

---

### Edge Cases

- **No tradeable strikes after liquidity filtering**: If all strikes fail volume/OI/spread thresholds, return diagnostic: "No liquid strikes available for [ticker]. Minimum volume: 100, found max: 23. Suggest using more liquid underlying or relaxing filters."

- **Regime label not found in config**: If user specifies --regime custom_label not in config.yml, system raises clear error: "Unknown regime 'custom_label'. Available: neutral, strong-neutral, low-volatility, volatility-dir-uncertain, mild-bearish, strong-bearish, mild-bullish, strong-bullish. Use --override to provide explicit parameters."

- **Pricing model convergence failure for all candidates**: If both Bjerksund-Stensland and Black-Scholes fallback fail for all candidates (e.g., due to extreme IV or rate inputs), return diagnostic: "All candidates failed pricing. Check config for invalid parameters (IV, rate, dividend). Last error: [specific error message]."

- **Monte Carlo variance does not stabilize**: If adaptive paths reach max cap (20k) but CI width remains above threshold, system proceeds with best estimate and flags uncertainty: "Warning: E[PnL] estimate for Trade #3 remains uncertain (CI: $400–$900) even at 20k paths. High path sensitivity detected."

- **Trade horizon exceeds longest available expiry**: If user specifies --trade-horizon 30 but longest expiry is 21 DTE, return diagnostic: "Trade horizon (30 days) exceeds longest available expiry (21 DTE). No candidates possible. Suggest reducing horizon or using underlying with longer-dated options."

- **Prefilter rejects all structures of one type**: If Stage 3 hard filters eliminate all Iron Condors but accept verticals, diagnostics note: "0 Iron Condors survived filters (capital constraint eliminated 100%). Survivors: 45 verticals, 12 strangles."

- **User overrides create conflicting constraints**: If user sets --override "filters.max_capital=5000" but also --override "filters.min_epnl=1000" (likely impossible for $5k capital), system detects and warns: "Warning: Conflicting constraints may eliminate all candidates. Max capital $5k with min E[PnL] $1k implies required ROC ≥20%, which is aggressive."

- **Missing IV data for some strikes**: If Schwab API returns IV=null for certain strikes, system skips those strikes but continues: "Skipped 3 strikes due to missing IV data. Remaining: 12 usable strikes."

- **Stale market data during live monitoring**: If market data feed is delayed >15 minutes during monitoring, system warns: "Stale data detected (last update: 18 min ago). Repricing may be inaccurate. Check market hours or data connection."

## Requirements *(mandatory)*

### Functional Requirements

#### Core Optimization Engine

- **FR-001**: System MUST accept user-provided ticker symbol, regime label, trade horizon (in days), and config.yml path as inputs to optimization run
- **FR-002**: System MUST validate regime label against config.yml definitions and raise clear error if unknown label is provided without explicit parameter overrides
- **FR-003**: System MUST support CLI overrides for any config parameter using --override "path.to.param=value" syntax (e.g., --override "mc.num_paths=10000")
- **FR-004**: System MUST fetch option chain data from Schwab API including strikes, expiries, bid/ask prices, IVs, volume, and open interest per contract
- **FR-005**: System MUST handle missing or incomplete option data by skipping problematic strikes but continuing optimization for remaining tradeable strikes

#### Multi-Stage Candidate Filtering

- **FR-006**: Stage 0 MUST select 3-5 expiries with days-to-expiration (DTE) in range [7, 45] days based on config-driven expiry selection rules
- **FR-007**: Stage 1 MUST filter strikes using moneyness window (e.g., K/S0 ∈ [0.85, 1.15] for neutral regimes) and liquidity thresholds (min volume, min OI, max bid-ask spread %) from config
- **FR-008**: Stage 2 MUST generate candidate structures (verticals, Iron Condors, straddles, strangles, butterflies, condors, etc.) using width limits (1-3 strikes for verticals) and structure templates from config
- **FR-009**: Stage 3 MUST apply cheap analytic prefiltering using Black-Scholes or Bjerksund-Stensland pricer with simple Normal/Student-t distribution assumption to approximate E[PnL], POP, and Greeks
- **FR-010**: Stage 3 MUST apply hard filters (capital ≤ max_capital, MaxLoss/capital ≤ max_loss_pct, E[PnL] ≥ min_epnl, POP ≥ min_pop_breakeven, POP_target ≥ min_pop_target) and reject candidates violating constraints
- **FR-011**: Stage 3 MUST keep only top K candidates per structure type (e.g., K=20-50) based on simplified composite score, advancing 50-200 total survivors to Stage 4
- **FR-012**: Stage 4 MUST perform full Monte Carlo scoring on survivors using selected distribution engine (GARCH-t, Student-t, Laplacian, regime-conditioned bootstrap, or ML conditional model) and selected pricer (Bjerksund-Stensland default, or BS/Heston/SLV per config)

#### Regime-to-Distribution Mapping

- **FR-013**: System MUST load regime definitions from config.yml regimes section, which maps each regime label to distribution parameters (mean_daily_return, daily_vol, skew, kurtosis_excess)
- **FR-014**: System MUST support three regime modes: "table" (use config.yml lookup), "calibrated" (bootstrap from historical bars matching regime features), "explicit" (user provides --override parameters)
- **FR-015**: For multi-day trade horizons (H>1), system MUST interpret regime parameters as daily values and compound via multi-step Monte Carlo simulation (not single H-day jump)
- **FR-016**: System MUST allow swapping distribution engines via config (garch_t, student_t, laplacian, bootstrap, ml_conditional) and instantiate corresponding PathGenerator implementation

#### Pricing Model Integration

- **FR-017**: System MUST implement shared OptionPricer interface with methods: price_option(underlying, strike, maturity, rate, dividend, iv_surface/state, type) and greeks(...)
- **FR-018**: System MUST support Black-Scholes (European, baseline), Bjerksund-Stensland (American, default), Heston (stochastic vol), and SLV/SVI (local vol surface, future) pricers via pluggable architecture
- **FR-019**: For Black-Scholes/Bjerksund-Stensland, system MUST support IV evolution modes: constant (same IV as entry), sticky-delta (IV adjusts for moneyness), or user-provided custom function
- **FR-020**: For Heston pricer, system MUST couple underlying price S_t and stochastic volatility v_t jointly during Monte Carlo path generation
- **FR-021**: If Bjerksund-Stensland fails to converge, system MUST automatically fallback to Black-Scholes for that candidate, emit warning, and continue without halting
- **FR-022**: If both Bjerksund-Stensland and Black-Scholes fail, system MUST skip that candidate entirely and log failure in diagnostics
- **FR-023**: System MUST support IV surface interpolation via grid representation (strike, expiry, iv) with bilinear interpolation for strikes/expiries between grid points

#### Monte Carlo Simulation & Metrics

- **FR-024**: System MUST accept trade_horizon parameter (H in days) from CLI/config and treat it as first-class input flowing through MC engine, scoring, and early-exit logic
- **FR-025**: System MUST simulate underlying price process over H trading days using selected distribution engine and step size (e.g., H × bars_per_day if intraday bars)
- **FR-026**: For each candidate, system MUST reprice all option legs at horizon using updated time-to-expiry (original_DTE - H) and evolved IV state per selected evolution mode
- **FR-027**: System MUST compute expected P&L over horizon: E[PnL_H] = E[sum of leg values at H] - entry cost - expected exit cost
- **FR-028**: System MUST compute probability of profit: POP_0 = P(PnL_H ≥ 0) and POP_target = P(PnL_H ≥ profit_target) where profit_target defaults to $500
- **FR-029**: System MUST compute return on capital: ROC_H = E[PnL_H] / capital_used where capital_used is net debit or margin requirement per structure type
- **FR-030**: System MUST compute tail-risk metrics: MaxLoss_trade (worst MC path), VaR_5% (5th percentile of PnL distribution), CVaR_5% (expected loss in worst 5% of paths)
- **FR-031**: System MUST compute position Greeks at entry: Delta, Theta, Gamma, Vega using selected pricer's greeks() method summed across all legs
- **FR-032**: System MUST use fixed baseline path count (e.g., 5,000 paths) but support adaptive path increases when variance is high or estimates are unstable, capped at max_paths (e.g., 20,000)
- **FR-033**: System MUST report 95% confidence intervals for E[PnL] and POP based on MC sample variance (e.g., E[PnL] = $650 [CI: $580–$720])

#### Composite Scoring

- **FR-034**: System MUST implement pluggable StrategyScorer interface with method: score(candidate, metrics, config) -> float
- **FR-035**: System MUST provide default intraday-spreads scorer implementing weighted composite: Score = w_pop·POP_norm + w_roc·ROC_norm + w_theta·ThetaReward - w_tail·TailPenalty - w_delta·DeltaPenalty - w_gamma·GammaPenalty - w_vega·VegaPenalty
- **FR-036**: System MUST normalize POP to [0,1] range (e.g., POP_norm = (POP - 0.5) / 0.5 clamped to [0,1]) and ROC to [0,1] using configurable scaling
- **FR-037**: System MUST compute ThetaReward = max(0, Theta / Theta_scale) capped at 1, rewarding positive time decay for income trades
- **FR-038**: System MUST compute penalties: DeltaPenalty = |Delta - Delta_target| / Delta_scale (default Delta_target=0 for neutral), GammaPenalty = |Gamma| / Gamma_scale, VegaPenalty = |Vega| / Vega_scale, TailPenalty = MaxLoss_trade / (max_capital × max_loss_pct)
- **FR-039**: System MUST load scorer weights from config scoring section with defaults: w_pop=0.35, w_roc=0.30, w_theta=0.10, w_tail=0.15, w_delta=0.05, w_gamma=0.03, w_vega=0.02
- **FR-040**: System MUST allow users to develop custom scorer plugins by subclassing StrategyScorer and placing in scorers/ directory, which are auto-discovered at runtime
- **FR-041**: System MUST provide score decomposition in output showing individual contributions of POP, ROC, Theta, and all penalties for each ranked trade

#### Transaction Costs

- **FR-042**: System MUST model entry fills as: short legs fill at bid price, long legs fill at ask price (pay-the-spread model)
- **FR-043**: System MUST apply commission of $0.65 per contract to each leg (configurable via config.commission_per_contract)
- **FR-044**: System MUST account for expected exit costs when computing E[PnL] by assuming pay-the-spread again at horizon (bid-ask spread at H may differ from entry)
- **FR-045**: System MUST apply hard filters and scoring on net P&L after subtracting entry costs and expected exit costs
- **FR-046**: System MUST clearly document cost assumptions in output (e.g., "Entry cost: $87 (spread) + $2.60 (commission). Expected exit cost: $92 (estimated spread at H)")
- **FR-047**: System MUST allow disabling commission for testing via --commission 0 override while retaining spread costs unless separately overridden

#### Output & Results

- **FR-048**: System MUST return Top-10 ranked trades to primary output interface (CLI stdout, JSON file, or API response)
- **FR-049**: System MUST internally cache Top-100 ranked trades for analysis and comparison purposes
- **FR-050**: For each Top-10 trade, system MUST provide leg-by-leg breakdown: strike, type (call/put), side (long/short), quantity, estimated fill price, Greek contribution
- **FR-051**: For each Top-10 trade, system MUST report all metrics: E[PnL], CI for E[PnL], POP_0, POP_target, ROC, MaxLoss, VaR_5%, CVaR_5%, position Greeks (Delta, Theta, Gamma, Vega), composite score, score decomposition
- **FR-052**: System MUST generate human-readable summary for manual review including structure type, strikes/expiries, max risk, max reward, breakeven points, and key assumptions
- **FR-053**: System MUST cache broker-ready JSON order specification for each Top-10 trade including leg details, limit prices, order type, and execution instructions
- **FR-054**: When zero candidates pass filters, system MUST return empty result with diagnostic summary explaining which filters rejected most candidates (e.g., "92% rejected by capital filter, 5% by MaxLoss filter")
- **FR-055**: Diagnostics MUST optionally suggest constraint relaxation hints (e.g., "Consider increasing max_capital to $20k—87 additional candidates would qualify") without automatically relaxing filters

#### Configuration & CLI

- **FR-056**: System MUST load all configuration from single config.yml file with sections: regimes, pricing, distributions, mc, filters, scoring, structures, data_source
- **FR-057**: System MUST support shared config sections (pricing, distributions, regimes) used by both US1 and US6, plus US-specific sections (intraday-spreads scoring vs stock-vs-option comparison)
- **FR-058**: System MUST allow --override "path.to.param=value" CLI flags to override any config parameter without editing files
- **FR-059**: Primary invocation MUST be: qse optimize-strategy --ticker TICKER --regime REGIME --trade-horizon H [--config path] [--override key=val]...
- **FR-060**: System MUST validate required parameters (ticker, regime) and provide clear error messages for missing or invalid inputs
- **FR-061**: System MUST target <30 seconds runtime for Top-10 computation at market open using after-hours data (Stage 0-4 with 5k paths baseline)
- **FR-062**: System MUST permit longer runtimes for batch/overnight modes with higher path counts (--override "mc.num_paths=20000") for increased accuracy

#### Live Position Monitoring

- **FR-063**: System MUST support exporting selected trade to JSON position file with structure: legs[], entry_prices[], entry_time, trade_horizon, regime, config_snapshot
- **FR-064**: System MUST support monitoring command: qse monitor --position POSITION_JSON --interval INTERVAL [--alert-config ALERTS_JSON]
- **FR-065**: During monitoring, system MUST fetch live market data (underlying price, option bids/asks, IVs) at specified interval (e.g., every 5 minutes)
- **FR-066**: System MUST reprice position using updated market data, recompute remaining horizon (H_remaining = max(1, H_total - days_elapsed)), and run MC paths from current state over H_remaining to estimate PnL_Hremaining, POP, VaR, CVaR
- **FR-067**: System MUST check early-exit conditions from alert config (e.g., close if PnL ≥ profit_target or PnL ≤ stop_loss) and trigger alerts when thresholds are breached
- **FR-068**: System MUST log monitoring events (repricing results, alert triggers) and optionally send notifications (email/SMS/webhook) when configured
- **FR-069**: System MUST continue monitoring until user manually stops or position reaches horizon (H_remaining = 0)
- **FR-070**: When early-exit condition is met, system MUST optionally generate broker-ready exit order JSON for automated or manual execution

#### Integration with US1

- **FR-071**: System MUST share OptionPricer interface with US1's MarketSimulator to avoid code duplication and guarantee consistent pricing
- **FR-072**: System MUST share distribution models (GARCH-t, Student-t, Laplacian, ML conditional) with US1 for regime-based simulations
- **FR-073**: System MUST use single config.yml for both US1 and US6 with shared sections (pricing, distributions) and US-specific scoring/comparison sections

#### Testing & Validation

- **FR-074**: System MUST include at least one synthetic reference test case (known option chain + regime -> expected Top-10 structure) to prevent regressions once stable candidate generator is ready
- **FR-075**: System MUST log detailed diagnostics for each optimization run including: stage timings, candidate counts per stage, filter rejection breakdowns, MC convergence stats, scorer component contributions

### Key Entities *(include if feature involves data)*

- **Regime**: Qualitative market outlook label (neutral, strong-bullish, etc.) mapped to statistical distribution parameters (mean_daily_return, daily_vol, skew, kurtosis_excess) for Monte Carlo simulation
- **OptionPricer**: Pluggable interface for option valuation models (Black-Scholes, Bjerksund-Stensland, Heston, SLV/SVI) providing price_option() and greeks() methods. Shared with 001-mvp-pipeline.
- **ReturnDistribution**: Pluggable interface for underlying return distribution models (GARCH-t, Student-t, Laplacian, bootstrap, ML conditional) providing generate_paths() method for regime-driven simulation. Shared with 001-mvp-pipeline (which uses fit/sample workflow).
- **StrategyScorer**: Pluggable interface for composite scoring functions (intraday-spreads, directional-bullish, volatility-play, gamma-scalping) providing score() method
- **CandidateStructure**: Representation of a multi-leg option position including legs[] (each with strike, type, side, quantity, fill_price), structure_type, expiry, capital_used, max_loss
- **Metrics**: Computed metrics bundle for a candidate including E[PnL], CI_epnl, POP_0, POP_target, ROC, MaxLoss, VaR_5%, CVaR_5%, position_greeks (Delta, Theta, Gamma, Vega)
- **IVSurface**: Grid representation of implied volatility surface holding (log_moneyness, tau, iv) points with bilinear interpolation via iv_at(S, K, T) method
- **Position**: Open trade representation for monitoring including legs[], entry_prices[], entry_time, trade_horizon, H_remaining, current_PnL, regime, config_snapshot

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can discover optimal option strategies from regime assumptions without manually enumerating candidates—optimizer returns Top-10 ranked trades with complete risk/reward analysis in <30 seconds for typical stock option chains (15 strikes × 4 expiries)
- **SC-002**: Multi-stage filtering reduces candidate search space from ~1000 raw structures to <200 survivors for full MC scoring, enabling computational tractability within runtime targets
- **SC-003**: Regime-driven distribution selection allows non-quant users to express market views naturally—specifying --regime strong-bullish correctly biases MC paths toward bullish outcomes with configurable statistical rigor
- **SC-004**: Pluggable pricing architecture enables swapping between Black-Scholes (baseline), Bjerksund-Stensland (default American), Heston (stochastic vol), and future SLV/SVI models without changing simulation/scoring code
- **SC-005**: Composite scoring functions accurately reflect user risk preferences—adjusting scorer weights (e.g., increasing w_theta from 0.10 to 0.20) surfaces more short-premium high-theta trades in Top-10
- **SC-006**: Transaction cost modeling prevents spuriously profitable trades—all ranked strategies remain net profitable after paying spread + $0.65/contract commission, with realistic mode rejecting ≥20% more candidates than optimistic mid-price fills
- **SC-007**: Monte Carlo confidence intervals quantify epistemic uncertainty—E[PnL] and POP are reported with 95% CIs, and adaptive paths mode reduces CI width to acceptable thresholds (e.g., E[PnL] ± $100) for Top-10 trades
- **SC-008**: Live position monitoring enables systematic intraday risk management—users can detect profit target or stop-loss conditions within 5 minutes of threshold breach and receive actionable alerts
- **SC-009**: Shared OptionPricer interface with US1 ensures pricing consistency—identical option legs priced in 009-option-optimizer optimizer and US1 stock-vs-option comparison produce identical valuations within numerical tolerance
- **SC-010**: Empty result diagnostics accelerate user learning—when no candidates pass filters, diagnostic summary explains rejection breakdowns (e.g., "87% rejected by capital filter") and suggests concrete adjustments without automatically relaxing constraints
- **SC-011**: Plugin architecture enables extensibility—users can develop custom StrategyScorer subclasses (e.g., directional-bullish rewarding +delta, volatility-play rewarding +vega) and system auto-discovers and applies them without core code changes
- **SC-012**: Performance scales to batch processing—optimizer can process 10 underlyings sequentially within 5 minutes (<30 sec per ticker) using default 5k paths, enabling pre-market screening workflows

## Assumptions

1. **Data Source**: Schwab API provides option chain data with sufficient coverage (strikes, expiries, IVs, volume, OI) for most liquid equities. For MVP, missing IV data triggers skip-and-continue behavior; future enhancement adds IV surface interpolation.

2. **Market Hours**: Optimizer is primarily used during after-hours or pre-market for next-day strategy selection (<30 sec target). Live monitoring during market hours tolerates 5-minute repricing intervals and accepts 15-minute data staleness warnings.

3. **Trade Horizon Scope**: Initial focus on short-term horizons (H=1 to H=5 days). Very short DTE (0-2 days) gamma plays are treated as a separate mode with conservative risk budgets. Longer horizons (H>7 days) are supported but not optimized for.

4. **Structure Types**: MVP implements verticals, Iron Condors, straddles, strangles. Butterflies, calendars, diagonals, and exotic structures are future enhancements once core architecture is validated.

5. **Pricing Model Defaults**: Bjerksund-Stensland is default pricer (fast American approximation). Heston requires calibrated parameters (kappa, theta, sigma, rho, v0) from IV surface fitting, which is advanced usage. SLV/SVI are future enhancements pending rich surface data.

6. **Distribution Engine Baselines**: GARCH-t is baseline for volatility clustering + heavy tails. Student-t offers fast alternative. Regime-conditioned bootstrap and ML conditional models require historical bar database with regime tagging, which is separate infrastructure work.

7. **Commission Structure**: $0.65/contract reflects Interactive Brokers pricing. Users with different brokers override via config. Spread costs (pay-the-spread model) are broker-agnostic.

8. **IV Evolution**: For BS/Bjerksund, sticky-delta is recommended default (industry standard). Constant IV is simple fallback. Heston drives IV dynamics endogenously. Custom IV functions are expert mode.

9. **Risk-Free Rate & Dividends**: Config provides single risk-free rate and dividend yield per ticker. Future enhancement: term structure of rates, discrete dividend schedules.

10. **Early Exercise**: Bjerksund-Stensland approximates optimal early exercise for American options. Exact exercise boundary (finite difference PDE) is future enhancement for extreme accuracy needs.

11. **Greeks Stability**: Position Greeks are computed at entry using selected pricer. Intraday Greek evolution during monitoring uses repricing at current spot/IV. Gamma/vega hedging is future enhancement.

12. **Configuration Complexity**: Single config.yml is sufficient for MVP scope. If config grows unwieldy (>500 lines), future refactoring to modular configs (pricing.yml, scoring.yml) is straightforward.

13. **User Expertise**: Optimizer is designed for quantitatively-aware traders familiar with options mechanics (strikes, expiries, Greeks, POP, ROC). Not a beginner education tool.

14. **Regulatory Compliance**: Optimizer provides decision support only, not automated order execution. Users are responsible for broker account permissions, margin requirements, and regulatory compliance. System generates broker-ready JSON orders for convenience but does not submit trades.

15. **Backtesting Alignment**: 009-option-optimizer optimization results are intended for forward-looking trade selection. Backtesting validation of selected strategies against historical data is separate capability (potential future US10 or US11).
