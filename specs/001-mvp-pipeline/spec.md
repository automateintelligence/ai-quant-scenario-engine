# Feature Specification: Backtesting & Strategy Spec Authoring

**Hierarchy**: Parent constitution at `.specify/memory/CONSTITUTION.md`; this spec governs downstream artifacts (`plan.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`).

**Feature Branch**: `001-mvp-pipeline`
**Spec Version**: 1.0.0
**Created**: 2025-11-15
**Last Updated**: 2025-11-16
**Status**: Draft
**Input**: User description: "Review 'planning/project_goals.md' and 'planning/project_overview.md' and then create specifications."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run stock vs option Monte Carlo comparison (Priority: P1)
Author runs a CLI to generate Monte Carlo price paths for a chosen symbol, applies paired stock and option strategies, and receives equity curves plus summary metrics to compare risk/reward.

**Why this priority**: Core goal is to evaluate stock vs option surfaces quickly on CPU-only VPS.
**Independent Test**: Execute CLI with sample symbol (e.g., AAPL) and verify outputs include both stock and option metrics plus saved report artifacts.

**Acceptance Scenarios**:
1. **Given** historical OHLCV is available for the symbol, **When** the user runs the CLI with defaults, **Then** the system produces N simulated paths, runs stock and option strategies, and prints/save metrics for both.  
2. **Given** the user provides a different distribution choice (e.g., Laplacian, student-t, etc.), **When** the CLI runs, **Then** the simulation uses that distribution and completes without errors, updating outputs accordingly.

---

### User Story 2 - Parameter grid exploration (Priority: P2)
Author defines a parameter grid (e.g., thresholds, DTE, strike offset) and runs batch evaluations in parallel, receiving aggregate metrics (mean, CVaR, drawdown) per configuration.

**Why this priority**: Enables strategy optimization and surface mapping beyond a single run.
**Independent Test**: Supply a small grid (≥3 configs) and confirm per-config metrics are produced and ranked by an objective function.

**Acceptance Scenarios**:
1. **Given** a grid of strategy parameters, **When** the batch job executes, **Then** results include metrics per config and an objective score suitable for ranking.  
2. **Given** N paths × M configs, **When** parallel execution runs, **Then** the job completes within configured resource limits and without race conditions or corrupted outputs.

---

### User Story 3 - Feature-enriched signals (Priority: P3)
Author augments simulations with technical indicators and optional macro series to drive signals without changing core engine code.

**Why this priority**: Supports experimentation with richer features while keeping engine modular.
**Independent Test**: Add an indicator definition (e.g., SMA/RSI) and confirm the strategy receives the feature columns and uses them in signal generation for a run.

**Acceptance Scenarios**:
1. **Given** a new indicator is declared in config, **When** a run executes, **Then** the indicator column is present in features and consumed by the strategy without code changes.  
2. **Given** a macro series is missing or empty, **When** a run executes, **Then** the system warns and continues using available features without crashing.

---

### User Story 4 - Stock screening via candidate selector (Priority: P1)
Author runs a CLI to scan a universe of symbols, apply a candidate-selection rule (gap/volume/volatility, etc.), and receive a ranked list of stocks that are good candidates for a specified strategy before the next trading day.

**Why this priority**: This is the front door of the workflow: you will only trade a few symbols, but need to quickly evaluate hundreds under transient, uncommon conditions.
**Independent Test**: Configure a universe of ≥100 symbols and a simple candidate rule (e.g., top 20 daily gainers/losers with volume filters), run the CLI, and confirm it returns a ranked candidate list with required metadata (symbol, trigger time, key features).

**Acceptance Scenarios**:
1. **Given** a universe definition and candidate criteria, **When** the user runs the screening CLI, **Then** the system produces a list of candidate episodes (symbol, timestamp, basic features) for the next-day strategy evaluation.
2. **Given** some symbols have missing or partial data in the requested window, **When** the screening job runs, **Then** those symbols are skipped or downgraded with clear warnings, and the rest of the screen completes successfully.
3. **Given** the user switches candidate criteria (e.g., from “top 20 gainers” to “gap-down + volume spike”), **When** the CLI is re-run, **Then** the system produces a different candidate set without requiring code changes to the backtest or strategy modules.

---

### User Story 5 - Strategy-symbol screening with optional conditional episodes (Priority: P1)
Author runs a CLI to evaluate a strategy across a universe of symbols, receiving a ranked list of which symbols work best for that strategy. Optionally, author can filter evaluation to only historical episodes matching specific market conditions (e.g., gaps, volume spikes) defined by a candidate selector from US4.

**Why this priority**: Directly answers "Which stocks should I trade with this strategy?" and optionally "Which stocks work best under specific transient conditions?" This bridges US4 (defining conditions) with strategy evaluation across symbols.

**Independent Test**:
1. Run screen with strategy on ≥100 symbols (unconditional), verify ranked output by performance metrics (sharpe/mean_pnl/etc.)
2. Add conditional selector filter, verify episodes are filtered and rankings change accordingly
3. Run without strategy (US4 mode), verify candidate selection works as before

**Three Operational Modes**:

**Mode A - Candidate Selection Only (US4 functionality preserved)**:
```bash
screen --universe sp500.csv --gap-min 0.03 --volume-z-min 1.5 --top 20
```
- No `--strategy` flag present
- Filters universe by conditions for recent/current market state
- Output: Symbols matching conditions right now (forward-looking)
- Use case: "What should I look at tomorrow morning?"

**Mode B - Unconditional Strategy Screening**:
```bash
screen --universe sp500.csv --strategy stock_sma_trend --rank-by sharpe --top 20
```
- `--strategy` flag present, no conditional filters
- Backtests strategy on ALL historical data per symbol
- Output: Symbols where strategy performed best historically
- Use case: "Which stocks work best for this strategy overall?"

**Mode C - Conditional Strategy Screening**:
```bash
screen --universe sp500.csv --strategy stock_sma_trend --rank-by sharpe \
  --conditional-file selectors/gap_down_volume_spike.yaml --top 20
```
- `--strategy` flag + `--conditional-file` or inline conditional params
- Backtests strategy ONLY on historical episodes where selector conditions met
- Output: Symbols where strategy performed best during specific conditions
- Use case: "Which stocks work best for this strategy when gaps occur?"

**Acceptance Scenarios**:
1. **Given** a universe of symbols and a strategy (Mode B), **When** screen runs unconditionally, **Then** system backtests strategy on all historical data per symbol and returns ranked list by specified metric (sharpe, mean_pnl, sortino, etc.) with run metadata indicating unconditional evaluation.

2. **Given** a conditional selector file (YAML format) and Mode C invocation, **When** screen runs with --conditional-file flag, **Then** system evaluates strategy ONLY on episodes where selector conditions are met and ranks symbols by conditional performance, with run metadata clearly indicating episode count per symbol and filtering criteria applied.

3. **Given** no --strategy flag (Mode A), **When** screen runs with only condition parameters, **Then** system performs candidate selection (US4 behavior) on recent bars and returns symbols matching current conditions without strategy evaluation.

4. **Given** some symbols lack sufficient conditional episodes (<10 episodes), **When** Mode C screen runs, **Then** those symbols are flagged with warnings in output but remain in ranked results with low-confidence indicators and episode counts clearly stated.

5. **Given** a conditional selector definition changes (e.g., gap_min threshold increased), **When** the conditional screen is re-run, **Then** the episode set changes per symbol and this is reflected in run metadata and metrics without modifying strategy or simulator code.

6. **Given** outputs from conditional vs unconditional runs on the same universe and strategy, **When** reviewing results side-by-side, **Then** artifacts clearly distinguish between evaluation modes (unconditional: all bars, conditional: filtered episodes) with different result file naming conventions.

---

### User Story 6 - Conditional Monte Carlo from candidate states (Priority: P2)
Author generates Monte Carlo paths that are specifically conditioned on a current candidate state (e.g., big gap with volume spike), runs stock vs option strategies over those paths, and reviews the conditional risk/reward distribution.

**Why this priority**: Bridges real historical episodes and synthetic scenarios, allowing "what-if" analysis given today's specific transient conditions.
**Independent Test**: For a symbol and a chosen candidate state, run a conditional Monte Carlo CLI (e.g., 1,000 paths × 60 steps) and verify that the run uses the state-conditioned model and produces summary distributions for stock and option P&L.

**Acceptance Scenarios**:
1. **Given** a current candidate state vector (features at t₀) and a chosen conditioning method (episode resampling or state-conditioned distribution), **When** the user runs the conditional Monte Carlo command, **Then** the system generates simulated paths that reflect that state and computes stock and option metrics over those paths.
2. **Given** the user supplies a fixed random seed, **When** the conditional Monte Carlo run is repeated with the same parameters, **Then** the path-level outcomes and summary metrics are identical (within numeric tolerances).
3. **Given** the selected conditioning method cannot be applied (e.g., too few similar historical episodes), **When** the user runs the command, **Then** the system emits a clear warning and either falls back to a configured default model or aborts gracefully with an error.

---

### User Story 6a - Distribution Model Validation and Selection (Priority: P2)
Author runs an automated distribution audit for a given ticker symbol to fit multiple return distribution models (Laplace, Student-t, GARCH-t), validates each model through comprehensive statistical tests, and receives a recommended "best fit" model based on empirical evidence of tail behavior, volatility clustering, and simulation realism—ensuring Monte Carlo paths in US1 and US6 reflect realistic market dynamics including black swans.

**Why this priority**: No single distribution model is universally correct for all assets or market regimes. Empirical validation ensures Monte Carlo simulations are built on distributions that actually reproduce observed characteristics (fat tails, volatility clustering, extreme event frequency) rather than assuming a default model. This is foundational for US1 (basic Monte Carlo) and US6 (conditional Monte Carlo) to generate realistic stress test scenarios.

**Independent Test**: For a liquid ticker (e.g., SPY) with 3 years of daily historical data, run the distribution audit CLI. Verify the system fits Laplace, Student-t, and GARCH(1,1)-t models, computes AIC/BIC, performs VaR backtesting with Kupiec/Christoffersen tests, validates simulation realism metrics, and outputs a ranked model comparison report with a clear recommendation and validation evidence.

**Acceptance Scenarios**:

**Preprocessing and Model Fitting:**

1. **Given** a ticker symbol and a lookback window (e.g., 3–5 years), **When** the user runs the distribution audit command, **Then** the system:
   - Loads historical OHLCV data for the specified period
   - Computes log returns: r_t = log(P_t / P_{t-1}) using close prices
   - Validates stationarity requirements (minimum sample size, no extreme outliers)
   - Fits three candidate models via Maximum Likelihood Estimation (MLE):
     * **Laplace** (double-exponential): Estimates location μ and scale b
     * **Student-t**: Estimates location μ, scale σ, and degrees of freedom ν
     * **GARCH(1,1)-t**: Estimates volatility parameters (ω, α, β) and residual degrees of freedom ν
   - Extracts standardized residuals for GARCH-t: z_t = r_t / σ̂_t
   - Records convergence status, number of iterations, and any fit warnings for each model

**In-Sample Goodness-of-Fit:**

2. **Given** fitted model parameters, **When** the system computes in-sample goodness-of-fit metrics, **Then** for each model it calculates and reports:
   - Log-likelihood (ℓ)
   - Akaike Information Criterion: AIC = 2k - 2ℓ (where k = number of parameters)
   - Bayesian Information Criterion: BIC = k×log(n) - 2ℓ (where n = sample size)
   - Model ranking by AIC and BIC (lower is better)
   - Interpretation guidance: "Student-t AIC is 45 points lower than Laplace → strong evidence for Student-t"

**Tail Fit Diagnostics:**

3. **Given** fitted models and empirical return distributions, **When** the system performs tail fit diagnostics, **Then** it:
   - Generates QQ plots comparing empirical quantiles vs model-predicted quantiles
   - Focuses analysis on tail regions (beyond ±2σ)
   - Computes empirical vs fitted tail quantiles at: 95%, 99%, 99.5% (for loss tail)
   - Calculates tail error metric for each model:
     * tail_error_99 = |VaR_model(0.99) - VaR_emp(0.99)| / |VaR_emp(0.99)|
   - Computes excess kurtosis for empirical vs each fitted distribution
   - Flags models where fitted kurtosis << empirical kurtosis (underestimating tail fatness)
   - Validates heavy-tail requirement: excess kurtosis ≥ 1.0
   - Outputs tail diagnostic summary with visual QQ plots and quantitative error metrics

**VaR Backtesting (Out-of-Sample):**

4. **Given** a 70/30 train/test split of historical data, **When** the system performs VaR backtesting, **Then** for each model it:
   - Refits model parameters on training data only (70%)
   - Generates one-step-ahead predictive VaR for each test day at 95% and 99% confidence
   - Counts actual VaR breaches (days where realized return < predicted VaR)
   - Computes expected breach frequency (5% for VaR₉₅, 1% for VaR₉₉)
   - Performs **Kupiec unconditional coverage test**:
     * H₀: Breach frequency matches expected frequency
     * Reports test statistic, p-value, and pass/fail at α=0.05
   - Performs **Christoffersen independence test**:
     * H₀: VaR breaches are independent (not clustered)
     * Reports test statistic, p-value, and pass/fail at α=0.05
   - Flags models that systematically underpredict breach frequency or show clustered breaches
   - Outputs backtest results table with breach counts, expected counts, test statistics, and diagnostic interpretation

**Simulation Realism Validation:**

5. **Given** each fitted model, **When** the system validates simulation realism, **Then** it:
   - Generates 10,000 Monte Carlo paths × 252 steps (one trading year) per model
   - Computes the following statistics for each simulated path:
     * Annualized volatility: σ_annual = σ_daily × √252
     * Autocorrelation of squared returns at lag 1: corr(r²_t, r²_{t-1}) (measures volatility clustering)
     * Maximum drawdown: max percentage decline from peak
     * Frequency of extreme single-day moves: count of |r_t| > 3% and |r_t| > 5%
   - Computes the same statistics on historical rolling windows (e.g., 252-day windows)
   - Compares distributions of simulated vs historical metrics:
     * Volatility: Does simulated σ range match historical σ range?
     * Clustering: Does simulated autocorr(r²) match historical autocorr(r²)?
     * Drawdowns: Does simulated max DD distribution align with historical DDs?
     * Extreme events: Does simulated frequency of >3%/5% moves match historical frequency?
   - Reports systematic biases, e.g.:
     * "Laplace simulations show 0.2% of days with |r| > 5%, but historical data shows 1.8% → tail too thin"
     * "GARCH-t clustering metric matches historical (0.15 vs 0.14) ✓"
   - Outputs simulation realism report with distributional comparisons and visual histograms

**Model Selection and Scoring:**

6. **Given** all validation metrics (AIC/BIC, tail errors, VaR backtest results, simulation realism), **When** the system computes model selection scores, **Then** it:
   - Normalizes AIC values: AIC_norm = (AIC - AIC_min) / (AIC_max - AIC_min)
   - Computes composite score for each model:
     * score = w₁×(-AIC_norm) + w₂×(-tail_error_99) + w₃×(-VaR_backtest_penalty) + w₄×(-vol_cluster_error)
     * Default weights: w₁=0.2, w₂=0.4, w₃=0.3, w₄=0.1 (prioritize tail accuracy and VaR calibration)
   - Applies mandatory constraints:
     * Excess kurtosis ≥ 1.0 (heavy-tailed requirement)
     * VaR backtest must not catastrophically fail both Kupiec and Christoffersen tests (p < 0.01 on both)
   - Selects the highest-scoring model that meets all constraints
   - Documents selection rationale with evidence:
     * "Selected Student-t (score: 0.87) over Laplace (0.62) and GARCH-t (0.79)"
     * "Student-t: Best tail fit (1.2% error), passed VaR backtests, matched clustering"
   - Warns if no model meets minimum standards:
     * "WARNING: All models show excess kurtosis < 1.0. Consider longer lookback or data quality review."
   - Stores selected model parameters and validation metadata in structured format (JSON)

**Caching and Reproducibility:**

7. **Given** a completed distribution audit for a symbol/lookback combination, **When** the results are stored, **Then** the system:
   - Caches fitted parameters, validation metrics, and model selection decision
   - Includes cache key: (symbol, lookback_days, end_date, data_source)
   - Timestamps the cache entry
   - On subsequent requests with matching cache key, returns cached results if < 30 days old
   - Provides CLI flag `--force-refit` to bypass cache and recompute

8. **Given** a fixed random seed and identical audit parameters, **When** the distribution audit is run multiple times, **Then**:
   - All MLE-fitted parameters are identical (within 1e-6 tolerance)
   - All test statistics (AIC, BIC, tail errors, Kupiec/Christoffersen p-values) are identical
   - Model selection decision is identical
   - Generated Monte Carlo paths (for simulation realism checks) are identical

**Integration with Monte Carlo Workflows:**

9. **Given** a validated distribution model from US6a audit, **When** the user runs US1 (basic Monte Carlo) or US6 (conditional Monte Carlo), **Then** the system:
   - Automatically loads the cached validated model for the specified symbol
   - Uses the empirically-selected model (Student-t or GARCH-t) instead of a hardcoded default
   - Logs model metadata in run provenance: model type, fit date, validation scores
   - Warns if cached validation is > 30 days old and suggests re-running audit
   - Allows user override to specify a different model via CLI flag if desired

10. **Given** no prior distribution audit exists for a symbol, **When** the user runs US1 or US6 without specifying a model, **Then** the system:
    - Emits a warning: "No validated model found for AAPL. Using default Laplace model."
    - Suggests running distribution audit: "Run `python -m quant_scenario_engine.cli audit-distributions --symbol AAPL` to validate models."
    - Proceeds with Laplace as safe default but marks run metadata with `model_validated: false`

**Error Handling and Edge Cases:**

11. **Given** insufficient historical data for a model (e.g., GARCH-t requires ≥252 bars), **When** the audit attempts to fit that model, **Then** the system:
    - Skips the model with a clear warning: "GARCH-t requires ≥252 bars, only 180 available. Skipping GARCH-t."
    - Continues fitting other models (Laplace, Student-t)
    - Documents the skip in the audit report

12. **Given** MLE fitting fails to converge for a model (e.g., numerical instability), **When** the audit encounters convergence failure, **Then** the system:
    - Logs convergence failure with diagnostic details (iterations, tolerance, error message)
    - Marks that model as "FAILED" in the audit report
    - Continues with other models
    - Fails the entire audit only if all three models fail to converge

---

### User Story 7 - Config-driven component swapping (Priority: P2)
Author changes configuration (YAML/env/CLI flags) to swap data sources, distribution models, and option pricers without editing code, and sees the new components reflected in subsequent runs.

**Why this priority**: Preserves modularity and lets you evolve the system (Schwab vs yfinance, Normal vs Student-T vs GARCH-T, Black–Scholes vs alternative pricer) without refactoring core engine logic
**Independent Test**: Run three successive CLI runs that vary only in configuration (e.g., yfinance+Normal+BS vs Schwab+Student-T+BS vs Schwab+Student-T+AltPricer) and confirm the wiring changes take effect with different outputs and no code modifications.

**Acceptance Scenarios**:
1. **Given** a configuration file specifying `data_source = "yfinance"` and `distribution = "student_t"`, **When** the user runs the CLI, **Then** the system uses yfinance and the Student-T model for that run.
2. **Given** the user changes configuration to `data_source = "schwab"` and `option_pricer = "heston"` (or another advanced pricer), **When** the CLI is re-run, **Then** the system uses the new data source and pricer, and the option-equity curves change while stock-equity logic remains unchanged.
3. **Given** an invalid or unsupported configuration value, **When** a run is started, **Then** the system fails fast with a clear configuration error and does not start the simulation.

---

### User Story 8 - Run provenance and replay (Priority: P3)
Author can inspect a run directory and reconstruct exactly how results were produced (data source, distribution, seeds, parameters), and can re-run a previous configuration to reproduce metrics.

**Why this priority**: Essential for trust, debugging, and research discipline, especially when you start iterating quickly and comparing strategies across days.
**Independent Test**: After running a stock vs option comparison, inspect the run metadata file, re-run the CLI with a `--replay <run_id>` flag, and confirm that the regenerated metrics match the original within tolerance.

**Acceptance Scenarios**:
1. **Given** any successful run, **When** the user inspects the run directory, **Then** it contains a metadata file (e.g., `run_meta.json`) that captures symbol(s), timeframe, data source, distribution model, seeds, strategy parameters, and component versions.
2. **Given** the system supports a `--replay` or equivalent option, **When** the user invokes a run using a previous run ID, **Then** the system either:

   * Regenerates paths and metrics that match the original run (if using seeds), or
   * Loads persisted Monte Carlo paths and replays the backtest to produce identical metrics.
3. **Given** some historical inputs (e.g., Parquet files) have changed since the original run, **When** a replay is requested, **Then** the system warns that underlying data has changed and marks the replay as non-strictly-comparable in the metadata.

---

### Edge Cases

- Missing or insufficient historical data for the symbol or date range.
- Distribution fit fails or returns implausible parameters (e.g., near-zero or explosive volatility).
- Option maturity shorter than simulation horizon; maturity reached mid-path.
- Zero/negative prices in input data or generated paths; stepwise NaNs in OHLCV or indicators.
- Path generation with extremely fat tails causing overflow/inf in price calculations.
- Parallel grid execution exceeding CPU/memory limits on the VPS.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load OHLCV for a symbol and date range from a pluggable data source (yfinance baseline; Schwab later), automatically downloading on-demand if not present in local cache, and validate presence of required columns. Local cache SHALL use Parquet format per DM-004 and be stored under `data/historical/{interval}/{symbol}.parquet`. On cache miss, system SHALL fetch from configured data source, validate, write to cache, then proceed with run.
- **FR-002**: System MUST fit Laplacian (double-exponential) returns as the default heavy-tailed model and generate N Monte Carlo paths of length T from that fit; additional models (e.g., Student-T, GARCH-T) MAY be provided as alternatives. Fitting MUST record estimator type (MLE/GMM), convergence status, log-likelihood, AIC/BIC, and implausible-parameter checks per model; non-stationary series MUST be rejected or transformed prior to fit (stationarity/AR preflight required). System SHOULD support comprehensive distribution validation and selection per **US6a** (Distribution Model Validation and Selection), including VaR backtesting, tail diagnostics, simulation realism checks, and empirical model scoring to determine the best-fit distribution for each symbol/lookback combination.
- **FR-003**: System MUST run a stock strategy over generated paths, producing per-path equity curves and summary metrics (mean/median P&L, drawdown, VaR/CVaR, Sharpe/Sortino).
- **FR-004**: System MUST run an option strategy over the same paths using an option pricer (Black-Scholes with configurable IV) and produce analogous metrics; option strategies MUST support strategy-driven early exercise hooks (e.g., `check_early_exercise`) to exit before expiry, using European pricing by default unless an American-capable pricer is configured.
```python
# engine/simulator.py

import numpy as np
import pandas as pd

class MarketSimulator:
    """Executes stock and option P&L given price paths and position arrays."""

    def simulate_stock(self, close: pd.Series, position: np.ndarray) -> pd.Series:
        px = close.values
        pnl = np.zeros_like(px)
        pnl[1:] = position[:-1] * (px[1:] - px[:-1])
        return pd.Series(np.cumsum(pnl), index=close.index)

    def simulate_option(
        self,
        close: pd.Series,
        position: np.ndarray,
        pricer,
        spec,
    ) -> pd.Series:
        px = close.values.astype(float)
        n = len(px)
        ttm = np.maximum(spec.maturity_days - np.arange(n), 1) / 252.0
        opt_px = pricer.price(px, spec, ttm)

        pnl = np.zeros_like(opt_px)
        pnl[1:] = position[:-1] * (opt_px[1:] - opt_px[:-1])
        return pd.Series(np.cumsum(pnl), index=close.index)
```
- **FR-005**: System MUST expose a CLI to execute the stock-vs-option comparison with configurable symbol, date range, distribution, paths, steps, and strategy parameters.
- **FR-006**: System MUST support feature/indicator injection (e.g., SMA/RSI via pandas-ta) without modifying strategy engine code and SHALL ship a baseline “hello world” strategy (dual-SMA crossover) for validation and demos.
- **FR-007**: System MUST support parameter grid/batch execution with parallelization and return an objective score per configuration.
- **FR-008**: System MUST emit artifacts per run (e.g., metrics JSON/CSV and optional plots/HTML reports) to a run output directory.
- **FR-009**: System MUST allow selection of data source and option pricer via configuration rather than code changes.
- **FR-010**: System MUST handle missing data gracefully with clear warnings and deterministic fallbacks (e.g., drop NaNs, fail-fast if below minimum length).
- **FR-011**: System MUST provide stubs for Schwab API data and advanced pricers while defaulting to yfinance + BS so the MVP runs locally.
- **FR-012**: System MUST allow deterministic seeding of random number generation for reproducible simulations.
- **FR-013**: System SHOULD cap Monte Carlo sampling based on estimated footprint `n_paths * n_steps * 8 bytes` plus **10% overhead** and refuse in-memory runs above **25% of available RAM**, automatically switching to memmap/npz and emitting a warning; MUST abort when estimate exceeds **50% of available RAM**.  
- **FR-014**: System SHOULD support macro/alternative series alignment (e.g., interpolation to bar frequency) with explicit tolerances: maximum forward/backfill gap of **3× bar interval** and warning when alignment error exceeds tolerance. Alignment method SHALL be configurable per feature (default `forward_fill`; `linear` or `spline` for smooth series) and documented in config.
- **FR-015**: System SHOULD document performance targets (paths × steps per second) for the CPU-only VPS and specify measurement methodology (hardware profile, wall-clock vs CPU, sample config).
- **FR-016**: System MUST support Black–Scholes pricing with per-strike implied volatility as the default option pricer, and MUST allow swapping in more advanced pricers (e.g., Heston/QuantLib) via configuration without changes to strategy or backtest code. Option pricing MUST handle maturity=horizon, ATM precision, invalid IV, and negative/zero prices with structured errors; strategy-driven early exercise hooks MUST be honored when an American-capable pricer is configured. IV sourcing SHALL follow a fallback chain (`yfinance` option chain → realized 30d volatility → config default), validate IV ∈ (0, 5), and record the chosen source in `run_meta`. **Invalid IV handling**: When IV is outside bounds (0, 5) or unavailable, system SHALL fall back to realized 30d volatility, emit `WARNING: Invalid IV={iv}. Using realized_vol_30d={vol}`, and record `iv_source: "realized_vol"` in run_meta; if realized vol unavailable, fall back to config default (default: 0.25) and record `iv_source: "config_default"`.
```python
# models/options.py

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

@dataclass
class OptionSpec:
    kind: str            # 'call' or 'put'
    strike: float
    maturity_days: int
    implied_vol: float
    risk_free_rate: float = 0.02  # configurable

class BlackScholesPricer:
    def price(self, spot, spec, ttm):
        S = np.asarray(spot, float)
        K = spec.strike
        vol = spec.implied_vol
        r = spec.risk_free_rate
        sqrt_t = np.sqrt(ttm)

        d1 = (np.log(S/K) + (r + 0.5*vol**2)*ttm) / (vol*sqrt_t)
        d2 = d1 - vol*sqrt_t

        if spec.kind == "call":
            return S * norm.cdf(d1) - K * np.exp(-r*ttm) * norm.cdf(d2)
        else:
            return K * np.exp(-r*ttm) * norm.cdf(-d2) - S * norm.cdf(-d1)
```

- **FR-017**: System MUST support both yfinance and Schwab API as historical data providers, using yfinance as the default source for development and allowing promotion of Schwab to primary source via configuration, with automatic fallback to yfinance if the primary source fails.
- **FR-018**: System MUST define and enforce configurable run time and resource limits for grid jobs on the VPS, with the initial defaults that:
  - A single baseline CLI run (e.g., 1,000 paths × 60 steps for one config) SHOULD complete in ≤ 10 seconds on the target VPS.
  - A grid job (e.g., up to 50 configurations with 1,000 paths × 60 steps each) SHOULD complete in ≤ 15 minutes wall-clock time.
  - The system MUST cap concurrent workers (e.g., max_workers ≤ 6 on an 8-core VPS), reserve at least 2 cores for the OS, and abort the run (with structured error) when pre-run estimates exceed thresholds; if estimates are within limits but runtime exceeds, emit escalating warnings and stop remaining jobs.
-  - Worker auto-detection MUST default to `max_workers = min(6, cores-2)` when unspecified.
- **FR-019**: System MUST prevent replay when underlying historical data version has changed since `run_meta` (unless an explicit `--force-replay`/`allow_data_drift` flag is provided), and must record the drift status in the replay output; detected schema drift SHALL block replay, while count/distribution drift SHALL warn and honor `--allow_data_drift`. Drift decisions and data fingerprint deltas MUST be recorded in `run_meta`.
- **FR-020**: System MUST enforce distribution parameter validation (per-model bounds, heavy-tail sanity checks) and fit convergence controls (max iterations, failure fallback to Laplace) with structured errors when validation fails. Comprehensive validation methodology detailed in **US6a** (Distribution Model Validation and Selection).
- **FR-021**: System MUST ensure reproducibility across seeded runs with numeric tolerance of ±1e-10 for path-level values; captures library versions, git SHA, system config (CPU/RAM/OS), and seed in `run_meta` and applies seeding to all MC operations (including conditional sampling).
- **FR-022**: System MUST detect numeric overflow/underflow or non-positive prices during log-return → price transforms and abort with a structured error; paths producing NaN/inf or ≤0 prices SHALL be rejected (no silent clipping).
- **FR-023**: System MUST apply a deterministic memory footprint estimator (`n_paths × n_steps × 8 bytes` plus 10% overhead) before MC execution and enforce storage policy selection based on the estimator (in-memory <25% RAM; memmap/npz ≥25%; abort ≥50%) with user-visible warnings; record the decision in `run_meta`.
- **FR-024**: System MUST define configuration precedence (CLI flags > environment variables > YAML files) and log any overrides applied per run.
- **FR-025**: System MUST fail fast on invalid or incompatible configuration combinations (e.g., pricer not supported for asset type) and MUST define defaults for all optional parameters in configuration schemas.
- **FR-026**: System MUST log component swap events (data source, distribution, pricer, selector) with timestamp, run_id, and prior→new values for auditability.
- **FR-027**: System MUST validate Parquet schemas on load (column names, dtypes, index type) and refuse runs on detected drift unless a compatibility rule is defined.
- **FR-028**: System MUST compute and store data version fingerprints (e.g., SHA256 of Parquet payload + schema hash) in `run_meta` to support drift detection (per FR-019).
- **FR-029**: System MUST enforce missing-data tolerances: fail/run warning when continuous gaps exceed 3× bar interval or total missing bars exceed 1% of window unless an explicit imputation rule is configured.
- **FR-030**: System MUST guarantee atomic, append-only writes of `run_meta.json` and artifacts; metadata is immutable after run completion.
- **FR-031**: Storage policy for Monte Carlo datasets SHALL remain non-persistent by default (DM-009), but persistence is permitted when explicitly requested (replay) or required for memmap fallback; `run_meta` MUST record when persistence is used.
- **FR-032**: System MUST enforce minimum sample sizes per distribution model (Laplace ≥60 bars, Student-T ≥60 bars, Normal ≥60 bars, GARCH-T ≥252 bars) and fail with structured error if unmet. See **US6a** for data preprocessing and model fitting requirements.
- **FR-033**: Each CLI command (`compare`, `grid`, `screen`, `conditional`, `replay`) MUST validate all parameters against contracts/openapi.yaml and reject unknown/invalid parameters with a clear error.
- **FR-034**: Artifacts MUST follow defined formats: metrics JSON/CSV schema, `run_meta.json` with provenance (seeds, versions, fingerprints, git SHA, system config), optional plots/HTML reports; schemas SHALL be versioned. Metrics MUST include VaR/CVaR (parametric + historical) with lookback window and estimator metadata; grid objectives MUST standardize candidate metrics (z-score) before weighting; metrics MUST report `bankruptcy_rate` (fraction of MC paths hitting ≤0) and early-exercise events when applicable.

- **FR-041 (Slippage/Fees Defaults)**: StrategyParams MUST support fee/slippage parameters with conservative defaults of 5 bps per notional trade and $0.65 per option contract; application is per-fill unless overridden. Defaults and overrides MUST be recorded in run_meta and applied consistently across strategies.
- **FR-035**: Candidate episodes MUST capture `(symbol, t0, horizon, state_features)` with horizon > 0; episode construction rules SHALL be documented and applied consistently across backtest and conditional MC.
- **FR-036**: Conditional Monte Carlo SHALL support both bootstrap (non-parametric) and parametric refit methods; fallback order (bootstrap → refit → unconditional) MUST be documented and logged when taken. Selector sparsity MUST trigger a documented fallback (e.g., unconditional MC with warning).
- **FR-037**: Distribution "implausible parameter" thresholds SHALL be defined per model (e.g., scale > 0 and finite; Student-T df ∈ [2, 100]; GARCH parameters within stationarity bounds) and violations must fail fast with structured errors; AR/stationarity diagnostics MUST be logged when applicable. Heavy-tailed validation SHALL require **excess kurtosis ≥ 1.0** for fitted distributions; emit `WARNING` if kurtosis < 0.5 but proceed with fit marked `heavy_tail_warning: true` in run_meta. Comprehensive parameter validation, tail diagnostics, VaR backtesting, and model selection criteria detailed in **US6a** (Distribution Model Validation and Selection).
- **FR-038**: Fail-fast is the default for invalid data/config/fits; recoverable fallbacks (e.g., secondary data source, unconditional MC) MUST emit warnings and record the fallback in logs and run_meta.
- **FR-039**: Logging MUST be structured (JSON) and include at minimum timestamp, run_id, component, event, severity, duration (when applicable), and key parameters; long-running jobs MUST emit progress updates. **Enhanced audit trail** SHALL additionally include: user/process context (user_id, pid, hostname), data lineage tracking (source file paths with versions/timestamps for all loaded data), and per-field config source provenance (CLI flag, ENV var, or YAML file with path/line number) recorded in run_meta.
- **FR-040**: Performance budgets MUST be documented and enforced: data load/fit/MC/strategy eval latencies, MC throughput targets, expected resource utilization, and grid scaling behavior; breaches MUST trigger structured warnings. **Performance threshold policy**: Emit `INFO` log when operation exceeds 1.5× documented budget, `WARNING` at 2× budget, and `ERROR` at 3× budget (ERROR blocks production deployment but not dev runs); apply to data loading, distribution fitting, MC simulation, and strategy evaluation. Performance measurement methodology SHALL be:
  - **Wall-clock time**: Measured via `time.perf_counter()` for end-to-end CLI duration.
  - **CPU time**: Measured via `time.process_time()` for computational work.
  - **Benchmark VPS**: 8 vCPU, 24 GB RAM, NVMe SSD; OS: Ubuntu 22.04; Python 3.11.
  - Measurements MUST be recorded in `run_meta.json` under `performance_metrics` with units (seconds, GB, paths/second).

# **Configuration Safety & Validation Requirements**

- **FR-041**: System MUST define "invalid configuration" exhaustively with the following error conditions:
  1. **Type mismatches**: String provided for integer field (e.g., `mc_paths: "one thousand"`).
  2. **Out-of-range values**: `mc_paths <= 0`, `mc_steps <= 0`, `max_workers <= 0`.
  3. **Unknown keys**: Configuration keys not in schema trigger strict validation errors.
  4. **Incompatible combinations**: E.g., `option_pricer: "heston"` with `asset_type: "stock"`.
  5. **Missing required fields**: `symbol`, `data_source`, `distribution` are mandatory for all runs.
  6. **Invalid enum values**: E.g., `distribution: "unknown_model"` not in `["laplace", "student_t", "normal", "garch_t"]`.
  7. **Constraint violations**: E.g., `strike <= 0`, `maturity_days <= 0`, `implied_vol <= 0` or `> 10`.

- **FR-042**: Configuration validation error messages MUST include:
  - **Field name**: Exact path in config (e.g., `strategy_params.threshold`).
  - **Invalid value**: The value provided by the user.
  - **Constraint**: What was violated (e.g., "must be > 0", "must be one of [...]").
  - **Fix suggestion**: Actionable next step (e.g., "Set mc_paths to a positive integer, e.g., 1000").
  - Example: `ConfigValidationError: Field 'mc_paths' has invalid value 'abc'. Constraint: must be integer > 0. Fix: Set mc_paths to a positive integer, e.g., 1000.`

- **FR-043**: Component wiring mechanism MUST be implemented via:
  - **Factory pattern**: `DataSourceFactory`, `DistributionFactory`, `PricerFactory`, `SelectorFactory`.
  - Each factory MUST maintain a registry (dict) of component types → implementation classes.
  - Configuration key (e.g., `data_source: "yfinance"`) maps to registered class at runtime.
  - Factories MUST raise `ComponentNotFoundError` if key is unregistered.
  - All component swaps MUST be logged with prior/new values in structured logs.

- **FR-044**: Configuration precedence SHALL be enforced as: **CLI flags > Environment variables > YAML files > Built-in defaults**. All precedence overrides MUST be logged at INFO level with format: `Config override: field='max_workers', source='CLI', old=4, new=6`.

- **FR-045**: System MUST define defaults for all optional parameters:
  - `mc_paths`: 1000
  - `mc_steps`: 60
  - `max_workers`: `min(6, os.cpu_count() or 1)`
  - `random_seed`: 42
  - `storage_policy`: "auto" (in-memory < 25% RAM; else memmap)
  - `gap_threshold`: 0.02 (2%)
  - `volume_spike_threshold`: 2.0 (2× avg volume)
  - `distance_threshold`: 2.0 (std devs)

# **Data Integrity & Drift Requirements**

- **FR-046**: "Data drift" SHALL be quantified with:
  1. **Schema drift**: Column name changes, dtype changes, index type changes.
  2. **Value distribution shifts**: Mean/std of returns change by >20% vs prior version.
  3. **Row count deltas**: Total bars change by >5% vs prior version.
  - Drift detection MUST compute hash over `(schema, row_count, mean_return, std_return)` and compare to `run_meta.json` on replay.
  - Drift status (none/schema/distribution/count) MUST be recorded in replay output.

- **FR-047**: "NaN handling" priority order SHALL be:
  1. **Drop**: Remove bars with NaN in critical columns (close, open, high, low, volume) if total dropped < 1% of window.
  2. **Forward-fill**: Propagate last valid observation up to 3× bar interval.
  3. **Backward-fill**: If gap at window start, backfill from first valid bar up to 3× bar interval.
  4. **Abort**: If gaps exceed tolerances, raise `InsufficientDataError` with gap statistics.
  - NaN handling method applied MUST be logged in `run_meta.json`.

- **FR-048**: "Source version" format SHALL be: `{provider}_{semantic_version}_{iso_date}`, e.g., `yfinance_0.2.31_2025-11-15` or `schwab_api_1.0.0_2025-11-15`. Git SHA MAY be appended: `yfinance_0.2.31_2025-11-15_a3f5b2c`.

# **Edge Case & Boundary Condition Requirements**

- **FR-049**: System MUST handle small `n_paths` (1–10) with warnings:
  - Emit `WARNING: n_paths={n_paths} is too small for reliable statistics. Recommend n_paths ≥ 100.`
  - Proceed with run but mark `statistics_reliable: false` in `run_meta.json`.

- **FR-050**: System MUST detect bankruptcy scenarios (all prices → 0) and:
  - Log `ERROR: All paths reached zero or negative prices. Strategy P&L undefined.`
  - Abort run with `BankruptcyError` and record failed paths in diagnostics.

- **FR-051**: System MUST detect zero-volatility scenarios (all returns = 0) and:
  - Log `WARNING: Zero volatility detected (std_return = 0). MC paths will be constant.`
  - Generate constant paths (all equal to S0) and mark `zero_volatility: true` in `run_meta.json`.

- **FR-052**: System MUST handle empty configuration files by:
  - Loading all built-in defaults (per FR-045).
  - Emitting `INFO: Empty config file; using all defaults.`

- **FR-053**: System MUST detect contradictory settings (e.g., `mc_paths: 100000` with `memory_limit_gb: 1`) by:
  - Running preflight memory estimator (FR-023).
  - Raising `ConfigConflictError: Estimated memory ({est_gb} GB) exceeds limit ({limit_gb} GB). Reduce mc_paths or increase limit.`

- **FR-054**: System MUST reject configuration changes mid-grid execution:
  - Grid worker processes snapshot config at start.
  - Config file modifications during grid run SHALL NOT affect in-flight workers.
  - Warn on detection: `WARNING: Config file modified during grid run. Changes ignored for current run.`

- **FR-055**: System MUST handle symbols with constant prices (no volatility) by:
  - Detecting `std(close) = 0` and emitting `WARNING: Constant price detected for {symbol}. Strategy signals undefined.`
  - Skip symbol in screening or mark `skipped: true` in results.

- **FR-056**: System MUST handle symbols with extreme gaps (>50% overnight moves) by:
  - Emitting `WARNING: Extreme gap detected: {gap_pct:.1%} at {timestamp}. Data quality questionable.`
  - Proceed with run but mark `extreme_gaps: [{timestamp, gap_pct}, ...]` in `run_meta.json`.

- **FR-057**: System MUST reject data with future dates or timestamp anomalies:
  - Detect timestamps > `datetime.now(tz=UTC)` and raise `TimestampAnomalyError: Future dates detected: {timestamps}.`
  - Detect non-monotonic timestamps and raise `TimestampAnomalyError: Non-monotonic index detected.`

- **FR-058**: System MUST handle `max_workers` boundary conditions:
  - `max_workers = 1`: Sequential execution with no parallelism (valid).
  - `max_workers > os.cpu_count()`: Clamp to `os.cpu_count()` and emit `WARNING: max_workers clamped to {clamped_value}.`
  - `max_workers <= 0`: Raise `ConfigValidationError: max_workers must be ≥ 1.`

- **FR-059**: System MUST handle grid with single config (degenerates to compare):
  - Detect `len(configs) = 1` and emit `INFO: Single config grid; behaves as compare.`
  - Produce same artifacts as compare command with grid metadata.

- **FR-060**: System MUST handle ATM options (strike = current price) with numerical precision:
  - Use tolerance `abs(strike - spot) < 0.01 * spot` to detect ATM.
  - Apply consistent rounding (e.g., round strike to nearest $0.01) to avoid float precision issues.
  - Log `INFO: ATM option detected (strike={strike}, spot={spot})` when tolerance met.

- **FR-087**: System MUST handle option maturity exactly equal to simulation horizon:
  - When `maturity_days == horizon`, option expires on final simulation step.
  - Final step option value SHALL be computed using intrinsic value only (no time value).
  - For calls: `max(S_final - K, 0)`; for puts: `max(K - S_final, 0)`.
  - Early exercise logic (if configured) SHALL be evaluated up to but not including the final step.
  - Log `INFO: Option maturity equals horizon ({horizon} bars). Final value computed as intrinsic only.`
  - Record `maturity_equals_horizon: true` in `run_meta.json`.

# **Non-Functional Requirements (Reliability, Observability, Maintainability)**

- **FR-061**: System MUST support graceful shutdown during long-running operations:
  - Register signal handlers for SIGTERM and SIGINT (Ctrl+C).
  - On signal, set shutdown flag and allow current worker tasks to finish.
  - Emit `INFO: Shutdown requested. Finishing current tasks...` and exit with status 130 (interrupted).

- **FR-062**: System MUST preserve partial results on failure:
  - Grid jobs SHALL write per-config results immediately after completion (not batched at end).
  - On failure, partial `grid_results.json` MUST contain completed configs with `incomplete: true` flag.

- **FR-063**: System MUST version configuration file format:
  - Config files MUST include `schema_version: "1.0"` at top level.
  - Loader MUST check version and refuse incompatible versions: `ConfigVersionError: Config schema {version} not supported. Expected 1.x.`

- **FR-064**: System MUST maintain backward compatibility for `run_meta.json`:
  - New fields MAY be added with defaults but MUST NOT remove/rename existing fields in v1.x releases.
  - Breaking changes require major version bump (e.g., 2.0) and migration tool.

- **FR-065**: System MUST define artifact cleanup policies:
  - **Default retention**: Keep all artifacts indefinitely (user-managed cleanup).
  - **Optional auto-cleanup**: If `cleanup_policy.max_age_days` is set, purge runs older than threshold on next CLI invocation.
  - Cleanup MUST log: `INFO: Purged {count} runs older than {days} days.`

# **Dependency & Assumption Requirements**

- **FR-066**: System MUST handle pandas-ta indicator unavailability:
  - Detect missing indicators via `try: import pandas_ta except ImportError`.
  - Raise `DependencyError: pandas-ta not installed. Required for feature '{feature_name}'. Install via: pip install pandas-ta`.

- **FR-067**: System MUST enforce scipy/numpy version compatibility:
  - Require `numpy >= 1.24, < 2.0` and `scipy >= 1.10, < 2.0`.
  - Check versions on import and raise `DependencyError: numpy {version} incompatible. Expected >= 1.24, < 2.0.`

- **FR-068**: System MUST document VPS OS and Python constraints:
  - **Supported OS**: Linux (Ubuntu 22.04 or compatible), macOS 12+ (development only).
  - **Python version**: Python 3.11.x (pinned in `.python-version` or `pyproject.toml`).
  - **Unsupported**: Windows, Python <3.11 or ≥3.12.

- **FR-069**: System MUST enforce task execution order:
  - Data load → Schema validation → Distribution fit → MC generation → Strategy eval → Metrics aggregation → Artifact write.
  - Blocking dependencies MUST prevent out-of-order execution (enforced via function call DAG).

- **FR-070**: System MUST ensure data/feature directory structure exists:
  - CLI commands MUST create `data/historical/{interval}/` and `data/features/{interval}/` directories if they do not exist (with appropriate permissions).
  - On first run, emit `INFO: Created data directory structure at {path}.`

- **FR-085**: System MUST implement cache staleness detection and refresh:
  - Cached Parquet files SHALL include metadata: `{"symbol": "AAPL", "interval": "1d", "start": "2020-01-01", "end": "2025-11-15", "source": "yfinance", "fetched_at": "2025-11-15T10:30:00Z", "last_close": 150.25}`.
  - On cache hit, check if requested date range exceeds cached range; if so, fetch incremental data and append.
  - **Corporate action detection**: Before incremental append, fetch overlapping bar (cached end date) from data source. If `abs(cached_close - fresh_close) / cached_close > 0.01` (1% divergence), assume corporate action (split/dividend adjustment) occurred and trigger full refresh with `WARNING: Historical prices adjusted (likely split/dividend). Full refresh required.`
  - Configurable staleness threshold (default: 1 day for daily data, 1 hour for intraday); re-fetch if `now() - fetched_at > staleness_threshold`.
  - User MAY provide `--force_refresh` flag to bypass cache entirely and fetch full history (useful after known corporate actions).
  - Emit `INFO: Using cached data for {symbol} ({start} to {end})` or `INFO: Fetching fresh data for {symbol} (cache stale or partial).`

- **FR-086**: System MUST handle data source failures during on-demand fetch:
  - On network timeout or API error, retry 3× with exponential backoff (1s, 2s, 4s).
  - After retries exhausted, check for stale cache; if present and `--allow_stale_cache` flag set, use stale data with `WARNING`.
  - If no cache and retries exhausted, raise `DataSourceError: Unable to fetch {symbol} from {source} after 3 retries. Check network/API status.`

- **FR-071**: System MUST pin required Python packages:
  - `requirements.txt` or `pyproject.toml` MUST specify exact versions (via `==`) or constrained ranges (via `>=,<`) for all direct dependencies.
  - Critical packages: `numpy==1.26.2`, `pandas==2.1.3`, `scipy==1.11.4`, `numba==0.58.1`, `statsmodels==0.14.0`, `typer==0.9.0`, `pydantic==2.5.0`.

# **Traceability & Reproducibility Requirements**

- **FR-072**: System MUST capture git commit hash in `run_meta.json`:
  - Detect via `subprocess.run(['git', 'rev-parse', 'HEAD'])` and store as `git_commit_sha`.
  - **Dirty working tree handling**: If uncommitted changes exist, record SHA + `dirty: true` + list of modified file paths in `run_meta.uncommitted_changes`.
  - **Missing .git directory**: Store `git_commit_sha: null`, `version_source: "unknown"`, emit `WARNING: Not a git repository. Git SHA not captured.`
  - **Never block runs**: Always proceed with best-effort version capture; treat all git errors as non-fatal.

- **FR-073**: System MUST capture system configuration in `run_meta.json`:
  - Include: `os_type` (Linux/macOS), `os_version`, `python_version`, `cpu_count`, `total_ram_gb`.
  - Example: `{"os_type": "Linux", "os_version": "Ubuntu 22.04", "python_version": "3.11.6", "cpu_count": 8, "total_ram_gb": 24}`.

- **FR-074**: System MUST guarantee numerical reproducibility across CPU architectures:
  - Require IEEE 754 compliance for float64 operations (standard on x86-64, ARM64).
  - For SIMD operations (via numba), set `fastmath=False` to preserve bit-exact results.
  - Document acceptable tolerance: `±1e-10` for path-level values; `±1e-6` for aggregated metrics.

- **FR-075**: System MUST version requirements themselves:
  - Spec document SHALL include `spec_version: "1.0.0"` in frontmatter.
  - Store spec version in `run_meta.json` as `spec_version`.

- **FR-076**: System MUST support backward compatibility testing:
  - Test suite SHALL include "golden run" tests that load `run_meta.json` from prior versions and verify metrics match within tolerance.

- **FR-077**: System MUST define migration paths for breaking changes:
  - Config format changes: Provide `migrate_config.py` script to convert v1 → v2.
  - Schema changes: Provide migration scripts in `migrations/` directory.

# **Warning Level & Tolerance Specification**

- **FR-078**: "Escalating warnings" (FR-018) SHALL be defined as:
  - **Level 1 (INFO)**: Informational messages for normal events (e.g., config overrides, defaults applied).
  - **Level 2 (WARNING)**: Recoverable issues that may affect results (e.g., data gaps filled, selector sparsity fallback).
  - **Level 3 (ERROR)**: Unrecoverable errors requiring abort (e.g., invalid config, distribution fit failure).
  - Long-running operations MUST emit Level 2 warnings at:
    - 50% of time budget: `WARNING: Operation at 50% of time budget.`
    - 90% of time budget: `WARNING: Operation at 90% of time budget. May exceed limit.`
    - 100% of time budget: Abort remaining tasks and exit with ERROR.

- **FR-079**: "Reasonable tolerances" for numeric comparison SHALL be:
  - **Monte Carlo path values**: `±1e-10` (bit-exact when seeded).
  - **Aggregated metrics** (mean P&L, Sharpe): `±1e-6`.
  - **Distribution parameters** (scale, location): `±1e-8`.
  - **Price alignment**: `±1e-2` (one cent for USD).

# **Recovery & Fallback Requirements**

- **FR-080**: Selector sparsity fallback SHALL trigger when:
  - Candidate episode count < `min_episodes` (default: 10).
  - Fallback: Use unconditional MC with full historical window.
  - Emit: `WARNING: Selector found {count} episodes (< {min_episodes}). Falling back to unconditional MC.`
  - Record `fallback_reason: "selector_sparsity"` in `run_meta.json`.

- **FR-081**: Grid job partial failure recovery SHALL:
  - Continue evaluating remaining configs when a config fails.
  - Log failure: `ERROR: Config {index} failed: {error_msg}. Continuing with remaining configs.`
  - Mark failed config in results: `{"config": {...}, "status": "failed", "error": "..."}`.
  - Grid run succeeds if ≥1 config completes; fails if all configs fail.

# **Candidate State Feature Vector Specification**

- **FR-082**: "Candidate state" feature vector MUST include:
  - **Price features** (normalized by S0): `[close/S0, open/S0, high/S0, low/S0]`.
  - **Volume features** (normalized by 20-day avg): `[volume/avg_volume_20d]`.
  - **Return features**: `[log_return_1d, log_return_5d, log_return_20d]`.
  - **Volatility features**: `[realized_vol_20d, realized_vol_60d]`.
  - **Optional indicators**: User-configured (e.g., `[rsi_14, macd, bb_upper]`).
  - All features MUST be z-score normalized: `(x - mean) / std` before distance calculations.
  - Feature vector schema MUST be recorded in `run_meta.json`.

# **Objective Function Specification**

- **FR-083**: Objective function scoring for grid ranking SHALL be defined as:
  - **Formula**: `objective_score = w1 * mean_pnl + w2 * sharpe_ratio + w3 * (-max_drawdown) + w4 * (-cvar_95)`
  - **Default weights**: `w1=0.3, w2=0.3, w3=0.2, w4=0.2` (sum to 1.0).
  - **Normalization**: Each metric MUST be z-score normalized across grid configs before weighting.
  - **Ranking**: Configs sorted descending by `objective_score`.
  - Weights MUST be configurable via `objective_weights` in config; logged in `run_meta.json`.

# **Parallel Execution Model Specification**

- **FR-084**: Parallel execution concurrency model SHALL be:
  - **Process pool**: Use `concurrent.futures.ProcessPoolExecutor` for grid jobs (config-level parallelism).
  - **Shared memory**: None; each worker process has independent memory space.
  - **IPC**: Parent process collects results via return values (no shared state).
  - **Thread pool**: NOT used (GIL contention for CPU-bound MC).
  - **Max workers**: Clamped to `min(max_workers_config, os.cpu_count() - 2)` to reserve 2 cores for OS.
  - Model details MUST be documented in `plan.md` under "Concurrency Model".

# **Candidate Selection Functional Requirements**
- **FR-CAND-001:** System SHALL implement a `CandidateSelector` abstraction that produces candidate timestamps based solely on information available at time t. The system MUST support the following selector methods exhaustively enumerated:
  1. **Gap/Volume Spike** (default, `selector_type: "gap_volume"`): Triggers when `abs(open - prev_close) / prev_close > gap_threshold` AND `volume / avg_volume_20d > volume_spike_threshold`. Default thresholds: `gap_threshold=0.02` (2%), `volume_spike_threshold=2.0` (2× avg volume).
  2. **Custom DSL** (`selector_type: "dsl"`): YAML-based boolean/threshold expressions over computed features without requiring code changes. DSL SHALL support boolean operators (AND, OR, NOT), comparison operators (>, <, >=, <=, ==), and feature references (e.g., `rsi_14 > 70 AND volume > 2 * avg_volume_20d`).
  3. All selector methods MUST be registered via `SelectorFactory` and selected by configuration key. Future selector types (e.g., `selector_type: "ml_classifier"`) SHALL be added via factory registration without modifying core selector abstraction.
- **FR-CAND-002:** System SHALL construct "candidate episodes" (symbol, t0, horizon) for all historical timestamps where the selector fires.
- **FR-CAND-003:** System SHALL support "conditional backtests," where strategies are evaluated only on candidate episodes and performance metrics are computed at the episode level.
- **FR-CAND-004:** System SHALL support "conditional Monte Carlo" that generates simulated price paths conditioned on the current state being similar to historical candidate episodes. "Sufficiently different" for conditional episode matching SHALL be quantified using:
  - **Euclidean distance** in normalized feature space with threshold `distance_threshold` (default: 2.0 standard deviations).
  - **Mahalanobis distance** when covariance matrix is available (optional, configurable via `use_mahalanobis: true`).
  - Distance metric selection MUST be configurable per run and logged in `run_meta.json`.
- **FR-CAND-005:** System SHOULD support at least one non-parametric episode-resampling method and one parametric state-conditioned return model.
- **FR-CAND-006:** System MUST ship at least one default selector (gap/volume spike rule) for MVP and allow selector swapping via configuration without code changes.
- **FR-CAND-006a (Screen CLI inputs/caching):** The `screen` CLI SHALL accept either (a) a CSV universe (`symbol,date,open,high,low,close,volume`) or (b) a comma-delimited `symbols` list with `start/end/interval` inputs. When symbols are provided, the CLI MUST download OHLCV via the same functional path as `fetch`, storing Parquet at `data/historical/interval={interval}/symbol={symbol}/_v1/data.parquet`, updating when newer/longer ranges are requested. If the requested date window is shorter than an existing Parquet file, the CLI MUST serve the slice from the Parquet file (using Parquet time-series filtering) instead of refetching.

```python
# interfaces/candidate_selector.py

from abc import ABC, abstractmethod
import pandas as pd

class CandidateSelector(ABC):
    """Determines whether a symbol is in a 'candidate' state at each timestamp."""

    @abstractmethod
    def score(self, features: pd.DataFrame) -> pd.Series:
        """
        Return a float score per timestamp indicating strength of candidate signal.
        Higher is more likely to be a candidate.
        """

    @abstractmethod
    def select(self, features: pd.DataFrame, threshold: float) -> pd.Series:
        """
        Return boolean Series: True where the candidate condition is met.
        """
```

```python
# modules/episodes.py

from dataclasses import dataclass
import pandas as pd

@dataclass
class CandidateEpisode:
    symbol: str
    t0: pd.Timestamp
    prices: pd.DataFrame
    features: pd.DataFrame
    state_features: pd.Series

def build_candidate_episodes(
    symbol: str,
    prices: pd.DataFrame,
    features: pd.DataFrame,
    selector,
    horizon: int,
):
    selected = selector.select(features, threshold=selector.threshold)
    episodes = []

    for t0 in selected[selected].index:
        window = slice(t0, t0 + pd.Timedelta(days=horizon))
        episodes.append(
            CandidateEpisode(
                symbol=symbol,
                t0=t0,
                prices=prices.loc[window],
                features=features.loc[window],
                state_features=features.loc[t0],
            )
        )

    return episodes
```

# **Data Management Functional Requirements (DM-Series)**

### **DM-015: Parquet Schema Validation**
The system SHALL validate Parquet schemas on load (column names, dtypes, index) and fail fast on drift unless a compatibility rule is configured.

### **DM-016: Data Fingerprinting for Drift Detection**
The system SHALL compute a deterministic fingerprint (e.g., SHA256 over data block + schema hash) per Parquet file and store it in run metadata to detect drift (with optional `--force-replay` override).

### **DM-017: Missing Data Tolerances**
The system SHALL fail or warn when continuous gaps exceed **3× bar interval** or total missing bars exceed **1% of the window**, unless an explicit imputation strategy is configured and recorded.

### **DM-018: Run Metadata Durability**
Run metadata SHALL be written atomically and treated as immutable after run completion; retries must not produce partial or corrupted metadata files.

### Conflict Resolution Notes
- **Memory thresholds (FR-013 vs FR-018)**: Both use the same estimator (<25% RAM in-memory; otherwise memmap/npz) and max_workers ≤ 6 on 8-core VPS; preflight checks enforce both.
- **Storage policy (DM-009 vs US8 replay)**: Default is non-persistent; persistence only when explicitly requested or required for memmap/replay, recorded in run_meta (FR-031).
- **Fail-fast vs fallback (FR-010 vs recovery)**: Fail-fast is default (FR-038); fallbacks (secondary data sources, unconditional MC) are allowed only with explicit warnings + run_meta entries.
- **max_workers alignment (CHK131)**: Resolved by clamping implementation to `min(max_workers_config, os.cpu_count() - 2)` with default=6 for 8-core VPS (per FR-058, FR-084). Contracts/OpenAPI SHALL accept any positive integer but document recommended max=6 for 8-core baseline. Runtime enforcement via clamp (not schema rejection) allows flexibility for larger VPS configs while maintaining safe defaults.

### **DM-001: Resolution Selection – Daily Bars**
The system SHALL use **daily OHLCV bars** for:
* Fitting long-horizon distribution models (Laplacian, Normal, Student-T, GARCH-T, etc.)
* Estimating volatility, drift, heavy tails, and regime behavior
* Portfolio-level allocation optimization
* Strategy prototyping and model development

### **DM-002: Resolution Selection – 5-Minute Bars**
The system SHALL use **5-minute OHLCV bars** for:
* Backtesting short-term strategies with holding periods from intraday to multiple days
* Modeling realistic intraday drawdowns and reversals
* Evaluating intraday entry/exit accuracy for swing-to-multi-day strategies

### **DM-003: Resolution Selection – 1-Minute Bars**
The system SHALL use **1-minute OHLCV bars** exclusively for:
* Live or near-live signal generation
* High-resolution execution modeling
* Intraday risk monitoring for symbols currently being traded (“Live Set”)

### **DM-004: Historical Data Storage Format**
The system SHALL store **all historical OHLCV data** in **Parquet** format with:
* Lossless columnar layout
* Compression (Snappy or ZSTD recommended)
* One file per symbol per resolution, unless partitioning by year is chosen

### **DM-005: Historical vs Derived Feature Separation**
The system SHALL store raw OHLCV and derived indicator/feature data in **separate Parquet files**, to preserve canonical raw data and enable fast recomputation or versioning of indicators.

### **DM-006: Historical Data Directory Layout**
The system SHALL organize historical data under:
```
data/historical/{interval}/{symbol}.parquet
```

And derived features under:
```
data/features/{interval}/{symbol}_features.parquet
```

### **DM-007: Universe/Watchlist/Live Sets**
The system SHALL maintain three symbol tiers:
1. **Universe:**
   * Hundreds of symbols
   * Store daily + 5-minute OHLCV in Parquet
2. **Watchlist:**
   * Dozens of symbols
   * Store daily + 5-minute OHLCV + derived feature files
3. **Live Set:**
   * 5–20 symbols
   * Store or stream daily + 5-minute + 1-minute OHLCV

### **DM-008: Monte Carlo Generation Default Behavior**
The system SHALL generate Monte Carlo paths **in memory by default** when the total required memory is **< 25% of available RAM** (≈6 GB on your 24 GB VPS).

### **DM-009: Monte Carlo Reproducibility Without Storing Paths**
Monte Carlo simulations SHALL be reproducible through:
* Stored distribution parameters
* Random seeds
* Simulation metadata (n_paths, n_steps, model type)

The system SHOULD NOT persist MC path matrices by default.

### **DM-010: Persistent Monte Carlo Storage Conditions**
The system SHALL persist Monte Carlo path datasets **only when**:
* They exceed 25% of RAM, **OR**
* They must be reused across many experiments, **OR**
* They are required for scenario replay / stress testing
Persisted MC datasets SHALL be stored in `.npz` format.

### **DM-011: Large Monte Carlo Datasets**
When Monte Carlo arrays exceed **50% of RAM**, the system SHALL use **`numpy.memmap`** backed by disk files to:
* Generate paths in chunks
* Stream paths during backtesting
* Avoid full memory loads

### **DM-012: Monte Carlo Storage Format**
Monte Carlo datasets that require persistence SHALL be stored as:
* `.npz` files for arrays intended to be reloaded
* `numpy.memmap` files when arrays exceed memory or require partial loading

### **DM-013: Historical Data Retention**
The system SHALL treat Parquet historical OHLCV data as **canonical source data** and retain it indefinitely with appropriate versioning (e.g., `_v2`, `_v3` when source changes).

### **DM-014: Monte Carlo Data as Ephemeral**
Monte Carlo path arrays SHALL be treated as **ephemeral** and only retained when required for reproducibility or stress tests.

---

# **Data Management Non-Functional Requirements (DNFR-Series)**
### **DNFR-001: Storage Efficiency**
The system SHOULD minimize disk usage by:
* Using compressed Parquet for historical data
* Avoiding unnecessary storage of Monte Carlo datasets
* Using `.npz` only when persistence is explicitly required

### **DNFR-002: Memory Efficiency**
Monte Carlo simulations SHOULD:
* Use in-memory arrays when <25% RAM
* Use memmap-backed arrays beyond that threshold
* Support streaming and chunked processing for very large paths

### **DNFR-003: Scalability With Resolution**
The system SHOULD allow:
* Daily and 5-minute historical datasets for hundreds of symbols
* 1-minute datasets only for a small Live Set
* Efficient querying and slicing of Parquet partitions by symbol and interval

### **DNFR-004: Fast Reproducibility**
The system SHALL guarantee reproducibility of Monte Carlo runs using:
* Stored distribution parameters
* Simulation metadata
* Random seeds
* Versioned model configurations

### Key Entities *(include if feature involves data)*

- **DataSource**: Provider for OHLCV/macro series; attributes include symbol, start/end, interval.  
- **ReturnDistribution**: Fitted model with parameters and `sample(n_paths, n_steps)` method.  
- **PricePath**: Simulated price matrix derived from ReturnDistribution; metadata includes seed, S0, distribution name.  
- **StrategyParams**: Parameter set for stock/option strategies (thresholds, position size, DTE, IV, strike offset).  
- **OptionSpec**: Option contract inputs (kind, strike, maturity_days, implied_vol, r).  
- **SimulationRun**: Execution record containing inputs (config, seed), outputs (equity curves, metrics), and artifact locations.  
- **MetricsReport**: Aggregated metrics per run/config (P&L stats, drawdown, VaR/CVaR, Sharpe/Sortino, objective score).

## Success Criteria *(mandatory)*

### Success Criteria Measurable Outcomes
- **SC-001**: A default CLI run (e.g., 1,000 paths × 60 steps) completes on the target VPS within the documented time budget (to be defined in FR-018).  
- **SC-002**: For any run, outputs include both stock and option metrics plus saved artifacts (CSV/JSON and optional plot/HTML) in a run directory.  
- **SC-003**: Grid runs produce ranked configurations sorted descending by objective score (per FR-083: weighted sum of z-score normalized mean_pnl, sharpe_ratio, -max_drawdown, -cvar_95 with default weights 0.3/0.3/0.2/0.2) and complete without data races or corrupted outputs. Sample output SHALL include config parameters, raw metrics, normalized metrics, weights applied, and final objective_score per config.  
- **SC-004**: Simulations are reproducible when a seed is provided (identical metrics on repeat runs).  
- **SC-005**: The system surfaces clear errors/warnings for missing data, failed fits, or resource-limit breaches, without silent failures.  
- **SC-006**: Swapping data source (yfinance ↔ Schwab stub) or option pricer requires only configuration changes, not code edits.

### Monte Carlo & Conditional Backtesting Success Criteria
* **SC-007**: For a fixed symbol and configuration, Monte Carlo runs with the same seed and parameters produce identical path-level outcomes and episode-level metrics across at least 3 repeated runs.
* **SC-008**: The system can run a conditional backtest over at least 100 historical candidate episodes for a single symbol and complete within the documented time budget (per FR-018).
* **SC-009**: For a given strategy, the system can produce and persist a comparative risk–return summary (e.g., mean P&L, CVaR, max drawdown) for stock vs option variants, conditioned only on candidate episodes, in a single CLI command.

### Candidate Selection & Episodic Evaluation Success Criteria
* **SC-010**: Given a configured `CandidateSelector`, the system can scan at least 5 years of daily data for a universe of ≥100 symbols and output a candidate episode list (symbol, t0, horizon) without errors.
* **SC-011**: When conditional backtesting is enabled, the system reports both unconditional metrics (all bars) and conditional metrics (candidate-only episodes) for each strategy configuration in the resulting artifacts.
* **SC-012**: Changing the candidate selection criteria (e.g., gap/volume threshold) requires no changes to core backtest or strategy code and results in a different episode set that is clearly logged and traceable.

### Data Management & Storage Policy Success Criteria
* **SC-013**: Historical OHLCV and feature data are written and reloaded from Parquet without schema drift (column names, dtypes, index) across at least one full export/import cycle.
* **SC-014**: When Monte Carlo arrays exceed the configured in-memory threshold, the system automatically falls back to `.npz`/`memmap` storage and still completes the run without raising out-of-memory errors.
* **SC-015**: For a given run ID, the system outputs a metadata file (e.g., `run_meta.json`) that fully describes data sources, model type, distribution parameters, random seeds, and key configuration flags, sufficient to reproduce the run.

### Modularity & Swap-Ability Success Criteria
* **SC-016**: Replacing the return distribution model (e.g., Normal → Student-T → GARCH-T) via configuration results in different Monte Carlo path behavior and metrics, with no code changes outside configuration or dependency injection wiring.
* **SC-017**: Replacing the option pricer (e.g., Black–Scholes → alternative pricer) via configuration produces different option-equity curves while leaving stock-equity curves unchanged, confirming isolation of concerns.
* **SC-018**: Plugging in a new strategy class (stock or option) that adheres to the `Strategy` interface requires no modification to the simulator or optimization modules to be included in CLI runs.

### Reliability, Error Handling, and Observability Success Criteria
* **SC-019**: When distribution fitting fails (e.g., insufficient data, non-convergence), the system logs a structured error explaining the cause, aborts the run gracefully, and exits with a non-zero status code.
* **SC-020**: When data gaps or missing values exceed a configurable tolerance, the system emits explicit warnings and either (a) drops affected episodes or (b) imputes as configured, recording the choice in the run metadata.
* **SC-021**: Each run directory contains sufficient logs (timestamps, parameters, progress markers) to trace which configurations were evaluated and in what order, enabling post-mortem analysis of grid or batch runs.

### Configuration Safety & Validation Success Criteria
* **SC-022**: Given any invalid configuration (per FR-041 enumeration), the system fails fast with a structured error message containing field name, invalid value, constraint, and fix suggestion (per FR-042).
* **SC-023**: Given a configuration with contradictory settings (e.g., high mc_paths + low memory_limit_gb), the preflight estimator detects the conflict and raises `ConfigConflictError` before starting the run (per FR-053).
* **SC-024**: Given an empty configuration file, the system loads all built-in defaults (per FR-045) and completes a run successfully with `INFO: Empty config file; using all defaults` logged.
* **SC-025**: Given a configuration change mid-grid execution, in-flight workers complete with original config and log `WARNING: Config file modified during grid run. Changes ignored for current run` (per FR-054).

### Data Integrity & Drift Success Criteria
* **SC-026**: Given a Parquet file with schema drift (column rename), the system detects drift on load and raises `SchemaError` unless a compatibility rule is defined (per FR-027).
* **SC-027**: Given a replay command for a run where data has drifted (per FR-046 metrics), the system refuses replay by default and requires `--allow_data_drift` flag to proceed (per FR-019).
* **SC-028**: Given data with NaN values, the system applies the priority order (drop → forward-fill → backward-fill → abort per FR-047) and logs the method used in `run_meta.json`.
* **SC-029**: Given data with extreme gaps (>50% overnight move), the system emits warning and records `extreme_gaps` metadata per FR-056.

### Edge Case & Boundary Condition Success Criteria
* **SC-030**: Given `n_paths=5`, the system emits `WARNING` about unreliable statistics, proceeds with run, and marks `statistics_reliable: false` in `run_meta.json` (per FR-049).
* **SC-031**: Given MC paths where all prices → 0 (bankruptcy scenario), the system aborts with `BankruptcyError` and records failed path count in diagnostics (per FR-050).
* **SC-032**: Given data with zero volatility (`std=0`), the system generates constant paths, emits `WARNING`, and marks `zero_volatility: true` in `run_meta.json` (per FR-051).
* **SC-033**: Given `max_workers=1`, the system runs sequentially without parallelism and completes successfully (per FR-058).
* **SC-034**: Given `max_workers=12` on an 8-core VPS, the system clamps to `8` and emits `WARNING: max_workers clamped to 8` (per FR-058).
* **SC-035**: Given a grid with single config, the system detects degenerate case, emits `INFO`, and produces artifacts equivalent to compare command (per FR-059).
* **SC-036**: Given ATM option (strike ≈ current price within 1% tolerance), the system applies consistent rounding and logs `INFO: ATM option detected` (per FR-060).
* **SC-037**: Given data with future timestamps, the system raises `TimestampAnomalyError` and aborts (per FR-057).
* **SC-038**: Given symbol with constant price (`std=0`), the system skips in screening with `WARNING` or marks `skipped: true` (per FR-055).
* **SC-073**: Given option with `maturity_days == horizon`, the system computes final option value as intrinsic only, logs `INFO: Option maturity equals horizon`, and records `maturity_equals_horizon: true` in `run_meta.json` (per FR-087).

### Non-Functional Success Criteria (Reliability, Maintainability)
* **SC-039**: Given SIGTERM or SIGINT during grid run, the system finishes current tasks, writes partial results, and exits with status 130 (per FR-061).
* **SC-040**: Given a grid job where 3/10 configs fail, the system writes `grid_results.json` with 7 successful configs and 3 marked `status: "failed"` (per FR-062, FR-081).
* **SC-041**: Given a config file with `schema_version: "2.0"` (unsupported), the system refuses to load and raises `ConfigVersionError` (per FR-063).
* **SC-042**: Given artifact cleanup policy `max_age_days: 30`, the system purges runs older than 30 days on next CLI invocation and logs count purged (per FR-065).

### Dependency & Assumption Success Criteria
* **SC-043**: Given missing `pandas-ta` when indicator is configured, the system raises `DependencyError` with installation instructions (per FR-066).
* **SC-044**: Given incompatible `numpy==2.0.0`, the system checks version on import and raises `DependencyError` (per FR-067).
* **SC-045**: Given missing `data/historical/1d/` directory on first run, the CLI creates the directory structure and emits `INFO: Created data directory structure` (per FR-070).

### On-Demand Data Fetching Success Criteria
* **SC-068**: Given a symbol not present in local cache, the system automatically fetches OHLCV from configured data source (yfinance), writes to Parquet cache, and proceeds with run without user intervention (per FR-001).
* **SC-069**: Given cached data that is stale (fetched >1 day ago for daily data), the system detects staleness, re-fetches from data source, updates cache, and emits `INFO: Fetching fresh data for {symbol} (cache stale)` (per FR-085).
* **SC-070**: Given cached data with partial date range (cached: 2020-2023, requested: 2020-2025), the system fetches incremental data (2023-2025), appends to cache, and proceeds without re-downloading entire history (per FR-085).
* **SC-071**: Given a data source fetch failure (network timeout, API error) after 3 retries, and no cached data available, the system raises `DataSourceError` with actionable message including retry count and suggested fixes (per FR-086).
* **SC-072**: Given a data source fetch failure and stale cache present, when user provides `--allow_stale_cache` flag, the system uses stale data with `WARNING: Using stale cached data due to fetch failure` (per FR-086).

### Traceability & Reproducibility Success Criteria
* **SC-046**: Given any successful run, `run_meta.json` includes `git_commit_sha`, `spec_version`, `python_version`, `cpu_count`, `total_ram_gb`, and all component versions (per FR-072, FR-073, FR-075).
* **SC-047**: Given two runs with same seed/config on same VPS, path-level values match within `±1e-10` and metrics within `±1e-6` (per FR-074, FR-079).
* **SC-048**: Given two runs with same seed/config on different CPU architectures (x86-64 vs ARM64), metrics match within `±1e-6` (per FR-074).

### Candidate Selection & Conditional MC Success Criteria
* **SC-049**: Given selector configuration `selector_type: "gap_volume"` with thresholds, the system produces candidate episodes using enumerated method (per FR-CAND-001).
* **SC-050**: Given conditional MC with `distance_metric: "euclidean"` and `distance_threshold: 2.0`, the system selects similar historical episodes within threshold and logs metric in `run_meta.json` (per FR-CAND-004).
* **SC-051**: Given selector sparsity (<10 episodes), the system falls back to unconditional MC, emits `WARNING`, and records `fallback_reason: "selector_sparsity"` (per FR-080).

### Grid Ranking & Objective Function Success Criteria
* **SC-052**: Given a grid of configs, the system computes `objective_score` per FR-083 formula, ranks configs descending, and outputs ranked list in `grid_results.json` (addresses CHK051).
* **SC-053**: Given custom objective weights `objective_weights: {w1: 0.5, w2: 0.5, w3: 0, w4: 0}`, the system applies custom weights and logs them in `run_meta.json` (per FR-083).

### Data Management Success Criteria (DM-Series Mappings)
* **SC-054 (DM-001/002/003)**: System successfully loads daily, 5-minute, and 1-minute OHLCV from Parquet for intended use cases (model fitting, backtesting, live signals).
* **SC-055 (DM-004/005/006)**: System stores and reloads historical OHLCV and derived features from Parquet following directory layout without schema drift.
* **SC-056 (DM-007)**: System maintains Universe/Watchlist/Live Set tiering with appropriate data resolutions per tier.
* **SC-057 (DM-008/009)**: System generates MC paths in memory by default (<25% RAM) and does not persist unless required (per FR-031).
* **SC-058 (DM-010/011/012)**: When MC exceeds 25% RAM threshold, system automatically uses memmap/npz storage and logs transition (per FR-013/FR-023).
* **SC-059 (DM-013)**: Historical Parquet data is versioned and retained; replays detect version changes and refuse unless forced (per FR-019).
* **SC-060 (DM-014)**: MC path arrays are treated as ephemeral; only persisted when explicitly required or memmap fallback triggered.

### Performance Budget Validation Success Criteria
* **SC-061**: Baseline run (1,000 paths × 60 steps) completes in ≤10s on benchmark VPS (8 vCPU, 24 GB RAM, Ubuntu 22.04, Python 3.11) with wall-clock time measured via `time.perf_counter()` (per FR-040).
* **SC-062**: Grid run (≤50 configs, 1,000 paths × 60 steps each) completes in ≤15 minutes on benchmark VPS with `max_workers=6` (per FR-018, FR-040).
* **SC-063**: Distribution fit (Laplace/Student-T) completes in ≤1s per symbol window (per plan.md performance budget).
* **SC-064**: MC generation achieves ≥50k steps/second aggregate throughput on benchmark VPS (per plan.md performance budget).
* **SC-065**: When operation exceeds 90% of time budget, system emits `WARNING: Operation at 90% of time budget` (per FR-078).

### VPS Configuration Variability Success Criteria
* **SC-066**: Given 4 vCPU VPS, baseline run completes in ≤15s (1.5× slowdown vs 8 vCPU baseline) (addresses CHK079).
* **SC-067**: Given 16 vCPU VPS, grid run with `max_workers=14` achieves ≤8 minutes (2× speedup vs 8 vCPU baseline) (addresses CHK079).

### User Story Acceptance → Success Criteria Traceability
* **US1** (Compare) → SC-001, SC-002, SC-004, SC-005
* **US2** (Grid) → SC-003, SC-052, SC-053, SC-062
* **US3** (Features) → SC-043 (pandas-ta)
* **US4** (Screening) → SC-010, SC-049, SC-051
* **US5** (Conditional backtest) → SC-008, SC-009, SC-011
* **US6** (Conditional MC) → SC-007, SC-050, SC-051
* **US7** (Config swapping) → SC-006, SC-017, SC-022
* **US8** (Replay) → SC-015, SC-027, SC-046, SC-047, SC-048

### Functional Requirements → Success Criteria Traceability
* **FR-001** (Data loading + on-demand fetch) → SC-054, SC-068
* **FR-002** (Distributions) → SC-004, SC-016, SC-063
* **FR-003** (Stock strategy) → SC-002
* **FR-004** (Option strategy) → SC-002
* **FR-005** (CLI) → SC-001, SC-005
* **FR-006** (Features) → SC-043
* **FR-007** (Grid) → SC-003, SC-052
* **FR-008** (Artifacts) → SC-002, SC-021
* **FR-009** (Config swapping) → SC-006, SC-017
* **FR-010** (Missing data) → SC-020, SC-028
* **FR-011** (Stubs) → SC-006
* **FR-012** (Seeding) → SC-004, SC-007, SC-047
* **FR-013** (Memory limits) → SC-014, SC-058
* **FR-014** (Alignment) → SC-020
* **FR-015** (Performance docs) → SC-061-SC-067
* **FR-016** (Option pricing) → SC-017
* **FR-017** (Data sources) → SC-006
* **FR-018** (Resource limits) → SC-001, SC-003, SC-062, SC-065
* **FR-019** (Replay) → SC-027, SC-059
* **FR-020** (Distribution validation) → SC-019
* **FR-021** (Reproducibility) → SC-007, SC-047, SC-048
* **FR-022** (Overflow handling) → SC-031
* **FR-023** (Memory estimator) → SC-014, SC-058
* **FR-024** (Config precedence) → SC-022
* **FR-025** (Fail-fast config) → SC-022
* **FR-026** (Component logging) → SC-006
* **FR-027** (Schema validation) → SC-026
* **FR-028** (Data fingerprinting) → SC-027
* **FR-029** (Missing data tolerances) → SC-028
* **FR-030** (Metadata durability) → SC-046
* **FR-031** (Storage policy) → SC-057, SC-058
* **FR-032** (Minimum samples) → SC-019
* **FR-033** (CLI validation) → SC-022
* **FR-034** (Artifact formats) → SC-002, SC-046
* **FR-035** (Episode construction) → SC-010
* **FR-036** (Conditional MC methods) → SC-050, SC-051
* **FR-037** (Implausible params) → SC-019
* **FR-038** (Fail-fast vs fallback) → SC-019, SC-051
* **FR-039** (Structured logging) → SC-021
* **FR-040** (Performance budgets) → SC-061-SC-065
* **FR-041** (Invalid config enumeration) → SC-022
* **FR-042** (Error messages) → SC-022
* **FR-043** (Component wiring) → SC-006, SC-017
* **FR-044** (Config precedence) → SC-022
* **FR-045** (Defaults) → SC-024
* **FR-046** (Data drift) → SC-027
* **FR-047** (NaN handling) → SC-028
* **FR-048** (Source version format) → SC-046
* **FR-049** (Small n_paths) → SC-030
* **FR-050** (Bankruptcy) → SC-031
* **FR-051** (Zero volatility) → SC-032
* **FR-052** (Empty config) → SC-024
* **FR-053** (Contradictory config) → SC-023
* **FR-054** (Mid-grid config change) → SC-025
* **FR-055** (Constant price) → SC-038
* **FR-056** (Extreme gaps) → SC-029
* **FR-057** (Timestamp anomalies) → SC-037
* **FR-058** (max_workers boundaries) → SC-033, SC-034
* **FR-059** (Single-config grid) → SC-035
* **FR-060** (ATM precision) → SC-036
* **FR-061** (Graceful shutdown) → SC-039
* **FR-062** (Partial results) → SC-040
* **FR-063** (Config versioning) → SC-041
* **FR-064** (Backward compat) → SC-046
* **FR-065** (Cleanup policies) → SC-042
* **FR-066** (pandas-ta fallback) → SC-043
* **FR-067** (numpy/scipy compat) → SC-044
* **FR-068** (OS/Python constraints) → SC-046
* **FR-069** (Task order) → (enforced by implementation DAG)
* **FR-070** (Directory pre-existence) → SC-045
* **FR-071** (Package pinning) → SC-046
* **FR-072** (Git SHA capture) → SC-046
* **FR-073** (System config capture) → SC-046
* **FR-074** (Cross-arch repro) → SC-048
* **FR-075** (Spec versioning) → SC-046
* **FR-076** (Backward compat tests) → SC-048
* **FR-077** (Migration paths) → SC-041
* **FR-078** (Warning levels) → SC-065
* **FR-079** (Numeric tolerances) → SC-047, SC-048
* **FR-080** (Selector sparsity) → SC-051
* **FR-081** (Grid partial failure) → SC-040
* **FR-082** (Candidate state features) → SC-050
* **FR-083** (Objective function) → SC-052, SC-053
* **FR-084** (Parallel execution model) → SC-062
* **FR-085** (Cache staleness detection) → SC-069, SC-070
* **FR-086** (Data source failure handling) → SC-071, SC-072
* **FR-CAND-001** (Selector abstraction) → SC-049
* **FR-CAND-002** (Episode construction) → SC-010
* **FR-CAND-003** (Conditional backtest) → SC-008, SC-009, SC-011
* **FR-CAND-004** (Conditional MC) → SC-007, SC-050
* **FR-CAND-005** (Methods) → SC-050, SC-051
* **FR-CAND-006** (Default selector) → SC-049

## Assumptions & Constraints

### Validated Assumptions (explicitly documented and enforced)
- **ASSUME-001**: VPS has **24 GB RAM** available. System MUST detect total RAM via `psutil.virtual_memory()` and validate ≥ 20 GB available (per FR-073).
- **ASSUME-002**: VPS has **8 vCPU cores** available. System MUST detect via `os.cpu_count()` and validate ≥ 4 cores (per FR-073).
- **ASSUME-003**: **Single-user execution** (no concurrent CLI instances from different users). System SHALL NOT implement file locking but SHOULD emit `WARNING` if concurrent runs to same output directory detected via PID file check.
- **ASSUME-004**: **Historical data may be cached locally** in Parquet format for performance. System SHALL fetch on-demand from configured data source (yfinance/Schwab) if not present in cache or if cache is stale (per FR-001, FR-085, FR-086). Users do NOT need to manually download data before first run.
- **ASSUME-005**: User has **write permissions** to `data/` and `runs/` directories. System MUST check on startup and raise `PermissionError` if unable to create required directories (per FR-070).
- **ASSUME-006**: **No live trading execution** in MVP scope. All runs are research/backtest only; no brokerage API integration beyond data ingestion stubs.
- **ASSUME-007**: **CPU-only execution** (no GPU). All numpy/scipy operations use CPU; no CUDA/cuBLAS/cuDNN dependencies.
- **ASSUME-008**: **Linux or macOS environment**. Windows is unsupported (per FR-068). System MUST detect OS on startup and emit `WARNING` if running on Windows.
- **ASSUME-009**: **Python 3.11.x** is installed and active. System MUST check `sys.version_info` and raise error if Python <3.11 or ≥3.12 (per FR-068).
- **ASSUME-010**: **Network connectivity** available for data source APIs (yfinance, Schwab). System SHALL handle network timeouts gracefully with retries (3× with exponential backoff).
- **ASSUME-011**: **IEEE 754 compliant float64** arithmetic on CPU. System MUST document that bit-exact reproducibility requires same CPU architecture family (per FR-074).

### Development Velocity Assumptions
- **ASSUME-012**: "Fast to prototype" means **≤2 weeks** from spec approval to working MVP (baseline compare + grid + screening) with ≥80% test coverage. This is a project management target, not a functional requirement.
- **ASSUME-013**: Core development is **single-developer** with occasional pairing. Code review is self-review + automated lint/type checks until external reviewer available.

## Clarifications & Anti-Requirements

### Clarification Sessions

#### Session 2025-11-16
- Q: How should the system quantify and validate that a fitted distribution is appropriately "heavy-tailed" for financial returns? → A: Excess kurtosis ≥ 1.0 with warning if < 0.5
- Q: When option pricing encounters invalid IV (implied volatility outside reasonable bounds), what should the system do? → A: Fallback to realized 30d volatility with warning and record iv_source in run_meta
- Q: What additional information should be included in the audit trail beyond the minimum logging requirements already specified? → A: Add user/process context (user_id, pid, hostname), data lineage (source files with versions), and per-field config source tracking (CLI/ENV/YAML)
- Q: What specific performance threshold breaches should trigger structured warnings or optimization requirements? → A: Tiered approach - INFO at 1.5× budget, WARNING at 2×, ERROR at 3× (blocks production); apply to data load, fitting, MC, strategy eval
- Q: How should the system capture version control state when git SHA is unavailable or workspace has uncommitted changes? → A: Best-effort capture via git rev-parse HEAD; record dirty=true + modified files if uncommitted changes; record git_sha=null with WARNING if no .git; never block runs

### Design Clarifications

- **Conditional Backtesting vs Conditional Monte Carlo**: Conditional backtesting replays historical episodes where a selector fired; conditional MC resamples/parameterizes returns conditioned on the same state and then simulates synthetic paths. Both must share selector definitions but produce different artifacts (historical metrics vs simulated distributions).
- **Anti-requirements**: MVP will not execute live trades, will not require GPUs, and will not store brokerage credentials in-repo; all runs are research-only (per ASSUME-006, ASSUME-007).

## Glossary (shared across plan/data-model)

- **candidate**: A symbol + timestamp emitted by a selector rule.
- **candidate episode**: `(symbol, t0, horizon, state_features)` slice derived from a candidate for evaluation.
- **conditional backtest**: Historical replay limited to candidate episodes.
- **conditional MC**: Monte Carlo paths generated from state-conditioned returns/bootstraps matching candidate state.
- **unconditional MC**: Monte Carlo paths from a model fitted on the full chosen window without conditioning.
- **run**: A single execution of compare/grid/conditional producing artifacts under `runs/<run_id>/`.
- **grid**: A batch of strategy configurations evaluated in parallel.
