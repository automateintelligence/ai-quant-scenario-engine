# Feature Specification: Backtesting & Strategy Spec Authoring

**Hierarchy**: Parent constitution at `.specify/memory/CONSTITUTION.md`; this spec governs downstream artifacts (`plan.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`).

**Feature Branch**: `001-mvp-pipeline`  
**Created**: 2025-11-15  
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

### User Story 5 - Conditional episode backtesting for candidates (Priority: P1)
Author runs a conditional backtest that evaluates a strategy only on historical episodes where the candidate selector would have fired, and receives episode-level and aggregate metrics for stock vs option variants.

**Why this priority**: Directly answers “How does this strategy behave given the weird transient conditions that make this stock a candidate?” rather than generic average performance.
**Independent Test**: Define a candidate selector (e.g., large gap + volume spike), build at least 50 historical episodes for a test symbol, run the conditional backtest, and verify episode-level and aggregate metrics are produced for both stock and option strategies.

**Acceptance Scenarios**:
1. **Given** a candidate selector and 5+ years of historical data for a symbol, **When** the user runs the conditional backtest, **Then** the system constructs episodes (symbol, t₀, horizon) and computes P&L for each episode for both stock and option strategies.
2. **Given** conditional backtesting is enabled, **When** a run completes, **Then** the outputs include both unconditional metrics (all bars) and conditional metrics (candidate-only episodes), clearly labeled and separated.
3. **Given** a change in candidate criteria (e.g., new thresholds), **When** the conditional backtest is re-run, **Then** the episode set changes and this change is reflected in the run metadata and metrics without modifying strategy or simulator code.

---

### User Story 6 - Conditional Monte Carlo from candidate states (Priority: P2)
Author generates Monte Carlo paths that are specifically conditioned on a current candidate state (e.g., big gap with volume spike), runs stock vs option strategies over those paths, and reviews the conditional risk/reward distribution.

**Why this priority**: Bridges real historical episodes and synthetic scenarios, allowing “what-if” analysis given today’s specific transient conditions.
**Independent Test**: For a symbol and a chosen candidate state, run a conditional Monte Carlo CLI (e.g., 1,000 paths × 60 steps) and verify that the run uses the state-conditioned model and produces summary distributions for stock and option P&L.

**Acceptance Scenarios**:
1. **Given** a current candidate state vector (features at t₀) and a chosen conditioning method (episode resampling or state-conditioned distribution), **When** the user runs the conditional Monte Carlo command, **Then** the system generates simulated paths that reflect that state and computes stock and option metrics over those paths.
2. **Given** the user supplies a fixed random seed, **When** the conditional Monte Carlo run is repeated with the same parameters, **Then** the path-level outcomes and summary metrics are identical (within numeric tolerances).
3. **Given** the selected conditioning method cannot be applied (e.g., too few similar historical episodes), **When** the user runs the command, **Then** the system emits a clear warning and either falls back to a configured default model or aborts gracefully with an error.

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

- **FR-001**: System MUST load OHLCV for a symbol and date range from a pluggable data source (yfinance baseline; Schwab later) and validate presence of required columns.
- **FR-002**: System MUST fit Laplacian (double-exponential) returns as the default heavy-tailed model and generate N Monte Carlo paths of length T from that fit; additional models (e.g., Student-T) MAY be provided as alternatives.
- **FR-003**: System MUST run a stock strategy over generated paths, producing per-path equity curves and summary metrics (mean/median P&L, drawdown, VaR/CVaR, Sharpe/Sortino).
- **FR-004**: System MUST run an option strategy over the same paths using an option pricer (Black-Scholes with configurable IV) and produce analogous metrics.
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
- **FR-006**: System MUST support feature/indicator injection (e.g., SMA/RSI via pandas-ta) without modifying strategy engine code.
- **FR-007**: System MUST support parameter grid/batch execution with parallelization and return an objective score per configuration.
- **FR-008**: System MUST emit artifacts per run (e.g., metrics JSON/CSV and optional plots/HTML reports) to a run output directory.
- **FR-009**: System MUST allow selection of data source and option pricer via configuration rather than code changes.
- **FR-010**: System MUST handle missing data gracefully with clear warnings and deterministic fallbacks (e.g., drop NaNs, fail-fast if below minimum length).
- **FR-011**: System MUST provide stubs for Schwab API data and advanced pricers while defaulting to yfinance + BS so the MVP runs locally.
- **FR-012**: System MUST allow deterministic seeding of random number generation for reproducible simulations.
- **FR-013**: System SHOULD cap Monte Carlo sampling based on estimated footprint `n_paths * n_steps * 8 bytes` and refuse in-memory runs above **25% of available RAM**, automatically switching to memmap/npz and emitting a warning.  
- **FR-014**: System SHOULD support macro/alternative series alignment (e.g., interpolation to bar frequency) with explicit tolerances: maximum forward/backfill gap of **3× bar interval** and warning when alignment error exceeds tolerance.  
- **FR-015**: System SHOULD document performance targets (paths × steps per second) for the CPU-only VPS.
- **FR-016**: System MUST support Black–Scholes pricing with per-strike implied volatility as the default option pricer, and MUST allow swapping in more advanced pricers (e.g., Heston/QuantLib) via configuration without changes to strategy or backtest code.
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
  - The system MUST cap concurrent workers (e.g., max_workers ≤ 6 on an 8-core VPS) and abort the run (with structured error) when pre-run estimates exceed thresholds; if estimates are within limits but runtime exceeds, emit escalating warnings and stop remaining jobs.
- **FR-019**: System MUST prevent replay when underlying historical data version has changed since `run_meta` (unless an explicit `--force-replay`/`allow_data_drift` flag is provided), and must record the drift status in the replay output.
- **FR-020**: System MUST enforce distribution parameter validation (per-model bounds, heavy-tail sanity checks) and fit convergence controls (max iterations, failure fallback to Laplace) with structured errors when validation fails.
- **FR-021**: System MUST ensure reproducibility across seeded runs with numeric tolerance of ±1e-10 for path-level values; captures library versions and seed in `run_meta` and applies seeding to all MC operations (including conditional sampling).
- **FR-022**: System MUST detect numeric overflow/underflow or non-positive prices during log-return → price transforms and abort with a structured error; paths producing NaN/inf or ≤0 prices SHALL be rejected (no silent clipping).
- **FR-023**: System MUST apply a deterministic memory footprint estimator (`n_paths × n_steps × 8 bytes` plus 10% overhead) before MC execution and enforce storage policy selection based on the estimator (in-memory <25% RAM; otherwise memmap/npz) with user-visible warnings.
- **FR-024**: System MUST define configuration precedence (CLI flags > environment variables > YAML files) and log any overrides applied per run.
- **FR-025**: System MUST fail fast on invalid or incompatible configuration combinations (e.g., pricer not supported for asset type) and MUST define defaults for all optional parameters in configuration schemas.
- **FR-026**: System MUST log component swap events (data source, distribution, pricer, selector) with timestamp, run_id, and prior→new values for auditability.
- **FR-027**: System MUST validate Parquet schemas on load (column names, dtypes, index type) and refuse runs on detected drift unless a compatibility rule is defined.
- **FR-028**: System MUST compute and store data version fingerprints (e.g., SHA256 of Parquet payload + schema hash) in `run_meta` to support drift detection (per FR-019).
- **FR-029**: System MUST enforce missing-data tolerances: fail/run warning when continuous gaps exceed 3× bar interval or total missing bars exceed 1% of window unless an explicit imputation rule is configured.
- **FR-030**: System MUST guarantee atomic, append-only writes of `run_meta.json` and artifacts; metadata is immutable after run completion.
- **FR-031**: Storage policy for Monte Carlo datasets SHALL remain non-persistent by default (DM-009), but persistence is permitted when explicitly requested (replay) or required for memmap fallback; `run_meta` MUST record when persistence is used.

# **Candidate Selection Functional Requirements**
- **FR-CAND-001:** System SHALL implement a `CandidateSelector` abstraction that produces candidate timestamps based solely on information available at time t.
- **FR-CAND-002:** System SHALL construct “candidate episodes” (symbol, t0, horizon) for all historical timestamps where the selector fires.
- **FR-CAND-003:** System SHALL support “conditional backtests,” where strategies are evaluated only on candidate episodes and performance metrics are computed at the episode level.
- **FR-CAND-004:** System SHALL support “conditional Monte Carlo” that generates simulated price paths conditioned on the current state being similar to historical candidate episodes.
- **FR-CAND-005:** System SHOULD support at least one non-parametric episode-resampling method and one parametric state-conditioned return model.
- **FR-CAND-006:** System MUST ship at least one default selector (gap/volume spike rule) for MVP and allow selector swapping via configuration without code changes.

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
- **SC-003**: Grid runs produce ranked configurations with objective scores and complete without data races or corrupted outputs.  
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

## Clarifications & Anti-Requirements

- **Conditional Backtesting vs Conditional Monte Carlo**: Conditional backtesting replays historical episodes where a selector fired; conditional MC resamples/parameterizes returns conditioned on the same state and then simulates synthetic paths. Both must share selector definitions but produce different artifacts (historical metrics vs simulated distributions).
- **Anti-requirements**: MVP will not execute live trades, will not require GPUs, and will not store brokerage credentials in-repo; all runs are research-only.

## Glossary (shared across plan/data-model)

- **candidate**: A symbol + timestamp emitted by a selector rule.
- **candidate episode**: `(symbol, t0, horizon, state_features)` slice derived from a candidate for evaluation.
- **conditional backtest**: Historical replay limited to candidate episodes.
- **conditional MC**: Monte Carlo paths generated from state-conditioned returns/bootstraps matching candidate state.
- **unconditional MC**: Monte Carlo paths from a model fitted on the full chosen window without conditioning.
- **run**: A single execution of compare/grid/conditional producing artifacts under `runs/<run_id>/`.
- **grid**: A batch of strategy configurations evaluated in parallel.
