# Feature Specification: Backtesting & Strategy Spec Authoring

**Feature Branch**: `001-create-specs`  
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
2. **Given** the user provides a different distribution choice (e.g., student_t), **When** the CLI runs, **Then** the simulation uses that distribution and completes without errors, updating outputs accordingly.

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
- **FR-005**: System MUST expose a CLI to execute the stock-vs-option comparison with configurable symbol, date range, distribution, paths, steps, and strategy parameters.
- **FR-006**: System MUST support feature/indicator injection (e.g., SMA/RSI via pandas-ta) without modifying strategy engine code.
- **FR-007**: System MUST support parameter grid/batch execution with parallelization and return an objective score per configuration.
- **FR-008**: System MUST emit artifacts per run (e.g., metrics JSON/CSV and optional plots/HTML reports) to a run output directory.
- **FR-009**: System MUST allow selection of data source and option pricer via configuration rather than code changes.
- **FR-010**: System MUST handle missing data gracefully with clear warnings and deterministic fallbacks (e.g., drop NaNs, fail-fast if below minimum length).
- **FR-011**: System MUST provide stubs for Schwab API data and advanced pricers while defaulting to yfinance + BS so the MVP runs locally.
- **FR-012**: System MUST allow deterministic seeding of random number generation for reproducible simulations.
- **FR-013**: System SHOULD cap Monte Carlo sampling to avoid memory exhaustion and provide user feedback when limits are exceeded.  
- **FR-014**: System SHOULD support macro/alternative series alignment (e.g., interpolation to bar frequency) for use as features.  
- **FR-015**: System SHOULD document performance targets (paths × steps per second) for the CPU-only VPS.

*Clarifications (max 3):*
- **FR-016**: System MUST price options using [NEEDS CLARIFICATION: preferred pricer beyond Black-Scholes? e.g., per-strike IV surface vs Heston/QuantLib baseline].  
- **FR-017**: System MUST source historical data from [NEEDS CLARIFICATION: default provider priority—yfinance vs Schwab API?]  
- **FR-018**: System MUST define acceptable run time/resource limits for grid jobs on the VPS [NEEDS CLARIFICATION: target limits].

# **Candidate Selection Functional Requirements**
- **FR-CAND-001:** System SHALL implement a `CandidateSelector` abstraction that produces candidate timestamps based solely on information available at time t.
- **FR-CAND-002:** System SHALL construct “candidate episodes” (symbol, t0, horizon) for all historical timestamps where the selector fires.
- **FR-CAND-003:** System SHALL support “conditional backtests,” where strategies are evaluated only on candidate episodes and performance metrics are computed at the episode level.
- **FR-CAND-004:** System SHALL support “conditional Monte Carlo” that generates simulated price paths conditioned on the current state being similar to historical candidate episodes.
- **FR-CAND-005:** System SHOULD support at least one non-parametric episode-resampling method and one parametric state-conditioned return model.

# **Data Management Functional Requirements (DM-Series)**

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
Here are the additional success criteria I would add, in the same style and level as what you already have.

### Monte Carlo & Conditional Backtesting Success Criteria

* **SC-007**: For a fixed symbol and configuration, Monte Carlo runs with the same seed and parameters produce identical path-level outcomes and episode-level metrics across at least 3 repeated runs.
* **SC-008**: The system can run a conditional backtest over at least 100 historical candidate episodes for a single symbol and complete within the documented time budget (to be defined in FR-0xx).
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
