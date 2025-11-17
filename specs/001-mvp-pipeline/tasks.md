# Tasks: Backtesting & Strategy Spec Authoring (Quant Scenario Engine)

**Input**: Design documents from `/specs/001-mvp-pipeline/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/openapi.yaml

**Testing Strategy**: Per Constitution Principle XV, ≥80% line coverage required. Tests MUST be written during implementation tasks, not deferred. Each module PR requires passing tests and coverage verification before merge. See "Testing Strategy" section below for detailed guidance.

**DM-Series Coverage**: Added T020d-g to address gaps identified in Specification_Analysis_Report.md (DM-001/002/003 resolution tiers, DM-005 feature separation, DM-007 symbol tiers, DM-013 versioning). Additional tasks T020h-m address caching, corporate actions, stationarity checks, and component logging per updated spec/plan.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story per constitution-driven development (Principle XVII).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

Per plan.md structure:
- **Source**: quant-scenario-engine/{features,models,runs,schema,interfaces,backtesting/{distributions,mc},optimizer,pricing,strategies,simulation,cli,config,data,utils}
- **Tests**: `tests/` (unit, integration, contract)
- **Artifacts**: `runs/`, `data/` (historical, features)
- **Package name**: `backtesting` (aligned with plan.md and quickstart.md)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create project directory structure per plan.md (quant-scenario-engine/{data,runs,schema,features,models,interfaces,backtesting/{distributions,mc},optimizer,pricing,strategies,simulation,cli,config,utils}, tests/{unit,integration,contract})
- [X] T002 Initialize Python 3.11 project with pyproject.toml and core dependencies (numpy, pandas, scipy, numba, statsmodels, arch, pandas-ta, quantstats, plotly, yfinance, typer, pytest)
- [X] T003 [P] Configure ruff linting and black formatting with settings in pyproject.toml
- [X] T004 [P] Setup mypy type checking configuration in pyproject.toml
- [X] T005 [P] Create .gitignore for Python (venv, __pycache__, *.pyc, .pytest_cache, runs/, data/historical/, data/features/)
- [X] T006 [P] Create README.md with quickstart instructions per plan.md onboarding section
- [X] T007 Create requirements-dev.txt with testing/dev dependencies (pytest, pytest-cov, pytest-mock, hypothesis, ruff, black, mypy) and setup pytest.ini with coverage config
- [X] T007a [P] Define performance budget benchmarks and add validation script (measure data load, fit, MC throughput, strategy eval) per FR-015 and FR-040 (scripts/benchmarks/perf_check.py)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T008 Create base exception classes in quant-scenario-engine/exceptions.py (DataSourceError, DistributionFitError, ResourceLimitError, PricingError, EpisodeGenerationError, ConfigError, ConfigValidationError, ConfigConflictError, SchemaError, InsufficientDataError, TimestampAnomalyError, BankruptcyError, DependencyError per FR-041/042)
- [X] T009 [P] Implement logging configuration in quant-scenario-engine/utils/logging.py with structured JSON output per FR-039 (JSONFormatter with run_id, component, duration_ms, symbol, config_index tracking)
- [X] T010 [P] Create RunConfig dataclass in quant-scenario-engine/schema/run_config.py per data-model.md with validation (add covariance_estimator enum: sample/ledoit_wolf/shrinkage_delta, var_method enum: parametric/historical, lookback_window int per FR-034)
- [X] T011 [P] Create OptionSpec dataclass in quant-scenario-engine/models/options.py per data-model.md FR-016 (add iv_source enum: yfinance/realized_vol/config_default, early_exercise flag for American options)
- [X] T012 [P] Create StrategyParams dataclass in quant-scenario-engine/schema/strategy.py per data-model.md (add fees float ≥0 default 0.0005, slippage float ≥0 default 0.65 per FR-041)
- [X] T013 Create ReturnDistribution abstract base class in quant-scenario-engine/interfaces/distribution.py per data-model.md (add estimator enum: mle/gmm, loglik/aic/bic floats, fit_status enum: success/warn/fail, min_samples int ≥60 per FR-020/037)
- [X] T014 Create Strategy abstract base class in quant-scenario-engine/interfaces/strategy.py per data-model.md
- [X] T015 Create OptionPricer abstract base class in quant-scenario-engine/interfaces/pricing.py per data-model.md
- [X] T016 Create CandidateSelector abstract base class in quant-scenario-engine/interfaces/candidate_selector.py per FR-CAND-001
- [X] T017 [P] Implement resource estimator in quant-scenario-engine/utils/resources.py (estimate memory footprint for n_paths × n_steps, enforce FR-013/FR-018 thresholds: estimated_gb = n_paths*n_steps*8*1.1/1e9, thresholds <25% in-memory, ≥25% memmap, ≥50% abort)
- [X] T018 [P] Create run metadata schema in quant-scenario-engine/schema/run_meta.py with JSON serialization per FR-008/FR-019/FR-030/FR-034 (include covariance_estimator, var_method, lookback_window, storage_policy, drift_status, iv_source, parameter_stability fields)
- [X] T018a [P] Capture reproducibility context in run_meta (seeds, library versions, environment: CPU/RAM/OS, git commit SHA) per FR-021 and CHK139-CHK141
- [X] T019 [P] Setup Parquet data directory structure in data/{historical,features}/{interval}/{symbol}.parquet per DM-006
- [X] T020 Create storage policy selector in backtesting/mc/storage.py (decide memory vs npz vs memmap based on RAM threshold per DM-008/DM-010/DM-011, add estimated_gb tracking per plan.md)
- [X] T020a [P] Add Parquet schema validation and fingerprint computation in quant-scenario-engine/data/validation.py (DM-015, DM-016, FR-027, FR-028: validate OHLCV columns, compute content hash for drift detection)
- [X] T020b [P] Add missing-data tolerance enforcement (3× interval continuous gap, 1% total) and configurable imputation rules in data validation (FR-029, DM-017)
- [X] T020c [P] Ensure run_meta writes are atomic/immutable and record persistence usage (FR-030, FR-031, DM-018: atomic file writes, immutability validation, storage policy tracking)
- [X] T020d [P] Document resolution tier strategy (daily for distribution fitting, 5-min for backtesting, 1-min for live) and implement tier selection logic in data validation (DM-001, DM-002, DM-003)
- [X] T020e [P] Enforce raw OHLCV vs derived features separation in Parquet storage (data/historical/{interval}/ vs data/features/{interval}/) per DM-005
- [X] T020f [P] Add Parquet historical data versioning support (_v2, _v3 suffixes) and retention policy documentation per DM-013
- [X] T020h Create DataLoader class in quant-scenario-engine/data/data_loader.py with cache-aside pattern per FR-085/086 (load_ohlcv with cache check, staleness detection per interval, incremental updates, metadata writes per plan.md lines 100-268)
- [X] T020i [P] Implement corporate action detection in DataLoader per FR-085 (fetch overlapping bar, compare cached_last_close vs fresh_close, trigger full refresh on >1% divergence with warning per plan.md lines 169-195)
- [X] T020j [P] Add force_refresh and allow_stale_cache flag support in DataLoader per FR-085/086 (force_refresh bypasses cache, allow_stale_cache uses stale on fetch failure per plan.md lines 132-136, 206-219)
- [X] T020g [P] Create Universe/Watchlist/Live Set configuration schema (symbol tiers: Universe=daily+5min, Watchlist=daily+5min+features, Live=all resolutions) per DM-007

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Testing Strategy (Applied Throughout All Phases)

**Constitutional Mandate**: Principle XV requires ≥80% line coverage with comprehensive testing discipline.

### Test-During-Implementation Pattern

**CRITICAL**: Tests are NOT deferred to Phase 12. Write tests during each implementation task:

```
Implementation Task Flow:
1. Implement module/function (e.g., T021 YFinanceDataSource)
2. Write corresponding tests (tests/unit/data/test_yfinance.py)
3. Run coverage: pytest --cov=backtesting.data.yfinance --cov-report=term-missing
4. Verify ≥80% coverage achieved
5. Fix coverage gaps
6. Mark task complete only when tests pass + coverage met
```

### Test Organization by Directory

```
tests/
├── unit/                    # Fast, isolated, mocked external I/O
│   ├── data/
│   │   ├── test_yfinance.py
│   │   ├── test_schwab_stub.py
│   │   ├── test_validation.py
│   │   └── test_data_loader.py
│   ├── distributions/
│   │   ├── test_laplace.py
│   │   ├── test_student_t.py
│   │   ├── test_validation.py
│   │   ├── test_stationarity.py
│   │   └── test_ar_detection.py
│   ├── mc/
│   │   ├── test_generator.py
│   │   └── test_storage.py
│   ├── pricing/
│   │   ├── test_black_scholes.py
│   │   └── test_py_vollib.py
│   ├── strategies/
│   │   ├── test_stock_basic.py
│   │   └── test_option_call.py
│   └── simulation/
│       ├── test_simulator.py
│       ├── test_metrics.py
│       └── test_compare.py
├── integration/             # End-to-end workflows, real I/O
│   ├── test_data_pipeline.py
│   ├── test_mc_end_to_end.py
│   ├── test_simulation_workflows.py
│   └── cli/
│       ├── test_compare_e2e.py
│       ├── test_grid_e2e.py
│       └── test_screen_e2e.py
└── contract/                # Schema/interface compliance
    ├── test_cli_schemas.py      # Validate against openapi.yaml
    ├── test_interfaces.py       # All implementations satisfy ABCs
    └── test_run_meta_schema.py  # Validate run_meta.json structure
```

### Test Requirements by Type

**Unit Tests** (target: 85% of test suite):
- Mock all external I/O (HTTP, file system, random number generation)
- Test single function/class in isolation
- Fast execution (<1s per test, <10s for entire unit suite)
- Focus on edge cases, boundary conditions, error paths
- Use parametrize for combinatorial cases

**Integration Tests** (target: 10% of test suite):
- Real file I/O with temp directories
- Full workflows from data load → artifacts
- Slower execution (acceptable <10s per test)
- Test component interactions and data flow
- Verify artifact generation and persistence

**Contract Tests** (target: 5% of test suite):
- Validate CLI inputs against contracts/openapi.yaml using schemathesis
- Verify all interface implementations satisfy ABC contracts
- Validate run_meta.json against defined schema
- Test configuration precedence (CLI > ENV > YAML)

### Coverage Requirements by Module Type

**Critical Paths** (100% coverage required):
- Monte Carlo generation (backtesting/mc/generator.py)
- Option pricing calculations (pricing/black_scholes.py)
- P&L simulation (simulation/simulator.py)
- Metrics calculations (simulation/metrics.py)

**Core Logic** (≥90% coverage required):
- Distributions (fit/sample methods)
- Strategies (signal generation)
- Data validation (schema checks, missing data)
- Storage policy (memory threshold decisions)
- DataLoader (caching, corporate actions)

**Infrastructure** (≥80% coverage required):
- CLI commands and validation
- Config loaders
- Exception handling
- Logging and artifacts

**Utilities** (≥70% coverage acceptable):
- Plotting/reporting (optional features)
- Helper functions
- Constants and enums

### Property-Based Testing Recommendations

Use `hypothesis` library for distributions and MC generation:

```python
# tests/unit/distributions/test_laplace.py
from hypothesis import given, strategies as st

@given(
    returns=st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=60, max_size=252),
    seed=st.integers(min_value=0, max_value=2**31)
)
def test_laplace_reproducibility(returns, seed):
    """Same seed produces identical samples."""
    dist = LaplaceDistribution()
    dist.fit(np.array(returns))
    sample1 = dist.sample(n_paths=100, n_steps=60, seed=seed)
    sample2 = dist.sample(n_paths=100, n_steps=60, seed=seed)
    np.testing.assert_array_equal(sample1, sample2)
```

### Testing Phases Mapped to Implementation

**Phase 2 (Foundation) - Test Setup**:
- T007: Add pytest, pytest-cov, hypothesis to requirements-dev.txt
- Create tests/{unit,integration,contract}/ directory structure
- Setup pytest.ini with coverage config
- Write tests for all interfaces (T013-T016) - 100% coverage required

**Phase 3 (US1) - Core Test Suite**:
- Unit tests for every T021-T051 implementation task
- Integration tests for data pipeline, MC generation, simulation workflows
- Contract tests for compare CLI against openapi.yaml
- Property-based tests for distributions

**Phase 4-10 (US2-US8) - Feature Tests**:
- Unit tests written during each story implementation
- Integration tests for new workflows (grid, screening, conditional)
- Contract tests for new CLI commands

**Phase 12 (Polish) - Coverage Gap Analysis**:
- T130: Run coverage analysis and fill gaps to ≥80%
- Contract test suite completion for all CLI commands
- Performance regression tests for SC-001/SC-002 budgets

### Merge Criteria (Constitutional Gate)

Before any PR merge:
1. ✅ All tests pass
2. ✅ New code achieves ≥80% line coverage (use `--cov-report=term-missing` to verify)
3. ✅ No coverage regression from prior modules
4. ✅ Contract tests validate against openapi.yaml (for CLI changes)
5. ✅ Linting passes (ruff, black, mypy)
6. ✅ Type hints present on all public functions

### Test Execution Commands

```bash
# Run all tests with coverage
pytest tests/ --cov=backtesting --cov=quant-scenario-engine --cov-report=term-missing --cov-report=html

# Run only unit tests (fast feedback)
pytest tests/unit/ -v

# Run integration tests (slower)
pytest tests/integration/ -v

# Run contract tests
pytest tests/contract/ -v

# Run specific module tests during development
pytest tests/unit/distributions/test_laplace.py -v --cov=backtesting.distributions.laplace

# Run property-based tests with more examples
pytest tests/unit/ -v --hypothesis-show-statistics

# Generate HTML coverage report
pytest --cov=backtesting --cov-report=html
# View: open htmlcov/index.html
```

### Test Fixtures and Helpers

Create shared fixtures in `tests/conftest.py`:

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_ohlcv():
    """Returns realistic OHLCV DataFrame for testing."""
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    return pd.DataFrame({
        'open': np.random.uniform(100, 110, 252),
        'high': np.random.uniform(110, 120, 252),
        'low': np.random.uniform(90, 100, 252),
        'close': np.random.uniform(100, 110, 252),
        'volume': np.random.randint(1e6, 10e6, 252),
    }, index=dates)

@pytest.fixture
def temp_data_dir(tmp_path):
    """Returns temporary directory for artifact testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "historical").mkdir()
    (data_dir / "features").mkdir()
    return data_dir

@pytest.fixture
def mock_run_config():
    """Returns valid RunConfig for testing."""
    return RunConfig(
        n_paths=100,
        n_steps=60,
        seed=42,
        distribution_model='laplace',
        data_source='yfinance'
    )
```

---

## Phase 3: User Story 1 - Run stock vs option Monte Carlo comparison (Priority: P1) 🎯 MVP

**Goal**: Execute CLI to generate Monte Carlo price paths, apply paired stock and option strategies, receive equity curves plus summary metrics

**Independent Test**: Execute CLI with sample symbol (e.g., AAPL) and verify outputs include both stock and option metrics plus saved report artifacts per spec.md acceptance scenarios

### Implementation for User Story 1

#### Data Layer (US1)

- [X] T021 [P] [US1] Implement YFinanceDataSource adapter in quant-scenario-engine/data/yfinance.py with fetch(symbol, start, end, interval) and exponential backoff retries per FR-001/086 (3 retries with 1s/2s/4s backoff)
- [X] T021a [US1] Integrate YFinanceDataSource with DataLoader in quant-scenario-engine/data/data_loader.py per T020h (wire _fetch_from_source to use YFinanceDataSource, handle retries and errors)
- [X] T022 [P] [US1] Create SchwabDataSourceStub in quant-scenario-engine/data/schwab_stub.py with matching interface per FR-011/FR-017
- [X] T023 [P] [US1] Implement data validation in quant-scenario-engine/data/validation.py (check OHLCV columns, handle missing data per FR-010/029, validate timestamps for monotonicity and future dates per FR-057)
- [X] T023a [US1] Implement data fingerprinting in quant-scenario-engine/data/validation.py per T020a (compute content hash from OHLCV data, store in metadata, compare on load for drift detection per FR-019/028)
- [X] T024 [US1] Create DataSource factory in quant-scenario-engine/data/factory.py to select yfinance vs schwab per config per FR-009/043 (extend FactoryBase from T020k, log component swaps)

#### Distribution Models (US1)

- [X] T025 [P] [US1] Implement LaplaceDistribution in backtesting/distributions/laplace.py with fit() and sample() per FR-002 (default model, include stationarity preflight via T020l, capture estimator=mle, loglik, aic, bic per T013)
- [X] T026 [P] [US1] Implement StudentTDistribution in backtesting/distributions/student_t.py with fit() and sample() per FR-002 (include stationarity preflight via T020l, AR detection via T020m, capture estimator metadata per T013)
- [X] T027 [P] [US1] Create distribution factory in backtesting/distributions/factory.py to select model by config (laplace default, student_t optional, extend FactoryBase from T020k)
- [X] T028 [US1] Implement distribution parameter validation in backtesting/distributions/validation.py per data-model.md (len(returns) >= min_samples check, seed validation)
- [X] T028a [US1] Add parameter bounds checking and implausible parameter rejection in backtesting/distributions/validation.py per FR-020/FR-037 (heavy-tail threshold: excess kurtosis ≥1.0; warn <0.5; reject implausible params; trigger fallback warnings)
- [X] T028b [US1] Implement convergence limits and fallback model logic in backtesting/distributions/validation.py per FR-020/FR-037 (detect fit failures, log warnings, fallback to simpler model with structured errors and heavy_tail_warning metadata)

#### Monte Carlo Engine (US1)

- [X] T029 [US1] Implement generate_price_paths() in backtesting/mc/generator.py (s0, distribution, n_paths, n_steps, seed) per plan.md reference interface (log-return → price transform with overflow/non-positive rejection per FR-022)
- [X] T030 [P] [US1] Implement in-memory path storage in backtesting/mc/storage.py for footprint <25% RAM per DM-008
- [X] T030a [P] [US1] Add estimated_gb tracking in PricePath dataclass per plan.md and data-model.md (compute n_paths*n_steps*8*1.1/1e9, store in PricePath.estimated_gb field)
- [X] T031 [P] [US1] Implement memmap/npz fallback in backtesting/mc/storage.py for footprint ≥25% RAM per DM-011/FR-013 (warn user on memmap activation, abort if ≥50% RAM)
- [X] T032 [US1] Add seeded random number generation with reproducibility guarantee per FR-012 (ensure all np.random calls use Generator with seed, validate deterministic output)

#### Option Pricing (US1)

- [X] T033 [US1] Implement BlackScholesPricer in quant-scenario-engine/pricing/black_scholes.py with price(spot, spec, ttm) per FR-016/FR-022 and plan.md reference (handle maturity vs horizon edge cases, ATM precision, invalid IV with structured errors)
- [X] T034 [P] [US1] Create OptionPricer factory in quant-scenario-engine/pricing/factory.py to select black_scholes (default) per FR-009/FR-016/FR-043 (extend FactoryBase from T020k, log pricer swaps)

#### Strategy Framework (US1)

- [X] T035 [US1] Create example StockBasicStrategy in quant-scenario-engine/strategies/stock_basic.py implementing Strategy interface per data-model.md (dual-SMA baseline per FR-006, apply fees/slippage defaults per FR-041)
- [X] T036 [US1] Create example OptionCallStrategy in quant-scenario-engine/strategies/option_call.py implementing Strategy interface with OptionSpec per data-model.md (emit option_spec in StrategySignals, handle early_exercise when pricer supports; FR-016/FR-041)
- [X] T037 [US1] Implement StrategySignals dataclass in quant-scenario-engine/schema/signals.py (signals_stock, signals_option, option_spec, features_used) per data-model.md and spec DM relationships

#### Simulation Engine (US1)

- [X] T038 [US1] Implement MarketSimulator class in quant-scenario-engine/simulation/simulator.py with simulate_stock() and simulate_option() per FR-003/FR-004/FR-050 and plan.md reference (handle bankruptcy paths, record bankruptcy_rate)
- [X] T039 [US1] Implement metrics calculator in quant-scenario-engine/simulation/metrics.py (mean/median P&L, drawdown, Sharpe/Sortino per FR-003/FR-034)
- [X] T039a [US1] Add VaR/CVaR calculation with parametric vs historical methods in quant-scenario-engine/simulation/metrics.py per FR-034 (support covariance_estimator: sample/ledoit_wolf/shrinkage_delta, var_method: parametric/historical, lookback_window, stability checks + metadata)
- [X] T040 [US1] Create SimulationRun orchestrator in quant-scenario-engine/simulation/run.py to coordinate MC, strategies, simulation per data-model.md (capture system_info: cpu_count/ram_gb/os, git_sha per T018a; FR-005)
- [X] T041 [US1] Implement run_compare() function in quant-scenario-engine/simulation/compare.py integrating all components for stock vs option comparison per FR-005
- [X] T041a [US1] Enforce CLI parameter validation against contracts/openapi.yaml for compare command per FR-033 (validate CompareRequest schema, fail-fast on invalid fields)

#### Artifacts & Persistence (US1)

- [X] T042 [P] [US1] Implement MetricsReport serialization in quant-scenario-engine/schema/metrics.py with JSON/CSV export per FR-008/FR-034 (include var_method, lookback, bankruptcy_rate, early_exercise_events fields per data-model.md)
- [X] T043 [P] [US1] Create run artifact writer in quant-scenario-engine/utils/artifacts.py (metrics, plots, run_meta.json) per FR-008/FR-030
- [X] T043a [P] [US1] Ensure run_meta/artifacts write atomically and include schema versions per FR-030/FR-031/FR-034 (atomic file writes, record persistence usage: storage_policy, covariance_estimator, var_method, drift_status, iv_source)
- [X] T044 [US1] Implement run_meta.json persistence in quant-scenario-engine/utils/run_meta.py with all required fields per FR-019/FR-034 and data-model.md example (seeds, versions, git SHA, system info, data fingerprints, storage policy, fallbacks, covariance/VaR metadata, parameter stability)

#### CLI (US1)

- [X] T045 [US1] Create Typer CLI entrypoint in quant-scenario-engine/cli/main.py with compare command per FR-005
- [X] T046 [US1] Implement config validation in quant-scenario-engine/cli/validation.py enforcing contracts/openapi.yaml CompareRequest schema (FR-033)
- [X] T047 [US1] Wire compare CLI to run_compare() in quant-scenario-engine/cli/commands/compare.py with proper error handling per plan.md error policies (map exceptions to exit codes per plan.md lines 522-556; FR-041/FR-042)
- [X] T048 [US1] Add CLI argument parsing for symbol, start, end, distribution, paths, steps, seed, data_source, option_pricer, strategies per contracts/openapi.yaml (FR-024/FR-033)
- [X] T049 [US1] Implement progress reporting for long-running operations per FR-039 (emit progress every 10 configs or 1 minute per plan.md lines 558-565)
- [X] T050 [US1] Add structured error handling with component-specific exit codes per FR-041/FR-042 (ConfigValidationError→1, InsufficientDataError→2, DistributionFitError→3, ResourceLimitError→4, KeyboardInterrupt→130 per plan.md lines 522-556)
- [X] T051 [US1] Implement CLI config precedence (CLI > ENV > YAML) per FR-024/FR-026 and plan.md (document precedence, log overrides per T020k)

**Checkpoint**: User Story 1 MVP complete - can execute stock vs option comparison and produce artifacts

---

## Phase 4: User Story 4 - Stock screening via candidate selector (Priority: P1)

**Goal**: Scan universe of symbols, apply candidate-selection rule, receive ranked list of stocks for strategy evaluation

**Independent Test**: Configure universe of ≥100 symbols and simple candidate rule (e.g., top 20 daily gainers/losers with volume filters), run CLI, confirm ranked candidate list with metadata per spec.md acceptance scenarios

**Note**: Implementing US4 before US2/US3 as it's P1 priority and provides foundation for US5/US6 conditional logic

### Implementation for User Story 4

#### Feature Engineering (US4)

- [X] T052 [P] [US4] Implement technical indicator wrapper in quant-scenario-engine/features/indicators.py using pandas-ta (SMA, RSI, volume_z) per FR-006/FR-066 (pandas-ta fallback logging; reference T020l/T020m for stationarity-aware feature computation)
- [X] T053 [P] [US4] Create feature computation pipeline in quant-scenario-engine/features/pipeline.py to enrich OHLCV with indicators per FR-006 (keep derived features separate per FR-029/DM005)
- [X] T054 [US4] Implement gap percentage calculator in quant-scenario-engine/features/gap.py (open vs prev_close) for selector rules per FR-056/FR-029 (track extreme gaps, log/tolerances)

#### Candidate Selection (US4)

- [X] T055 [US4] Implement GapVolumeSelector in quant-scenario-engine/selectors/gap_volume.py extending CandidateSelector per FR-CAND-001/FR-CAND-006 (default MVP selector, gap+volume rules) and SC-049
- [X] T056 [US4] Implement selector.score() and selector.select() methods per data-model.md CandidateSelector interface (log warnings for missing features per FR-CAND-006, enforce min_episodes threshold per SC-020)
- [X] T057 [P] [US4] Create CandidateEpisode dataclass in quant-scenario-engine/schema/episode.py per data-model.md and FR-CAND-002 (symbol, horizon, score)
- [X] T058 [US4] Implement build_candidate_episodes() in quant-scenario-engine/selectors/episodes.py per data-model.md reference (validate horizon > 0, state_features captured, maturity_days >= horizon for option specs per FR-035 and SC-010)
- [X] T058a [US4] Document and enforce candidate episode construction rules across backtest and conditional MC per FR-035 (horizon validation, state_features tracking, episode metadata for run_meta/scenario logging)

#### Screening Engine (US4)

- [X] T059 [US4] Implement universe screening logic in quant-scenario-engine/simulation/screen.py to scan symbols and apply selector per FR-CAND-001/FR-033
- [X] T060 [US4] Add parallel symbol processing with max_workers cap per FR-018 concurrency model (ProcessPoolExecutor w/ worker clamping per plan.md lines 329-367)
- [X] T061 [US4] Implement candidate ranking and top_n filtering in quant-scenario-engine/simulation/screen.py per contracts/openapi.yaml ScreenRequest and FR-033
- [X] T062 [US4] Add error handling for missing/partial data per symbol per SC-020/FR-029 (log warnings, continue with available symbols, report failures in run_meta)

#### CLI (US4)

- [X] T063 [US4] Create screen command in quant-scenario-engine/cli/commands/screen.py with universe CSV or symbols list per contracts/openapi.yaml ScreenRequest and FR-033 CLI schema; reuse fetch path with Parquet caching/slicing per FR-CAND-006a
- [X] T064 [US4] Wire screen CLI to screening engine with validation per contracts/openapi.yaml ScreenResponse (FR-033); ensure symbol input path auto-downloads/extends Parquet and slices shorter windows from existing data
- [X] T065 [US4] Implement ScreenResponse serialization with candidate list (symbol, t0, state_features) per contracts/openapi.yaml (FR-033)

**Checkpoint**: User Story 4 complete - can screen universe and generate candidate lists

---

## Phase 5: User Story 5 - Conditional episode backtesting for candidates (Priority: P1)

**Goal**: Run conditional backtest evaluating strategy only on historical episodes where candidate selector fired, receive episode-level and aggregate metrics

**Independent Test**: Define candidate selector (e.g., large gap + volume spike), build ≥50 historical episodes for test symbol, run conditional backtest, verify episode-level and aggregate metrics for stock and option per spec.md acceptance scenarios

### Implementation for User Story 5

#### Strategy-Symbol Screening Core (US5)

**Mode A (US4 preserved)**: Already implemented in Phase 4 - candidate selection without strategy evaluation

**Mode B (Unconditional Strategy Screening)**:
- [X] T066a [US5] Implement run_strategy_screen() in quant-scenario-engine/simulation/screen.py to evaluate strategy across universe of symbols on all historical data
- [X] T066b [US5] Add symbol-level metrics aggregation in quant-scenario-engine/simulation/screen.py per US5 acceptance scenario 1 (sharpe, mean_pnl, sortino, max_drawdown per symbol)
- [X] T066c [US5] Implement ranking and top-N selection logic with configurable rank_by metric in screen.py
- [X] T066d [US5] Create ScreenResult model in quant-scenario-engine/models/screen.py with ranked symbols, metrics, and metadata

**Mode C (Conditional Strategy Screening)**:
- [X] T067 [US5] Implement conditional episode filtering in quant-scenario-engine/simulation/conditional.py to extract candidate-only windows per FR-CAND-003 (conditional backtest episodes)
- [X] T068 [US5] Extend run_strategy_screen() to support optional conditional filtering using selector from US4 per US5 acceptance scenario 2
- [X] T069 [US5] Extend MarketSimulator to support episode-level P&L tracking in quant-scenario-engine/simulation/simulator.py per FR-CAND-003/FR-050 (bankruptcy tracking for episodes)
- [X] T070 [US5] Implement episode-level metrics aggregation in quant-scenario-engine/simulation/metrics.py per SC-009/FR-034 (per-episode P&L, aggregate stats, var_method metadata)
- [X] T071 [US5] Add unconditional vs conditional metrics comparison reporting per SC-011/SC-009 (US5 acceptance scenario 6 requirements)
- [X] T072 [US5] Implement low-confidence flagging for symbols with <10 episodes per US5 acceptance scenario 4

#### Selector Definition (US5 Mode C)

- [X] T073 [US5] Define YAML schema for conditional selector files in selectors/ directory with fields: name, description, parameters, logic
- [X] T074 [US5] Implement selector file parser in quant-scenario-engine/selectors/loader.py to load YAML definitions
- [X] T075 [US5] Create example selector files: selectors/gap_down_volume_spike.yaml, selectors/breakout_momentum.yaml

#### CLI (US5 Three Modes)

- [X] T076 [US5] Extend screen command in quant-scenario-engine/cli/commands/screen.py to detect mode from flags (no --strategy = Mode A, --strategy only = Mode B, --strategy + --conditional-file = Mode C)
- [X] T077 [US5] Add --strategy, --rank-by, --conditional-file parameters to screen CLI per contracts/openapi.yaml ScreenRequest schema
- [X] T078 [US5] Wire Mode B (unconditional) to run_strategy_screen() with proper artifact generation per FR-008
- [X] T079 [US5] Wire Mode C (conditional) to run_strategy_screen(conditional=True) with proper artifact generation per FR-008/FR-CAND-003
- [X] T080 [US5] Implement min_episodes validation and fallback warning per FR-CAND-006/SC-008 (warn when <10 episodes per symbol)
- [X] T081 [US5] Add result file naming conventions to distinguish unconditional vs conditional runs per US5 acceptance scenario 6 (e.g., screen_results_unconditional.json vs screen_results_conditional.json)

**Checkpoint**: User Story 5 complete - can run three-mode screening: candidate selection (US4), unconditional strategy-symbol ranking, and conditional strategy-symbol ranking

---

## Phase 6: User Story 2 - Parameter grid exploration (Priority: P2)

**Goal**: Define parameter grid, run batch evaluations in parallel, receive aggregate metrics (mean, CVaR, drawdown) per configuration

**Independent Test**: Supply small grid (≥3 configs) and confirm per-config metrics are produced and ranked by objective function per spec.md acceptance scenarios

### Implementation for User Story 2

#### Grid Execution (US2)

- [ ] T074 [US2] Implement grid parameter expansion in quant-scenario-engine/simulation/grid.py to generate StrategyParams combinations per contracts/openapi.yaml StrategyGridConfig
- [ ] T075 [US2] Create grid runner with ProcessPoolExecutor in quant-scenario-engine/simulation/grid.py using max_workers from config per plan.md concurrency model (lines 329-367: clamp workers, collect partial results, handle failures per FR-062/081)
- [ ] T076 [US2] Implement per-config metrics collection and aggregation in quant-scenario-engine/simulation/grid.py
- [ ] T077 [US2] Add objective function scoring and ranking in quant-scenario-engine/simulation/metrics.py per FR-007 and spec.md US2 acceptance scenario 1
- [ ] T078 [US2] Implement resource limit preflight check in quant-scenario-engine/simulation/grid.py per FR-018 (estimate time/memory for N configs using T017 estimator, abort if > thresholds, emit warnings at 50%/90% per plan.md lines 379-387)
- [ ] T079 [US2] Add race condition prevention and output integrity validation per spec.md US2 acceptance scenario 2 (atomic writes per T043a, validate result completeness)

#### CLI (US2)

- [ ] T080 [US2] Create grid command in quant-scenario-engine/cli/commands/grid.py with grid parameter per contracts/openapi.yaml GridRequest
- [ ] T081 [US2] Wire grid CLI to grid runner with progress tracking and warnings per FR-018 (emit progress, time budget warnings per T049)
- [ ] T082 [US2] Implement GridResponse serialization with ranked configurations per contracts/openapi.yaml

**Checkpoint**: User Story 2 complete - can execute parameter grids with ranking

---

## Phase 7: User Story 6 - Conditional Monte Carlo from candidate states (Priority: P2)

**Goal**: Generate Monte Carlo paths conditioned on current candidate state, run strategies over conditional paths, review conditional risk/reward distribution

**Independent Test**: For symbol and candidate state, run conditional Monte Carlo CLI (e.g., 1,000 paths × 60 steps) and verify state-conditioned model usage and summary distributions per spec.md acceptance scenarios

### Implementation for User Story 6

#### Conditional Sampling (US6)

- [ ] T083 [US6] Implement episode bootstrapping sampler in backtesting/distributions/episode_bootstrap.py for non-parametric conditional MC per FR-CAND-005/SC-007
- [ ] T084 [P] [US6] Implement state-conditioned distribution refit in backtesting/distributions/conditional.py for parametric conditional MC per FR-CAND-005 (reference T020l/T020m for stationarity/AR checks on conditional samples)
- [ ] T085 [US6] Create conditional MC method selector in backtesting/mc/conditional.py (bootstrap vs refit) with fallback logic per FR-CAND-005/FR-036/FR-038 (try bootstrap → parametric refit → unconditional, log method selection)
- [ ] T086 [US6] Implement minimum episode threshold check with warning and fallback per spec.md US6 acceptance scenario 3/FR-CAND-006 (warn when <30 episodes, trigger fallback)
- [ ] T087 [US6] Create run_conditional_mc() in quant-scenario-engine/simulation/conditional_mc.py integrating conditional sampling with strategies per FR-CAND-004/FR-036
- [ ] T087a [US6] Log method selection and fallbacks (bootstrap → refit → unconditional) in run_meta/logs per FR-036/FR-038/FR-CAND-004 (record in run_meta.json, emit structured logs)

#### CLI (US6)

- [ ] T088 [US6] Extend conditional CLI to support monte_carlo mode in quant-scenario-engine/cli/commands/conditional.py per contracts/openapi.yaml ConditionalRequest (FR-033)
- [ ] T089 [US6] Add state_features input parsing and validation per contracts/openapi.yaml CandidateEpisode (FR-CAND-002)
- [ ] T090 [US6] Wire conditional MC CLI to run_conditional_mc() with reproducibility guarantee per spec.md US6 acceptance scenario 2/SC-007 (ensure seed propagation, deterministic output)

**Checkpoint**: User Story 6 complete - can generate conditional Monte Carlo paths from candidate states

---

## Phase 8: User Story 7 - Config-driven component swapping (Priority: P2)

**Goal**: Change configuration (YAML/env/CLI flags) to swap data sources, distribution models, option pricers without code edits

**Independent Test**: Run three successive CLI runs varying only configuration (e.g., yfinance+Normal+BS vs Schwab+Student-T+BS vs Schwab+Student-T+AltPricer) and confirm wiring changes per spec.md acceptance scenarios

### Implementation for User Story 7

#### Configuration Management (US7)

- [ ] T091 [US7] Create YAML configuration schema in quant-scenario-engine/config/schema.py matching contracts/openapi.yaml request schemas (FR-009/FR-033)
- [ ] T092 [P] [US7] Implement YAML config loader in quant-scenario-engine/config/loader.py with validation per FR-033/FR-009
- [ ] T093 [P] [US7] Add environment variable override support in quant-scenario-engine/config/env.py per plan.md onboarding (FR-024/FR-026 precedence)
- [ ] T094 [US7] Implement config validation with fail-fast on invalid values per spec.md US7 acceptance scenario 3 (structured ConfigValidationError per T008, FR-041 guidance)
- [ ] T095 [US7] Wire all factories (data, distribution, pricer) to read from unified config per FR-009/FR-043 (ensure T020k logging active on swaps)
- [ ] T095a [US7] Enforce configuration precedence (CLI > ENV > YAML) and log overrides per FR-024/FR-026 (document precedence in README, emit INFO logs on overrides per T020k)
- [ ] T095b [US7] Detect and block incompatible config combinations with structured errors per FR-025 (e.g., pricer not supported, conflicting resource limits, emit ConfigConflictError per T008)

#### Advanced Components (US7)

- [ ] T096 [P] [US7] Implement NormalDistribution in backtesting/distributions/normal.py for config comparison per FR-002 (include stationarity preflight per T025/T026 pattern)
- [ ] T097 [P] [US7] Add optional PyVollibPricer in quant-scenario-engine/pricing/py_vollib.py per FR-016 advanced toggle and SC-017
- [ ] T098 [P] [US7] Create QuantLib pricer stub in quant-scenario-engine/pricing/quantlib_stub.py for future extension per FR-016

#### CLI (US7)

- [ ] T099 [US7] Add --config flag to all CLI commands to load YAML config file per FR-024/FR-033
- [ ] T100 [US7] Implement CLI flag override of config values per plan.md onboarding workflow and FR-024/FR-026 precedence (CLI args override YAML values per T095a)
- [ ] T101 [US7] Document config file format in README.md with examples per spec.md US7 acceptance scenarios and FR-024/FR-033 requirements

**Checkpoint**: User Story 7 complete - component swapping via configuration verified

---

## Phase 9: User Story 3 - Feature-enriched signals (Priority: P3)

**Goal**: Augment simulations with technical indicators and optional macro series to drive signals without changing engine code

**Independent Test**: Add indicator definition (e.g., SMA/RSI) and confirm strategy receives feature columns and uses them in signal generation per spec.md acceptance scenarios

### Implementation for User Story 3

#### Feature Management (US3)

- [ ] T102 [US3] Implement dynamic indicator registry in quant-scenario-engine/features/registry.py to declare indicators via config per FR-006
- [ ] T103 [US3] Create indicator definition schema in quant-scenario-engine/schema/indicators.py supporting pandas-ta function specs
- [ ] T104 [US3] Extend feature pipeline to apply registered indicators from config per spec.md US3 acceptance scenario 1
- [ ] T105 [P] [US3] Implement macro series loader in quant-scenario-engine/data/macro.py with alignment logic per FR-014 (integrate with DataLoader pattern from T020h)
- [ ] T106 [P] [US3] Add macro series alignment with interpolation and tolerance enforcement per FR-014 (max 3× bar interval forward/backfill, warn on exceeding tolerance)
- [ ] T107 [US3] Implement missing feature warning system per spec.md US3 acceptance scenario 2

#### Strategy Integration (US3)

- [ ] T108 [US3] Update Strategy interface to accept features DataFrame in generate_signals() per data-model.md
- [ ] T109 [US3] Modify example strategies to demonstrate feature usage (e.g., SMA crossover, RSI threshold in StockBasicStrategy)
- [ ] T110 [US3] Add features_used tracking in StrategySignals per data-model.md (record which features were actually used in signal generation)

**Checkpoint**: User Story 3 complete - feature enrichment without code changes verified

---

## Phase 10: User Story 8 - Run provenance and replay (Priority: P3)

**Goal**: Inspect run directory to reconstruct how results were produced, re-run previous configuration to reproduce metrics

**Independent Test**: After running comparison, inspect run metadata file, re-run CLI with --replay <run_id> flag, confirm regenerated metrics match original per spec.md acceptance scenarios

### Implementation for User Story 8

#### Provenance Tracking (US8)

- [ ] T111 [US8] Enhance run_meta.json to capture all provenance fields per spec.md US8 acceptance scenario 1/SC-015 (symbol, timeframe, data_source, distribution, seeds, strategy_params, component_versions; ensure T044 completeness)
- [ ] T112 [P] [US8] Implement data version fingerprinting in quant-scenario-engine/data/versioning.py to detect Parquet schema/content changes per FR-019/SC-015 (reference T023a fingerprinting, add version comparison logic)
- [ ] T113 [US8] Add component version tracking in run_meta (package versions, git commit) per spec.md US8 acceptance scenario 1/FR-021 (already in T018a, ensure git SHA capture working)

#### Replay Functionality (US8)

- [ ] T114 [US8] Implement replay mode in quant-scenario-engine/simulation/replay.py to reload run_meta and regenerate paths per FR-019/FR-034 and spec.md US8 acceptance scenario 2
- [ ] T115 [US8] Add data drift detection with warning/block per FR-019/SC-015 (compare fingerprints from T112 vs current data, block unless allow_data_drift=true)
- [ ] T116 [US8] Implement npz-backed replay to load persisted MC paths when available per spec.md US8 acceptance scenario 2 option 2 (check for .npz files, load if present, otherwise regenerate; FR-014 alignment of storage)
- [ ] T117 [US8] Add replay metadata tagging (is_replay, original_run_id, data_drift_status) per FR-019/FR-021 (extend run_meta schema, record drift decisions)

#### CLI (US8)

- [ ] T118 [US8] Create replay command in quant-scenario-engine/cli/commands/replay.py per contracts/openapi.yaml ReplayRequest (FR-033)
- [ ] T119 [US8] Implement --replay flag or run_meta_path input with validation (FR-033/FR-019)
- [ ] T120 [US8] Wire replay CLI to replay mode with proper error handling per spec.md US8 acceptance scenario 3/FR-041 (handle missing run_meta, data drift blocks, MC path missing)

**Checkpoint**: User Story 8 complete - full provenance and replay capability verified

---

## Phase 11: Advanced Features (Optional Extensions)

**Purpose**: Advanced toggles and performance optimizations referenced in plan.md but not blocking MVP

- [ ] T121 [P] Implement GARCH-T distribution in backtesting/distributions/garch_t.py behind use_garch flag per FR-002, FR-032, FR-037 (include stationarity checks per T025/T026 pattern, warn on expected latency per research.md decision 2)
- [ ] T122 [P] Add numba JIT compilation to hot paths in MC generator for >2× speedup per FR-040 performance budget (annotate generate_price_paths with @njit, validate performance gains)
- [ ] T123 [P] Implement optional plotly report generation in quant-scenario-engine/utils/plots.py per FR-008
- [ ] T124 [P] Add quantstats integration for tearsheet generation in quant-scenario-engine/utils/quantstats_report.py per plan.md dependencies (optional reporting extension)
- [ ] T125 Create performance profiling utilities in quant-scenario-engine/utils/profiling.py to validate SC-001/SC-002/SC-003 time budgets
- [ ] T125a Add structured logging (JSON) diagnostics when performance budgets breach per FR-039/040 (emit warnings at 50%/90% of time budgets, detailed timing breakdowns)

---

## Phase 12: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and final validation

- [ ] T126 [P] Add comprehensive docstrings to all public interfaces per ASSUME-012 (≥80% test coverage quality standard) and constitution testing discipline
- [ ] T127 [P] Create architecture diagram (DataSource → Distribution → MC → Strategies → Simulation → Metrics) per plan.md phase 2
- [ ] T128 [P] Document all CLI commands in README.md with examples per plan.md onboarding
- [ ] T129 [P] Add type hints to all functions and validate with mypy --strict per constitution quality standards
- [ ] T130 Run code coverage analysis and ensure ≥80% coverage per ASSUME-012, constitution, and plan.md testing
- [ ] T131 Perform security audit of external data sources and input validation per FR-057 (timestamp validation), FR-086 (data source failures), and constitution
- [ ] T132 Run linting (ruff) and formatting (black) across entire codebase per constitution quality standards
- [ ] T133 Create quickstart.md validation script in scripts/validate_quickstart.py to test documented workflows per FR-076 (backward compatibility testing)
- [ ] T134 Document performance budget validation in scripts/benchmark.py (SC-001: 1k×60 ≤10s, SC-002: grid ≤15m per T007a benchmarks)
- [ ] T135 Add constitution compliance checklist in CONSTITUTION_CHECK.md covering all principles
- [ ] T136 Create deployment guide for CPU-only VPS in docs/deployment.md per FR-068 (VPS OS and Python constraints) and plan.md target platform

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational - No dependencies on other stories 🎯 MVP
- **User Story 4 (Phase 4)**: Depends on Foundational + US1 (data layer, features) - Foundation for US5/US6
- **User Story 5 (Phase 5)**: Depends on US4 (candidate selection) - Can proceed after US4
- **User Story 2 (Phase 6)**: Depends on US1 (simulation engine) - Independent from US4/US5
- **User Story 6 (Phase 7)**: Depends on US4 + US5 (conditional logic) - Can proceed after US5
- **User Story 7 (Phase 8)**: Depends on US1 (factories) - Can start anytime after US1
- **User Story 3 (Phase 9)**: Depends on US1 (feature pipeline) - Low priority, can be last
- **User Story 8 (Phase 10)**: Depends on US1 (run artifacts) - Can proceed after US1
- **Advanced Features (Phase 11)**: Optional - depends on relevant user stories
- **Polish (Phase 12)**: Depends on all desired user stories being complete

### User Story Dependencies

```
Foundation (Phase 2) BLOCKS ALL

After Foundation:
├─ US1 (P1) 🎯 MVP ─────────────────┬─ US2 (P2) Grid
│                                   ├─ US7 (P2) Config swap
│                                   ├─ US8 (P3) Replay
│                                   └─ US3 (P3) Features
│
└─ US4 (P1) Screening ──────────────┬─ US5 (P1) Conditional backtest
                                    │
                                    └─ US6 (P2) Conditional MC
```

### Critical Path for MVP (US1 only)

1. Phase 1: Setup (T001-T007a) → ~1 hour
2. Phase 2: Foundational (T008-T020m) → ~4 hours (expanded from 3 with caching tasks)
3. Phase 3: US1 Implementation (T021-T051) → ~10 hours (expanded from 8 with new features)
4. **Total MVP time: ~15 hours of focused development**

### Parallel Opportunities

#### Within Setup (Phase 1)
- T003, T004, T005, T006, T007, T007a can all run in parallel after T001-T002

#### Within Foundational (Phase 2)
- T009, T010, T011, T012, T017, T018, T018a, T019 can run in parallel after T008
- T020a-g can run in parallel after T019-T020
- T020h-m can run in parallel after T020 (caching/stationarity/logging infrastructure)
- Data layer, schemas, and utilities are independent

#### Within User Story 1 (Phase 3)
- Data adapters: T021, T022, T023 parallel (T021a, T023a sequential after their parents)
- Distributions: T025, T026, T028a, T028b can overlap
- Storage: T030, T030a, T031 can overlap
- Pricers: T033, T034 after T015
- Strategies: T035, T036 parallel after T014
- Artifacts: T042, T043, T043a parallel
- CLI: T048-T051 can overlap

#### Across User Stories (after Foundational)
- US1 + US2 + US7 + US8 can proceed in parallel (different concerns)
- US4 must complete before US5
- US5 must complete before US6
- US3 independent, can proceed anytime after US1

---

## Parallel Example: User Story 1 Data Layer

```bash
# Launch all data adapters together:
Task: "Implement YFinanceDataSource with retries in data/yfinance.py"
Task: "Create SchwabDataSourceStub in data/schwab_stub.py"
Task: "Implement data validation in data/validation.py"

# Then sequential:
Task: "Integrate YFinanceDataSource with DataLoader" (depends on T020h, T021)
Task: "Implement fingerprinting" (depends on T023)
Task: "Create DataSource factory" (depends on all adapters)
```

## Parallel Example: User Story 1 Distributions

```bash
# Launch both distribution models together:
Task: "Implement LaplaceDistribution with stationarity checks"
Task: "Implement StudentTDistribution with AR detection"
Task: "Add parameter bounds checking"
Task: "Implement convergence limits"

# After all complete:
Task: "Create distribution factory"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only) 🎯

Recommended for single developer or tight deadline:

1. **Week 1**: Complete Phase 1 + Phase 2 (T001-T020m) → Foundation ready
2. **Week 2**: Complete Phase 3 (T021-T051) → US1 MVP done
3. **STOP and VALIDATE**: Run compare CLI with AAPL, verify artifacts, test success criteria SC-001/SC-002
4. **Deploy**: Single working feature ready for demo/production

**Why MVP first?**
- Delivers value immediately (stock vs option comparison)
- Validates entire technical stack end-to-end including caching, stationarity checks, VaR/CVaR
- Provides foundation for all other stories
- De-risks architecture decisions early

### Incremental Delivery (Recommended)

For sustained development with multiple capabilities:

1. **Sprint 1** (Setup + Foundation): T001-T020m → Foundation ready
2. **Sprint 2** (US1 MVP): T021-T051 → Compare CLI working ✅
3. **Sprint 3** (US4 Screening): T052-T065 → Screen CLI working ✅
4. **Sprint 4** (US5 Conditional): T066-T073 → Conditional backtest working ✅
5. **Sprint 5** (US2 Grid): T074-T082 → Grid optimization working ✅
6. **Sprint 6** (US6 Conditional MC): T083-T090 → Full conditional suite ✅
7. **Sprint 7** (US7 Config + US8 Replay): T091-T120 → System mature ✅
8. **Sprint 8** (US3 Features + Polish): T102-T136 → Production ready ✅

Each sprint adds value without breaking previous features.

### Parallel Team Strategy

With 3+ developers:

1. **All developers**: Complete Setup + Foundational together (T001-T020m)
2. **Once Foundational done**:
   - Developer A: US1 (T021-T051) - Critical path
   - Developer B: US4 (T052-T065) after US1 data layer (T021-T024)
   - Developer C: US2 (T074-T082) after US1 simulation (T038-T041)
3. **After US1 + US4 complete**:
   - Developer A: US7 (T091-T101)
   - Developer B: US5 (T066-T073) depends on US4
   - Developer C: US8 (T111-T120)
4. **Final sprint**: US3, US6, Advanced, Polish together

**Parallelization gains**: ~40% faster with 3 developers vs sequential

---

## Notes

- [P] tasks = different files, no dependencies within phase
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable per constitution
- File paths follow plan.md structure: quant-scenario-engine/{module}/file.py
- All tasks reference specific FRs, DMs, or acceptance criteria from spec.md
- Commit after each task or logical group per plan.md onboarding
- Stop at any checkpoint to validate story independently per spec.md
- Constitution check at end: specification-driven (Principle XVII), contracts (Principle VIII), testing ≥80% (Principle XV), simplicity (Principle III)

---

## Success Criteria Mapping

| Success Criteria | Validated By Tasks | Phase |
|------------------|-------------------|-------|
| SC-001: Baseline run ≤10s | T029-T032, T134 (benchmark) | US1 + Polish |
| SC-002: Artifacts include stock+option metrics | T042-T044 (artifacts with VaR/CVaR) | US1 |
| SC-003: Grid produces ranked configs | T076-T077 (ranking) | US2 |
| SC-004: Reproducible with seed | T032 (seeding), T117 (replay) | US1 + US8 |
| SC-005: Clear errors for failures | T008 (exceptions), T050 (CLI errors) | Foundation + US1 |
| SC-006: Component swap via config | T091-T095 (config), T020k (logging) | US7 |
| SC-007: Conditional MC reproducible | T087 (conditional MC), T090 (seeding) | US6 |
| SC-008: Conditional backtest ≥100 episodes | T066-T069 (conditional) | US5 |
| SC-009: Stock vs option conditional metrics | T068-T070 (episode metrics) | US5 |
| SC-010: Screen ≥100 symbols | T059-T061 (screening) | US4 |
| SC-011: Unconditional vs conditional reporting | T070 (comparison) | US5 |
| SC-012: Selector change without code | T055-T058 (selector), T094 (config) | US4 + US7 |
| SC-013: Parquet schema consistency | T019 (structure), T020a (validation), T020f (versioning), T023 (data validation) | Foundation + US1 |
| SC-014: Auto memmap fallback | T031 (memmap), T020 (policy), T030a (estimated_gb) | Foundation + US1 |
| SC-015: run_meta.json complete | T044 (run_meta), T111 (provenance) | US1 + US8 |
| SC-016: Distribution swap changes behavior | T027 (factory), T096 (normal) | US1 + US7 |
| SC-017: Pricer swap changes option curves | T034 (factory), T097 (py_vollib) | US1 + US7 |
| SC-018: New strategy pluggable | T014 (interface), T035-T036 (examples) | Foundation + US1 |
| SC-019: Fit failure structured error | T008 (exceptions), T028b (validation) | Foundation + US1 |
| SC-020: Data gap warnings | T023 (validation), T062 (screening) | US1 + US4 |
| SC-021: Sufficient logs for debugging | T009 (logging), T044 (artifacts), T125a (diagnostics) | Foundation + US1 + Advanced |
