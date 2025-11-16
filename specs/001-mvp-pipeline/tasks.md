# Tasks: Backtesting & Strategy Spec Authoring (Quant Scenario Engine)

**Input**: Design documents from `/specs/001-mvp-pipeline/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/openapi.yaml

**Tests**: Not explicitly requested in specification - implementing core functionality only per constitution principle (build only what spec requires)

**DM-Series Coverage**: Added T020d-g to address gaps identified in Specification_Analysis_Report.md (DM-001/002/003 resolution tiers, DM-005 feature separation, DM-007 symbol tiers, DM-013 versioning). Existing tasks T019, T020, T020a-c, T030-031 cover remaining DM requirements.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story per constitution-driven development (Principle XVII).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

Per plan.md structure:
- **Source**: (quant-scenario-engine/{features,models,runs,schema,interfaces,backtesting/{distributions,mc},optimizer,pricing,strategies,simulation,cli,config}, 
- **Tests**: `tests/` (unit, integration, contract)
- **Artifacts**: `runs/`, `data/` (historical, features)
- **Package name**: `backtesting` (aligned with plan.md and quickstart.md)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project directory structure per plan. (quant-scenario-engine/{data,runs,schema,features,models,runs,interfaces,backtesting/{distributions,mc},optimizer,pricing,strategies,simulation,cli,config}, tests/{unit,integration,contract})
- [ ] T002 Initialize Python 3.11 project with pyproject.toml and core dependencies (numpy, pandas, scipy, numba, statsmodels, arch, pandas-ta, quantstats, plotly, yfinance, typer, pytest)
- [ ] T003 [P] Configure ruff linting and black formatting with settings in pyproject.toml
- [ ] T004 [P] Setup mypy type checking configuration in pyproject.toml
- [ ] T005 [P] Create .gitignore for Python (venv, __pycache__, *.pyc, .pytest_cache, runs/, data/historical/, data/features/)
- [ ] T006 [P] Create README.md with quickstart instructions per plan.md onboarding section
- [ ] T007 Create requirements-dev.txt with testing/dev dependencies (pytest, pytest-cov, ruff, black, mypy)

# Added to address new FR/NFR/DM gaps
- [ ] T007a Define performance budget benchmarks and add validation script (measure data load, fit, MC throughput, strategy eval) per FR-015 and FR-040 (scripts/benchmarks/perf_check.py)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T008 Create base exception classes in quant-scenario-engine/exceptions.py (DataSourceError, DistributionFitError, ResourceLimitError, PricingError, EpisodeGenerationError)
- [ ] T009 [P] Implement logging configuration in quant-scenario-engine/utils/logging.py with structured JSON output
- [ ] T010 [P] Create RunConfig dataclass in quant-scenario-engine/schema/run_config.py per data-model.md with validation
- [ ] T011 [P] Create OptionSpec dataclass in quant-scenario-engine/models/options.py per data-model.md FR-016
- [ ] T012 [P] Create StrategyParams dataclass in quant-scenario-engine/schema/strategy.py per data-model.md
- [ ] T013 Create ReturnDistribution abstract base class in quant-scenario-engine/interfaces/distribution.py per data-model.md
- [ ] T014 Create Strategy abstract base class in quant-scenario-engine/interfaces/strategy.py per data-model.md
- [ ] T015 Create OptionPricer abstract base class in quant-scenario-engine/interfaces/pricing.py per data-model.md
- [ ] T016 Create CandidateSelector abstract base class in quant-scenario-engine/interfaces/candidate_selector.py per FR-CAND-001
- [ ] T017 [P] Implement resource estimator in quant-scenario-engine/utils/resources.py (estimate memory footprint for n_paths × n_steps, enforce FR-013/FR-018 thresholds)
- [ ] T018 [P] Create run metadata schema in quant-scenario-engine/schema/run_meta.py with JSON serialization per FR-008/FR-019
- [ ] T018a [P] Capture reproducibility context in run_meta (seeds, library versions, environment: CPU/RAM/OS, git commit) per FR-021 and CHK139-CHK141
- [ ] T019 [P] Setup Parquet data directory structure in data/{historical,features}/{interval}/{symbol}.parquet per DM-006
- [ ] T020 Create storage policy selector in backtesting/mc/storage.py (decide memory vs npz vs memmap based on RAM threshold per DM-008/DM-010/DM-011)
- [ ] T020a Add Parquet schema validation and fingerprint computation in quant-scenario-engine/data/validation.py (DM-015, DM-016, FR-027, FR-028)
- [ ] T020b Add missing-data tolerance enforcement (3× interval continuous gap, 1% total) and configurable imputation rules in data validation (FR-029, DM-017)
- [ ] T020c Ensure run_meta writes are atomic/immutable and record persistence usage (FR-030, FR-031, DM-018)
- [ ] T020d [P] Document resolution tier strategy (daily for distribution fitting, 5-min for backtesting, 1-min for live) and implement tier selection logic in data validation (DM-001, DM-002, DM-003)
- [ ] T020e [P] Enforce raw OHLCV vs derived features separation in Parquet storage (data/historical/{interval}/ vs data/features/{interval}/) per DM-005
- [ ] T020f [P] Add Parquet historical data versioning support (_v2, _v3 suffixes) and retention policy documentation per DM-013
- [ ] T020g [P] Create Universe/Watchlist/Live Set configuration schema (symbol tiers: Universe=daily+5min, Watchlist=daily+5min+features, Live=all resolutions) per DM-007

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Run stock vs option Monte Carlo comparison (Priority: P1) 🎯 MVP

**Goal**: Execute CLI to generate Monte Carlo price paths, apply paired stock and option strategies, receive equity curves plus summary metrics

**Independent Test**: Execute CLI with sample symbol (e.g., AAPL) and verify outputs include both stock and option metrics plus saved report artifacts per spec.md acceptance scenarios

### Implementation for User Story 1

#### Data Layer (US1)

- [ ] T021 [P] [US1] Implement YFinanceDataSource adapter in quant-scenario-engine/data/yfinance.py with load_ohlcv(symbol, start, end, interval) per FR-001
- [ ] T022 [P] [US1] Create SchwabDataSourceStub in quant-scenario-engine/data/schwab_stub.py with matching interface per FR-011/FR-017
- [ ] T023 [P] [US1] Implement data validation in quant-scenario-engine/data/validation.py (check OHLCV columns, handle missing data per FR-010)
- [ ] T024 [US1] Create DataSource factory in quant-scenario-engine/data/factory.py to select yfinance vs schwab per config per FR-009

#### Distribution Models (US1)

- [ ] T025 [P] [US1] Implement LaplaceDistribution in backtesting/distributions/laplace.py with fit() and sample() per FR-002 (default model)
- [ ] T026 [P] [US1] Implement StudentTDistribution in backtesting/distributions/student_t.py with fit() and sample() per FR-002
- [ ] T027 [P] [US1] Create distribution factory in backtesting/distributions/factory.py to select model by config (laplace default, student_t optional)
- [ ] T028 [US1] Implement distribution parameter validation and fallback logic in backtesting/distributions/validation.py per data-model.md

#### Monte Carlo Engine (US1)

- [ ] T029 [US1] Implement generate_price_paths() in backtesting/mc/generator.py (s0, distribution, n_paths, n_steps, seed) per plan.md reference interface
- [ ] T030 [P] [US1] Implement in-memory path storage in backtesting/mc/storage.py for footprint <25% RAM per DM-008
- [ ] T031 [P] [US1] Implement memmap/npz fallback in backtesting/mc/storage.py for footprint ≥25% RAM per DM-011/FR-013
- [ ] T032 [US1] Add seeded random number generation with reproducibility guarantee per FR-012

#### Option Pricing (US1)

- [ ] T033 [US1] Implement BlackScholesPricer in quant-scenario-engine/pricing/black_scholes.py with price(spot, spec, ttm) per FR-016 and plan.md reference
- [ ] T034 [P] [US1] Create OptionPricer factory in quant-scenario-engine/pricing/factory.py to select black_scholes (default) per FR-009/FR-016

#### Strategy Framework (US1)

- [ ] T035 [US1] Create example StockBasicStrategy in quant-scenario-engine/strategies/stock_basic.py implementing Strategy interface per data-model.md
- [ ] T036 [US1] Create example OptionCallStrategy in quant-scenario-engine/strategies/option_call.py implementing Strategy interface with OptionSpec per data-model.md
- [ ] T037 [US1] Implement StrategySignals dataclass in quant-scenario-engine/schema/signals.py (signals_stock, signals_option, option_spec, features_used) per data-model.md

#### Simulation Engine (US1)

- [ ] T038 [US1] Implement MarketSimulator class in quant-scenario-engine/simulation/simulator.py with simulate_stock() and simulate_option() per FR-003/FR-004 and plan.md reference
- [ ] T039 [US1] Implement metrics calculator in quant-scenario-engine/simulation/metrics.py (mean/median P&L, drawdown, VaR/CVaR, Sharpe/Sortino) per FR-003
- [ ] T040 [US1] Create SimulationRun orchestrator in quant-scenario-engine/simulation/run.py to coordinate MC, strategies, simulation per data-model.md
- [ ] T041 [US1] Implement run_compare() function in quant-scenario-engine/simulation/compare.py integrating all components for stock vs option comparison per FR-005
- [ ] T041a [US1] Enforce CLI parameter validation against contracts/openapi.yaml for compare command (FR-033)

#### Artifacts & Persistence (US1)

- [ ] T042 [P] [US1] Implement MetricsReport serialization in quant-scenario-engine/schema/metrics.py with JSON/CSV export per FR-008
- [ ] T043 [P] [US1] Create run artifact writer in quant-scenario-engine/utils/artifacts.py (metrics, plots, run_meta.json) per FR-008
- [ ] T043a [P] [US1] Ensure run_meta/artifacts write atomically and include schema versions; record persistence usage (FR-030, FR-031, FR-034)
- [ ] T044 [US1] Implement run_meta.json persistence in quant-scenario-engine/utils/run_meta.py with all required fields per FR-019 and data-model.md example

#### CLI (US1)

- [ ] T045 [US1] Create Typer CLI entrypoint in quant-scenario-engine/cli/main.py with compare command per FR-005
- [ ] T046 [US1] Implement config validation in quant-scenario-engine/cli/validation.py enforcing contracts/openapi.yaml CompareRequest schema
- [ ] T047 [US1] Wire compare CLI to run_compare() in quant-scenario-engine/cli/commands/compare.py with proper error handling per plan.md error policies
- [ ] T048 [US1] Add CLI argument parsing for symbol, start, end, distribution, paths, steps, seed, data_source, option_pricer, strategies per contracts/openapi.yaml

**Checkpoint**: User Story 1 MVP complete - can execute stock vs option comparison and produce artifacts

---

## Phase 4: User Story 4 - Stock screening via candidate selector (Priority: P1)

**Goal**: Scan universe of symbols, apply candidate-selection rule, receive ranked list of stocks for strategy evaluation

**Independent Test**: Configure universe of ≥100 symbols and simple candidate rule (e.g., top 20 daily gainers/losers with volume filters), run CLI, confirm ranked candidate list with metadata per spec.md acceptance scenarios

**Note**: Implementing US4 before US2/US3 as it's P1 priority and provides foundation for US5/US6 conditional logic

### Implementation for User Story 4

#### Feature Engineering (US4)

- [ ] T049 [P] [US4] Implement technical indicator wrapper in quant-scenario-engine/features/indicators.py using pandas-ta (SMA, RSI, volume_z) per FR-006
- [ ] T050 [P] [US4] Create feature computation pipeline in quant-scenario-engine/features/pipeline.py to enrich OHLCV with indicators
- [ ] T051 [US4] Implement gap percentage calculator in quant-scenario-engine/features/gap.py (open vs prev_close) for selector rules

#### Candidate Selection (US4)

- [ ] T052 [US4] Implement GapVolumeSelector in quant-scenario-engine/selectors/gap_volume.py extending CandidateSelector per FR-CAND-001/FR-CAND-006 (default MVP selector)
- [ ] T053 [US4] Implement selector.score() and selector.select() methods per data-model.md CandidateSelector interface
- [ ] T054 [P] [US4] Create CandidateEpisode dataclass in quant-scenario-engine/schema/episode.py per data-model.md
- [ ] T055 [US4] Implement build_candidate_episodes() in quant-scenario-engine/selectors/episodes.py per data-model.md reference code
- [ ] T055a [US4] Document and enforce candidate episode construction rules (horizon > 0, state_features captured) across backtest and conditional MC (FR-035)

#### Screening Engine (US4)

- [ ] T056 [US4] Implement universe screening logic in quant-scenario-engine/simulation/screen.py to scan symbols and apply selector per FR-CAND-001
- [ ] T057 [US4] Add parallel symbol processing with max_workers cap per FR-018 concurrency model
- [ ] T058 [US4] Implement candidate ranking and top_n filtering in quant-scenario-engine/simulation/screen.py per contracts/openapi.yaml ScreenRequest
- [ ] T059 [US4] Add error handling for missing/partial data per symbol per spec.md US4 acceptance scenario 2

#### CLI (US4)

- [ ] T060 [US4] Create screen command in quant-scenario-engine/cli/commands/screen.py with universe and selector arguments per contracts/openapi.yaml ScreenRequest
- [ ] T061 [US4] Wire screen CLI to screening engine with validation per contracts/openapi.yaml ScreenResponse
- [ ] T062 [US4] Implement ScreenResponse serialization with candidate list (symbol, t0, state_features) per contracts/openapi.yaml

**Checkpoint**: User Story 4 complete - can screen universe and generate candidate lists

---

## Phase 5: User Story 5 - Conditional episode backtesting for candidates (Priority: P1)

**Goal**: Run conditional backtest evaluating strategy only on historical episodes where candidate selector fired, receive episode-level and aggregate metrics

**Independent Test**: Define candidate selector (e.g., large gap + volume spike), build ≥50 historical episodes for test symbol, run conditional backtest, verify episode-level and aggregate metrics for stock and option per spec.md acceptance scenarios

### Implementation for User Story 5

#### Conditional Backtesting (US5)

- [ ] T063 [US5] Implement conditional episode filtering in quant-scenario-engine/simulation/conditional.py to extract candidate-only windows per FR-CAND-003
- [ ] T064 [US5] Extend MarketSimulator to support episode-level P&L tracking in quant-scenario-engine/simulation/simulator.py
- [ ] T065 [US5] Implement episode-level metrics aggregation in quant-scenario-engine/simulation/metrics.py per spec.md SC-009
- [ ] T066 [US5] Create run_conditional_backtest() in quant-scenario-engine/simulation/conditional.py orchestrating selector → episodes → strategy evaluation per FR-CAND-003
- [ ] T067 [US5] Add unconditional vs conditional metrics comparison reporting per spec.md US5 acceptance scenario 2

#### CLI (US5)

- [ ] T068 [US5] Create conditional command in quant-scenario-engine/cli/commands/conditional.py with selector parameter per contracts/openapi.yaml ConditionalRequest
- [ ] T069 [US5] Wire conditional CLI to run_conditional_backtest() with proper artifact generation per FR-008
- [ ] T070 [US5] Implement min_episodes validation and fallback warning per FR-CAND-006 and spec.md US5 acceptance scenario 3

**Checkpoint**: User Story 5 complete - can run conditional backtests on historical candidate episodes

---

## Phase 6: User Story 2 - Parameter grid exploration (Priority: P2)

**Goal**: Define parameter grid, run batch evaluations in parallel, receive aggregate metrics (mean, CVaR, drawdown) per configuration

**Independent Test**: Supply small grid (≥3 configs) and confirm per-config metrics are produced and ranked by objective function per spec.md acceptance scenarios

### Implementation for User Story 2

#### Grid Execution (US2)

- [ ] T071 [US2] Implement grid parameter expansion in quant-scenario-engine/simulation/grid.py to generate StrategyParams combinations per contracts/openapi.yaml StrategyGridConfig
- [ ] T072 [US2] Create grid runner with ProcessPoolExecutor in quant-scenario-engine/simulation/grid.py using max_workers from config per plan.md concurrency model
- [ ] T073 [US2] Implement per-config metrics collection and aggregation in quant-scenario-engine/simulation/grid.py
- [ ] T074 [US2] Add objective function scoring and ranking in quant-scenario-engine/simulation/metrics.py per FR-007 and spec.md US2 acceptance scenario 1
- [ ] T075 [US2] Implement resource limit preflight check in quant-scenario-engine/simulation/grid.py per FR-018 (estimate time/memory for N configs, abort if > thresholds)
- [ ] T076 [US2] Add race condition prevention and output integrity validation per spec.md US2 acceptance scenario 2

#### CLI (US2)

- [ ] T077 [US2] Create grid command in quant-scenario-engine/cli/commands/grid.py with grid parameter per contracts/openapi.yaml GridRequest
- [ ] T078 [US2] Wire grid CLI to grid runner with progress tracking and warnings per FR-018
- [ ] T079 [US2] Implement GridResponse serialization with ranked configurations per contracts/openapi.yaml

**Checkpoint**: User Story 2 complete - can execute parameter grids with ranking

---

## Phase 7: User Story 6 - Conditional Monte Carlo from candidate states (Priority: P2)

**Goal**: Generate Monte Carlo paths conditioned on current candidate state, run strategies over conditional paths, review conditional risk/reward distribution

**Independent Test**: For symbol and candidate state, run conditional Monte Carlo CLI (e.g., 1,000 paths × 60 steps) and verify state-conditioned model usage and summary distributions per spec.md acceptance scenarios

### Implementation for User Story 6

#### Conditional Sampling (US6)

- [ ] T080 [US6] Implement episode bootstrapping sampler in backtesting/distributions/episode_bootstrap.py for non-parametric conditional MC per FR-CAND-005
- [ ] T081 [P] [US6] Implement state-conditioned distribution refit in backtesting/distributions/conditional.py for parametric conditional MC per FR-CAND-005
- [ ] T082 [US6] Create conditional MC method selector in backtesting/mc/conditional.py (bootstrap vs refit) with fallback logic per FR-CAND-005
- [ ] T083 [US6] Implement minimum episode threshold check with warning and fallback per spec.md US6 acceptance scenario 3
- [ ] T084 [US6] Create run_conditional_mc() in quant-scenario-engine/simulation/conditional_mc.py integrating conditional sampling with strategies per FR-CAND-004
- [ ] T084a [US6] Log method selection and fallbacks (bootstrap → refit → unconditional) in run_meta/logs per FR-036/FR-038

#### CLI (US6)

- [ ] T085 [US6] Extend conditional CLI to support monte_carlo mode in quant-scenario-engine/cli/commands/conditional.py
- [ ] T086 [US6] Add state_features input parsing and validation per contracts/openapi.yaml CandidateEpisode
- [ ] T087 [US6] Wire conditional MC CLI to run_conditional_mc() with reproducibility guarantee per spec.md US6 acceptance scenario 2

**Checkpoint**: User Story 6 complete - can generate conditional Monte Carlo paths from candidate states

---

## Phase 8: User Story 7 - Config-driven component swapping (Priority: P2)

**Goal**: Change configuration (YAML/env/CLI flags) to swap data sources, distribution models, option pricers without code edits

**Independent Test**: Run three successive CLI runs varying only configuration (e.g., yfinance+Normal+BS vs Schwab+Student-T+BS vs Schwab+Student-T+AltPricer) and confirm wiring changes per spec.md acceptance scenarios

### Implementation for User Story 7

#### Configuration Management (US7)

- [ ] T088 [US7] Create YAML configuration schema in quant-scenario-engine/config/schema.py matching contracts/openapi.yaml request schemas
- [ ] T089 [P] [US7] Implement YAML config loader in quant-scenario-engine/config/loader.py with validation
- [ ] T090 [P] [US7] Add environment variable override support in quant-scenario-engine/config/env.py per plan.md onboarding
- [ ] T091 [US7] Implement config validation with fail-fast on invalid values per spec.md US7 acceptance scenario 3
- [ ] T092 [US7] Wire all factories (data, distribution, pricer) to read from unified config per FR-009
- [ ] T092a [US7] Enforce configuration precedence (CLI > ENV > YAML) and log overrides per FR-024/FR-026
- [ ] T092b [US7] Detect and block incompatible config combinations (e.g., pricer not supported) with structured errors per FR-025

#### Advanced Components (US7)

- [ ] T093 [P] [US7] Implement NormalDistribution in backtesting/distributions/normal.py for config comparison per FR-002
- [ ] T094 [P] [US7] Add optional PyVollibPricer in quant-scenario-engine/pricing/py_vollib.py per FR-016 advanced toggle
- [ ] T095 [P] [US7] Create QuantLib pricer stub in quant-scenario-engine/pricing/quantlib_stub.py for future extension per FR-016

#### CLI (US7)

- [ ] T096 [US7] Add --config flag to all CLI commands to load YAML config file
- [ ] T097 [US7] Implement CLI flag override of config values per plan.md onboarding workflow
- [ ] T098 [US7] Document config file format in README.md with examples per spec.md US7 acceptance scenarios

**Checkpoint**: User Story 7 complete - component swapping via configuration verified

---

## Phase 9: User Story 3 - Feature-enriched signals (Priority: P3)

**Goal**: Augment simulations with technical indicators and optional macro series to drive signals without changing engine code

**Independent Test**: Add indicator definition (e.g., SMA/RSI) and confirm strategy receives feature columns and uses them in signal generation per spec.md acceptance scenarios

### Implementation for User Story 3

#### Feature Management (US3)

- [ ] T099 [US3] Implement dynamic indicator registry in quant-scenario-engine/features/registry.py to declare indicators via config per FR-006
- [ ] T100 [US3] Create indicator definition schema in quant-scenario-engine/schema/indicators.py supporting pandas-ta function specs
- [ ] T101 [US3] Extend feature pipeline to apply registered indicators from config per spec.md US3 acceptance scenario 1
- [ ] T102 [P] [US3] Implement macro series loader in quant-scenario-engine/data/macro.py with alignment logic per FR-014
- [ ] T103 [P] [US3] Add macro series alignment with interpolation and tolerance enforcement per FR-014 (max 3× bar interval forward/backfill)
- [ ] T104 [US3] Implement missing feature warning system per spec.md US3 acceptance scenario 2

#### Strategy Integration (US3)

- [ ] T105 [US3] Update Strategy interface to accept features DataFrame in generate_signals() per data-model.md
- [ ] T106 [US3] Modify example strategies to demonstrate feature usage (e.g., SMA crossover, RSI threshold)
- [ ] T107 [US3] Add features_used tracking in StrategySignals per data-model.md

**Checkpoint**: User Story 3 complete - feature enrichment without code changes verified

---

## Phase 10: User Story 8 - Run provenance and replay (Priority: P3)

**Goal**: Inspect run directory to reconstruct how results were produced, re-run previous configuration to reproduce metrics

**Independent Test**: After running comparison, inspect run metadata file, re-run CLI with --replay <run_id> flag, confirm regenerated metrics match original per spec.md acceptance scenarios

### Implementation for User Story 8

#### Provenance Tracking (US8)

- [ ] T108 [US8] Enhance run_meta.json to capture all provenance fields per spec.md US8 acceptance scenario 1 (symbol, timeframe, data_source, distribution, seeds, strategy_params, component_versions)
- [ ] T109 [P] [US8] Implement data version fingerprinting in quant-scenario-engine/data/versioning.py to detect Parquet schema/content changes per FR-019
- [ ] T110 [US8] Add component version tracking in run_meta (package versions, git commit) per spec.md US8 acceptance scenario 1

#### Replay Functionality (US8)

- [ ] T111 [US8] Implement replay mode in quant-scenario-engine/simulation/replay.py to reload run_meta and regenerate paths per FR-019 and spec.md US8 acceptance scenario 2
- [ ] T112 [US8] Add data drift detection with warning/block per FR-019 (unless allow_data_drift=true)
- [ ] T113 [US8] Implement npz-backed replay to load persisted MC paths when available per spec.md US8 acceptance scenario 2 option 2
- [ ] T114 [US8] Add replay metadata tagging (is_replay, original_run_id, data_drift_status) per FR-019

#### CLI (US8)

- [ ] T115 [US8] Create replay command in quant-scenario-engine/cli/commands/replay.py per contracts/openapi.yaml ReplayRequest
- [ ] T116 [US8] Implement --replay flag or run_meta_path input with validation
- [ ] T117 [US8] Wire replay CLI to replay mode with proper error handling per spec.md US8 acceptance scenario 3

**Checkpoint**: User Story 8 complete - full provenance and replay capability verified

---

## Phase 11: Advanced Features (Optional Extensions)

**Purpose**: Advanced toggles and performance optimizations referenced in plan.md but not blocking MVP

- [ ] T118 [P] Implement GARCH-T distribution in backtesting/distributions/garch_t.py behind use_garch flag per research.md decision 2
- [ ] T119 [P] Add numba JIT compilation to hot paths in MC generator for >2× speedup per plan.md performance budget
- [ ] T120 [P] Implement optional plotly report generation in quant-scenario-engine/utils/plots.py per FR-008
- [ ] T121 [P] Add quantstats integration for tearsheet generation in quant-scenario-engine/utils/quantstats_report.py per plan.md dependencies
- [ ] T122 Create performance profiling utilities in quant-scenario-engine/utils/profiling.py to validate SC-001/SC-002/SC-003 time budgets
- [ ] T122a Add structured logging (JSON) with required fields and progress events; emit diagnostics when performance budgets breach (FR-039, FR-040)

---

## Phase 12: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and final validation

- [ ] T123 [P] Add comprehensive docstrings to all public interfaces per constitution testing discipline
- [ ] T124 [P] Create architecture diagram (DataSource → Distribution → MC → Strategies → Simulation → Metrics) per plan.md phase 2
- [ ] T125 [P] Document all CLI commands in README.md with examples per plan.md onboarding
- [ ] T126 [P] Add type hints to all functions and validate with mypy --strict
- [ ] T127 Run code coverage analysis and ensure ≥80% coverage per constitution and plan.md testing
- [ ] T128 Perform security audit of external data sources and input validation per constitution
- [ ] T129 Run linting (ruff) and formatting (black) across entire codebase
- [ ] T130 Create quickstart.md validation script in scripts/validate_quickstart.py to test documented workflows
- [ ] T131 Document performance budget validation in scripts/benchmark.py (SC-001: 1k×60 ≤10s, SC-002: grid ≤15m)
- [ ] T132 Add constitution compliance checklist in CONSTITUTION_CHECK.md covering all principles
- [ ] T133 Create deployment guide for CPU-only VPS in docs/deployment.md per plan.md target platform

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

1. Phase 1: Setup (T001-T007) → ~1 hour
2. Phase 2: Foundational (T008-T020) → ~3 hours
3. Phase 3: US1 Implementation (T021-T048) → ~8 hours
4. **Total MVP time: ~12 hours of focused development**

### Parallel Opportunities

#### Within Setup (Phase 1)
- T003, T004, T005, T006, T007 can all run in parallel after T001-T002

#### Within Foundational (Phase 2)
- T009, T010, T011, T012, T017, T018, T019 can run in parallel after T008
- Data layer, schemas, and utilities are independent

#### Within User Story 1 (Phase 3)
- Data adapters: T021, T022, T023 parallel
- Distributions: T025, T026 parallel
- Storage: T030, T031 parallel
- Pricers: T033, T034 after T015
- Strategies: T035, T036 parallel after T014
- Artifacts: T042, T043 parallel

#### Across User Stories (after Foundational)
- US1 + US2 + US7 + US8 can proceed in parallel (different concerns)
- US4 must complete before US5
- US5 must complete before US6
- US3 independent, can proceed anytime after US1

---

## Parallel Example: User Story 1 Data Layer

```bash
# Launch all data adapters together:
Task: "Implement YFinanceDataSource in quant-scenario-engine/data/yfinance.py"
Task: "Create SchwabDataSourceStub in quant-scenario-engine/data/schwab_stub.py"
Task: "Implement data validation in quant-scenario-engine/data/validation.py"

# Then sequential:
Task: "Create DataSource factory in quant-scenario-engine/data/factory.py"
```

## Parallel Example: User Story 1 Distributions

```bash
# Launch both distribution models together:
Task: "Implement LaplaceDistribution in backtesting/distributions/laplace.py"
Task: "Implement StudentTDistribution in backtesting/distributions/student_t.py"

# After both complete:
Task: "Create distribution factory in backtesting/distributions/factory.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only) 🎯

Recommended for single developer or tight deadline:

1. **Week 1**: Complete Phase 1 + Phase 2 (T001-T020) → Foundation ready
2. **Week 2**: Complete Phase 3 (T021-T048) → US1 MVP done
3. **STOP and VALIDATE**: Run compare CLI with AAPL, verify artifacts, test success criteria SC-001/SC-002
4. **Deploy**: Single working feature ready for demo/production

**Why MVP first?**
- Delivers value immediately (stock vs option comparison)
- Validates entire technical stack end-to-end
- Provides foundation for all other stories
- De-risks architecture decisions early

### Incremental Delivery (Recommended)

For sustained development with multiple capabilities:

1. **Sprint 1** (Setup + Foundation): T001-T020 → Foundation ready
2. **Sprint 2** (US1 MVP): T021-T048 → Compare CLI working ✅
3. **Sprint 3** (US4 Screening): T049-T062 → Screen CLI working ✅
4. **Sprint 4** (US5 Conditional): T063-T070 → Conditional backtest working ✅
5. **Sprint 5** (US2 Grid): T071-T079 → Grid optimization working ✅
6. **Sprint 6** (US6 Conditional MC): T080-T087 → Full conditional suite ✅
7. **Sprint 7** (US7 Config + US8 Replay): T088-T117 → System mature ✅
8. **Sprint 8** (US3 Features + Polish): T099-T133 → Production ready ✅

Each sprint adds value without breaking previous features.

### Parallel Team Strategy

With 3+ developers:

1. **All developers**: Complete Setup + Foundational together (T001-T020)
2. **Once Foundational done**:
   - Developer A: US1 (T021-T048) - Critical path
   - Developer B: US4 (T049-T062) after US1 data layer (T021-T024)
   - Developer C: US2 (T071-T079) after US1 simulation (T038-T041)
3. **After US1 + US4 complete**:
   - Developer A: US7 (T088-T098)
   - Developer B: US5 (T063-T070) depends on US4
   - Developer C: US8 (T108-T117)
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
| SC-001: Baseline run ≤10s | T029-T032 (MC), T131 (benchmark) | US1 + Polish |
| SC-002: Artifacts include stock+option metrics | T042-T044 (artifacts) | US1 |
| SC-003: Grid produces ranked configs | T073-T074 (ranking) | US2 |
| SC-004: Reproducible with seed | T032 (seeding), T114 (replay) | US1 + US8 |
| SC-005: Clear errors for failures | T008 (exceptions), T047 (CLI errors) | Foundation + US1 |
| SC-006: Component swap via config | T088-T092 (config) | US7 |
| SC-007: Conditional MC reproducible | T084 (conditional MC), T087 (seeding) | US6 |
| SC-008: Conditional backtest ≥100 episodes | T063-T066 (conditional) | US5 |
| SC-009: Stock vs option conditional metrics | T065-T067 (episode metrics) | US5 |
| SC-010: Screen ≥100 symbols | T056-T058 (screening) | US4 |
| SC-011: Unconditional vs conditional reporting | T067 (comparison) | US5 |
| SC-012: Selector change without code | T052-T055 (selector), T091 (config) | US4 + US7 |
| SC-013: Parquet schema consistency | T019 (structure), T020a (validation), T020f (versioning), T023 (data validation) | Foundation + US1 |
| SC-014: Auto memmap fallback | T031 (memmap), T020 (policy) | Foundation + US1 |
| SC-015: run_meta.json complete | T044 (run_meta), T108 (provenance) | US1 + US8 |
| SC-016: Distribution swap changes behavior | T027 (factory), T093 (normal) | US1 + US7 |
| SC-017: Pricer swap changes option curves | T034 (factory), T094 (py_vollib) | US1 + US7 |
| SC-018: New strategy pluggable | T014 (interface), T035-T036 (examples) | Foundation + US1 |
| SC-019: Fit failure structured error | T008 (exceptions), T028 (validation) | Foundation + US1 |
| SC-020: Data gap warnings | T023 (validation), T059 (screening) | US1 + US4 |
| SC-021: Sufficient logs for debugging | T009 (logging), T044 (artifacts) | Foundation + US1 |
