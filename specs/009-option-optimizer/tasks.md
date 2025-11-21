# Tasks: Option Strategy Optimizer (009-option-optimizer)

**Input**: Design documents from `/specs/009-option-optimizer/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Include targeted tests where the spec calls out independent verification (runtime, diagnostics, pricing fallbacks, monitoring).

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup (Shared Infrastructure)

- [X] T001 Verify local environment has Python 3.11 dependencies installed for data clients and CLI (`pip install -e .[dev]`) (plan.md tech stack)
- [X] T002 [P] Create docs/data-sources/README.md capturing available data providers (Schwab primary, yfinance fallback) per plan.md (research Decision 1/2)

---

## Phase 2: Foundational (Blocking Prerequisites)

- [X] T003 Parse Schwab Trader API PDFs (`docs/*.pdf`) and planning/yfinance_specs.md; document REST contracts (quotes, option chain, history, fundamentals, analyst info, market data) in docs/data-sources/schwab_contract.md with request/response fields, auth, limits, and gaps (spec FR-004/FR-005, US1/US2 inputs)
- [X] T004 [P] Define data provider interface and config schema updates (schwab primary, yfinance fallback, timeouts/retries) in `src/qse/data/__init__.py` and config comments (spec FR-056/FR-058)
- [X] T005 [P] Implement SchwabDataSource client for quotes/option chain/OHLCV/fundamentals/analyst endpoints with authenticated HTTP wrapper and error normalization in `src/qse/data/schwab.py` (spec FR-004/FR-005)
- [X] T006 [P] Implement fallback chaining wrapper that calls Schwab then yfinance with warning on fallback in `src/qse/data/factory.py` and `src/qse/data/data_loader.py` (spec FR-005 Edge: missing data)
- [X] T007 Add unit tests for data providers and fallback behavior in `tests/unit/data/test_schwab.py` and extend `tests/unit/data/test_yfinance.py` (spec FR-005, Edge cases)
- [X] T008 Update cache/loader paths and validation to support option chain snapshots (bid/ask/IV/OI/volume) in `src/qse/data/data_loader.py` (spec FR-004/FR-007/FR-023)
- [X] T009 [P] Wire CLI fetch command to accept `--data-source schwab` with fallback telemetry in `src/qse/cli/commands/fetch.py` (spec FR-056/FR-059)
- [X] T010 Establish parallel agent/cloud execution plan for [P] tasks (docs/data-sources/parallel-runbook.md) to maximize throughput (user request)

---

## Phase 3: User Story 1 - Single-Command Strategy Optimization (Priority: P1) 🎯 MVP

**Goal**: Optimize and return Top-10 strategies from ticker+regime+horizon (full universe may take up to 1 hour).
**Independent Test**: Re-evaluate an existing Top-10 list with fresh market data in <30s via `qse optimize-strategy --ticker NVDA --regime strong-bullish --trade-horizon 1 --retest top10.json` (spec FR-061 adjusted).

- [X] T011 [US1] Connect CLI optimize-strategy to new data provider selection (schwab primary, yfinance fallback) in `src/qse/cli/commands/optimize.py` (spec FR-001/FR-004/FR-059)
- [X] T012 [P] [US1] Ensure overrides flow (mc.*, filters.*, scoring.*) from CLI to config application in `src/qse/config/loader.py` (spec FR-003/FR-058)
- [X] T013 [US1] Emit Top-10/Top-100 artifacts and diagnostics per spec and retest path (<30s) in `src/qse/optimizers/strategy_optimizer.py` (spec FR-048–FR-055, FR-061)

---

## Phase 4: User Story 2 - Multi-Stage Candidate Filtering (Priority: P1)

**Goal**: Stage 0-4 filtering reduces thousands to 50–200 MC survivors; full sweep can run up to 1 hour; cached Top-10 retests <30s.
**Independent Test**: Synthetic chain run logs stage counts and runtime; retest path meets <30s.

- [X] T014 [P] [US2] Implement Stage 0 expiry selection (3–5 expiries in [7,45] DTE) in `src/qse/optimizers/candidate_filter.py` (spec FR-006)
- [X] T015 [P] [US2] Implement Stage 1 moneyness/liquidity strike filters with config thresholds in `src/qse/optimizers/candidate_filter.py` (spec FR-007)
- [X] T016 [US2] Implement Stage 2 structure generation (verticals width 1–3, iron condors, straddles/strangles) in `src/qse/optimizers/candidate_generator.py` (spec FR-008)
- [X] T017 [US2] Implement Stage 3 analytic prefilter + hard constraints + top-K per structure in `src/qse/optimizers/prefilter.py` (spec FR-009–FR-011)
- [X] T018 [US2] Integrate Stage 4 MC scoring trigger for survivors in `src/qse/optimizers/strategy_optimizer.py` (spec FR-012)

---

## Phase 5: User Story 3 - Regime-Driven Distribution Selection (Priority: P1)

**Goal**: Regime labels map to distribution params across table/calibrated/explicit modes; multi-day compounding.
**Independent Test**: Regime strong-bullish loads parameters and compounds over trade_horizon=3.

- [X] T019 [P] [US3] Implement regime loader supporting table/calibrated/explicit in `src/qse/distributions/regime_loader.py` (spec FR-013/FR-014)
- [X] T020 [US3] Propagate trade_horizon and bars_per_day to path generation for compounding in `src/qse/distributions/path_generator.py` (spec FR-015/FR-025)
- [X] T021 [P] [US3] Add CLI/config validation for unknown regimes and mode selection in `src/qse/config/validation.py` (spec FR-002/FR-014)

---

## Phase 6: User Story 4 - Pluggable Pricing Models (Priority: P2)

**Goal**: OptionPricer interface with Bjerksund default, BS fallback, Heston hook; IV evolution modes.
**Independent Test**: --pricing-model black_scholes vs bjerksund runs both successfully; fallback emits warning.

- [X] T022 [US4] Finalize OptionPricer interface and IV evolution modes (constant, sticky-delta, custom) in `src/qse/pricing/interfaces.py` (spec FR-017/FR-019)
- [X] T023 [P] [US4] Implement BjerksundStenslandPricer + fallback to BlackScholes in `src/qse/pricing/bjerksund.py` and `src/qse/pricing/black_scholes.py` (spec FR-018/FR-021/FR-022)
- [X] T024 [P] [US4] Add pricer factory and config wiring (shared with US1) in `src/qse/pricing/factory.py` (spec FR-071/FR-073)
- [X] T025 [US4] Add unit tests covering convergence failure fallback in `tests/unit/pricing/test_pricers.py` (spec FR-021/FR-022)

---

## Phase 7: User Story 5 - Composite Strategy Scoring (Priority: P2)

**Goal**: Pluggable StrategyScorer with intraday-spreads default and score decomposition.
**Independent Test**: Weight changes shift rankings; score decomposition printed.

- [X] T026 [P] [US5] Implement StrategyScorer interface + plugin loader in `src/qse/scorers/base.py` (spec FR-034/FR-040)
- [X] T027 [US5] Implement intraday-spreads scorer with normalization/penalties in `src/qse/scorers/intraday_spreads.py` (spec FR-035–FR-039)
- [X] T028 [US5] Add score decomposition output to optimizer results in `src/qse/optimizers/strategy_optimizer.py` (spec FR-041)
- [X] T029 [P] [US5] Add unit tests for weight override and ranking changes in `tests/unit/scorers/test_intraday_spreads.py` (spec SC-005)

---

## Phase 8: User Story 6 - Transaction Cost Modeling (Priority: P2)

**Goal**: Apply pay-spread fills + $0.65/contract commissions; filters/scoring on net P&L.
**Independent Test**: Iron Condor example computes entry/exit costs and net E[PnL].

- [X] T030 [US6] Implement pay-spread fill and commission model applied at entry/exit in `src/qse/optimizers/costs.py` (spec FR-042–FR-044)
- [X] T031 [P] [US6] Wire cost model into prefilter and full MC metrics in `src/qse/optimizers/prefilter.py` and `src/qse/optimizers/strategy_optimizer.py` (spec FR-045/FR-046)
- [X] T032 [US6] Add tests for cost application and overrides in `tests/unit/optimizers/test_costs.py` (spec FR-047)

---

## Phase 9: User Story 7 - Confidence Intervals & Diagnostics (Priority: P3)

**Goal**: Report 95% CIs for E[PnL]/POP; adaptive paths up to cap; diagnostics for empty results.
**Independent Test**: Adaptive paths double when CI width > threshold; empty filters produce rejection breakdown.

- [X] T033 [P] [US7] Implement CI computation and adaptive path loop with cap in `src/qse/optimizers/metrics.py` (spec FR-032/FR-033)
- [X] T034 [US7] Add diagnostics bundle (stage counts, rejection breakdown, hints) in `src/qse/optimizers/diagnostics.py` (spec FR-054/FR-055/FR-075)
- [X] T035 [P] [US7] Add tests for adaptive path behavior and empty-result diagnostics in `tests/unit/optimizers/test_diagnostics.py` (spec SC-007/SC-010)

---

## Phase 10: User Story 8 - Live Position Monitoring (Priority: P3)

**Goal**: Monitor exported position, reprice remaining horizon, trigger alerts.
**Independent Test**: `qse monitor --position position.json --interval 300` reprices and alerts on thresholds.

- [ ] T036 [US8] Implement position JSON schema + loader (legs, config snapshot, alerts) in `src/qse/monitoring/position.py` (spec FR-063/FR-064)
- [ ] T037 [P] [US8] Implement monitoring loop (fetch data, reprice with OptionPricer, simulate remaining horizon) in `src/qse/monitoring/monitor.py` (spec FR-065–FR-069)
- [ ] T038 [US8] Add CLI command for monitor with alert outputs in `src/qse/cli/commands/monitor.py` (spec FR-064/FR-067)
- [ ] T039 [P] [US8] Add integration test with stub data for alert triggering in `tests/integration/test_monitor.py` (spec FR-067/FR-070)

---

## Phase 11: Polish & Cross-Cutting Concerns

- [ ] T040 [P] Update quickstart.md with Schwab primary/fallback workflows in `specs/009-option-optimizer/quickstart.md` (spec FR-056/Quickstart)
- [ ] T041 Add performance/runbook notes for 1-hour full sweep + <30s retest and batch 10 tickers to `docs/runbooks/option-optimizer.md` (spec SC-001/SC-012)
- [ ] T042 [P] Final lint/test sweep (`ruff check .`, `pytest`) and capture known issues

---

## Phase 12: Quality, Contracts, and Resilience

- [X] T043 [P] Update `specs/009-option-optimizer/spec.md` FR-061 language to distinguish full-sweep runtime (≤1 hour) vs retest mode (<30s) and align quickstart examples (spec FR-061)
- [X] T044 [P] Define adaptive path thresholds (CI width, doubling policy, max cap handling) in config defaults and implement in `qse/optimizer/metrics.py` with docstring references (spec FR-032/FR-033)
- [ ] T045 [P] Add contract tests for optimize-strategy and monitor responses against `contracts/openapi-qse.yaml` in `tests/contract/test_optimize_contract.py` and `tests/contract/test_monitor_contract.py` (contracts alignment, FR-048–FR-055)
- [ ] T046 [P] Add resilience tests covering Schwab outage/fallback and pricer convergence failures in `tests/integration/test_resilience.py` (Edge cases, FR-005, FR-021/FR-022)
- [ ] T047 [P] Add property-based tests (hypothesis) for candidate filtering invariants (monotonicity of filters, width limits) in `tests/property/test_candidate_filtering.py` (FR-006–FR-011 robustness)
- [ ] T048 Enforce coverage/quality gate (e.g., pytest-cov threshold) in CI config and document in `docs/runbooks/option-optimizer.md` (Constitution Section II.V, testing discipline)

---

## Dependencies & Execution Order

- Foundational (Phase 2) blocks all user stories; T003–T010 must complete first to verify Schwab contracts and data pipeline.
- US1 depends on data provider wiring (T011–T013) and override/config handling.
- US2 depends on Stage 0–3 pipeline (T014–T018) before MC scoring.
- US3 depends on regime loader and propagation (T019–T021).
- US4 supplies pricers consumed by US1/US2/US8.
- US5 scoring plugs into optimizer after metrics; depends on US2/US4.
- US6 cost model feeds prefilter + MC; depends on US2 pipeline.
- US7 diagnostics depend on metrics/scoring outputs.
- US8 monitoring depends on pricer, data provider, and metrics modules.
- Quality/Contracts/Resilience tasks (T043–T048) should follow core implementations but can run in parallel where marked [P]; ensure spec/quickstart updated before final sweep.

### Parallel Opportunities

- Tasks marked [P] should leverage parallel agents/cloud execution per T010 to reduce wall time.
- Within stories, [P] tasks can proceed concurrently once their prerequisites are complete.
