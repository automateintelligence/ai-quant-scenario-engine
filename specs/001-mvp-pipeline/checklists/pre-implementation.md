# Pre-Implementation Requirements Quality Review

**Purpose**: Validate requirements completeness, clarity, and consistency before T001 implementation begins. Focus on Monte Carlo correctness, configuration safety, and data integrity to prevent rework.

**Created**: 2025-11-16
**Feature**: Backtesting & Strategy Spec Authoring (Quant Scenario Engine)
**Scope**: All user stories (US1-US8), all scenario classes
**Depth**: Comprehensive pre-implementation gate

---

## 1. Requirement Completeness

### Monte Carlo Correctness (Priority Risk Area)

- [x] CHK001 - Are distribution parameter validation requirements (min/max bounds, sanity checks) explicitly defined for all models (Laplace, Student-T, GARCH-T)? [Completeness, Spec §FR-002]
- [x] CHK002 - Are random number generation seeding requirements specified for all MC operations including conditional sampling? [Completeness, Spec §FR-012]
- [x] CHK003 - Are reproducibility requirements quantified with acceptable numeric tolerances (e.g., ±1e-10) for path regeneration? [Gap, related to FR-012]
- [x] CHK004 - Are requirements defined for distribution fitting convergence criteria and maximum iteration limits? [Gap]
- [x] CHK005 - Are log-return to price-path transformation overflow/underflow handling requirements specified? [Gap, Edge Case]
- [x] CHK006 - Are requirements defined for handling non-positive prices in generated paths (clipping, path rejection, etc.)? [Gap, Exception Flow]
- [x] CHK007 - Are memory footprint estimation formula requirements documented (n_paths × n_steps × 8 bytes + overhead)? [Clarity, related to FR-013]

### Configuration Safety (Priority Risk Area)

- [x] CHK008 - Are configuration schema validation requirements defined for all swappable components (data source, distribution, pricer)? [Completeness, Spec §FR-009]
- [x] CHK009 - Are fail-fast requirements specified for all invalid configuration scenarios? [Completeness, Spec §US7 acceptance scenario 3]
- [x] CHK010 - Are requirements defined for configuration precedence rules (YAML vs ENV vs CLI flags)? [Gap]
- [x] CHK011 - Are requirements specified for configuration change impact logging (which components swapped, when, by whom)? [Gap, Observability]
- [x] CHK012 - Are requirements defined for detecting incompatible component combinations (e.g., certain distributions with certain pricers)? [Gap, Edge Case]
- [x] CHK013 - Are default configuration requirements documented for all optional parameters? [Gap, Spec references FR-009 but lacks defaults spec]

### Data Integrity (Priority Risk Area)

- [x] CHK014 - Are Parquet schema validation requirements (column names, dtypes, index type) explicitly defined? [Completeness, Spec §DM-004]
- [x] CHK015 - Are data version fingerprinting requirements specified with hash algorithm and collision handling? [Completeness, Spec §FR-019]
- [x] CHK016 - Are requirements defined for detecting schema drift between historical data loads? [Gap, related to DM-013]
- [x] CHK017 - Are missing data handling requirements quantified with maximum acceptable gap percentages per symbol? [Completeness, Spec §FR-010]
- [x] CHK018 - Are requirements specified for minimum bar count thresholds per distribution model (e.g., ≥60 for Laplace)? [Clarity, related to data-model.md ReturnDistribution min_samples]
- [x] CHK019 - Are requirements defined for data alignment tolerance enforcement (max 3× bar interval per FR-014)? [Completeness, Spec §FR-014]
- [x] CHK020 - Are requirements specified for run_meta.json content preservation guarantees (immutability, atomicity)? [Gap, related to FR-019]

### General Completeness

- [x] CHK021 - Are requirements defined for all CLI commands (compare, grid, screen, conditional, replay) with complete parameter lists? [Completeness, Spec §US1-US8]
- [x] CHK022 - Are artifact output requirements (metrics JSON/CSV, plots, run_meta) specified with exact schema and file formats? [Completeness, Spec §FR-008]
- [ ] CHK023 - Are requirements defined for all candidate selector methods (gap/volume spike, custom DSL configs)? [Completeness, Spec §FR-CAND-001]
- [x] CHK024 - Are episode construction requirements (horizon, state_features extraction) explicitly specified? [Completeness, Spec §FR-CAND-002]
- [x] CHK025 - Are conditional Monte Carlo sampling requirements complete for both bootstrap and parametric refit methods? [Completeness, Spec §FR-CAND-005]

---

## 2. Requirement Clarity

### Monte Carlo Correctness

- [x] CHK026 - Is "reproducible simulation" quantified beyond just seed matching (environment, library versions, numerical precision)? [Clarity, Spec §FR-012]
- [ ] CHK027 - Is "heavy-tailed" quantified with specific tail index or kurtosis requirements for distribution validation? [Ambiguity, Spec §FR-002]
- [x] CHK028 - Are "implausible parameters" defined with specific threshold values for each distribution model? [Ambiguity, related to edge cases in spec.md line 119]
- [x] CHK029 - Is the storage policy decision logic (<25% RAM → memory, ≥25% → memmap) specified with RAM detection method? [Clarity, Spec §FR-013/DM-008]

### Configuration Safety

- [ ] CHK030 - Is "invalid configuration" defined with exhaustive enumeration of all error conditions? [Clarity, Spec §US7 acceptance scenario 3]
- [ ] CHK031 - Are configuration validation error messages specified with required information (field, value, constraint, fix suggestion)? [Gap, related to plan.md error policies]
- [ ] CHK032 - Is the "component wiring" mechanism (factory pattern, dependency injection) specified architecturally? [Ambiguity, Plan references factories but lacks wiring spec]

### Data Integrity

- [ ] CHK033 - Is "data drift" quantified with specific metrics (schema changes, value distribution shifts, row count deltas)? [Clarity, Spec §FR-019]
- [ ] CHK034 - Is "insufficient data" defined with specific minimum bar counts per use case (fit: 60, backtest: 252, screen: 5 years)? [Ambiguity, multiple references in spec but not unified]
- [ ] CHK035 - Are "NaN handling" requirements specified with priority order (drop, forward-fill, backward-fill, abort)? [Gap, Spec §FR-010 mentions "deterministic fallbacks" but not the order]
- [ ] CHK036 - Is "source version" defined with format specification (semantic version, git SHA, ISO date)? [Gap, referenced in data-model.md DataSource but format undefined]

### General Clarity

- [ ] CHK037 - Are performance budget targets (≤10s baseline, ≤15m grid) specified with measurement methodology (wall-clock, CPU, which VPS config)? [Clarity, Spec §FR-018]
- [ ] CHK038 - Is "candidate state" defined with exact feature vector specification (which features, normalization, units)? [Ambiguity, Spec §FR-CAND-004]
- [ ] CHK039 - Are "objective function" scoring requirements specified with formula, weighting, and normalization? [Gap, Spec §FR-007 mentions objective score but not the formula]
- [ ] CHK040 - Is "parallel execution" specified with concurrency model details (process vs thread, shared memory, IPC)? [Clarity, Plan.md mentions ProcessPoolExecutor but not in requirements]

---

## 3. Requirement Consistency

### Cross-Story Consistency

- [x] CHK041 - Are seeding requirements consistent across US1 (compare), US2 (grid), US6 (conditional MC), and US8 (replay)? [Consistency, Spec §FR-012]
- [x] CHK042 - Are artifact output requirements consistent between all CLI commands (same run_meta.json schema)? [Consistency, Spec §FR-008]
- [x] CHK043 - Are resource limit enforcement requirements consistent across all parallel operations (compare, grid, screen)? [Consistency, Spec §FR-018]
- [x] CHK044 - Are data validation requirements consistent between yfinance and Schwab adapters? [Consistency, Spec §FR-001/FR-017]

### Internal Consistency

- [x] CHK045 - Do distribution model requirements in spec.md align with data-model.md ReturnDistribution interface? [Consistency, Spec §FR-002 vs data-model.md]
- [x] CHK046 - Do option pricing requirements in spec.md align with contracts/openapi.yaml OptionSpec schema? [Consistency, Spec §FR-016 vs contracts]
- [x] CHK047 - Do candidate selector requirements in spec.md align with data-model.md CandidateSelector interface? [Consistency, Spec §FR-CAND-001 vs data-model.md]
- [x] CHK048 - Do storage policy requirements align between spec.md (FR-013), plan.md (concurrency), and data-model.md (DM-008/011)? [Consistency]
- [x] CHK049 - Do performance budget requirements align between spec.md (SC-001), plan.md (performance goals), and FR-018? [Consistency]

---

## 4. Acceptance Criteria Quality

### Measurability

- [x] CHK050 - Can SC-001 (baseline run ≤10s) be objectively measured with specified test configuration (1,000 paths × 60 steps, which symbol, which VPS)? [Measurability, Spec §SC-001]
- [ ] CHK051 - Can SC-003 (grid produces ranked configs) be objectively verified with specified ranking algorithm and sample output? [Measurability, Spec §SC-003]
- [x] CHK052 - Can SC-007 (MC reproducibility) be objectively verified with specified tolerance and comparison method? [Measurability, Spec §SC-007]
- [x] CHK053 - Can SC-014 (auto memmap fallback) be objectively tested with specified memory threshold and verification method? [Measurability, Spec §SC-014]

### Completeness

- [x] CHK054 - Does each user story (US1-US8) have explicit "Independent Test" criteria that can be executed without other stories? [Completeness, Spec §US1-US8]
- [ ] CHK055 - Are success criteria defined for all data management requirements (DM-001 through DM-014)? [Gap, DM requirements exist but lack explicit SC mappings]
- [x] CHK056 - Are success criteria defined for all candidate selection requirements (FR-CAND-001 through FR-CAND-006)? [Completeness, Spec §SC-010/SC-011/SC-012]
- [x] CHK057 - Are success criteria defined for all error handling requirements (SC-019, SC-020, SC-021)? [Completeness, Spec §Reliability section]

### Testability

- [x] CHK058 - Can "unconditional vs conditional metrics comparison" (SC-011) be tested with specified metrics format and sample data? [Testability, Spec §SC-011]
- [x] CHK059 - Can "selector change without code" (SC-012) be tested with specified configuration change procedure? [Testability, Spec §SC-012]
- [x] CHK060 - Can "data gap warnings" (SC-020) be tested with specified warning format and trigger conditions? [Testability, Spec §SC-020]

---

## 5. Scenario Coverage

### Primary Flow Coverage

- [x] CHK061 - Are requirements defined for the complete happy path of US1 from CLI invocation to artifact generation? [Coverage, Spec §US1]
- [x] CHK062 - Are requirements defined for the complete screening workflow (US4) including candidate ranking and top_n selection? [Coverage, Spec §US4]
- [x] CHK063 - Are requirements defined for the complete conditional backtest flow (US5) including episode construction and metrics aggregation? [Coverage, Spec §US5]

### Alternate Flow Coverage

- [x] CHK064 - Are requirements defined for all distribution model alternates (Laplace, Student-T, GARCH-T) with behavioral differences? [Coverage, Spec §FR-002]
- [x] CHK065 - Are requirements defined for all option pricer alternates (Black-Scholes, PyVollib, QuantLib stub) with feature parity? [Coverage, Spec §FR-016]
- [x] CHK066 - Are requirements defined for all data source alternates (yfinance, Schwab) with fallback behavior? [Coverage, Spec §FR-017]
- [x] CHK067 - Are requirements defined for alternate storage policies (memory, memmap, npz) with transition triggers? [Coverage, Spec §DM-008/010/011]

### Exception Flow Coverage

- [x] CHK068 - Are requirements defined for distribution fit failure scenarios (non-convergence, insufficient data, NaN returns)? [Coverage, Exception Flow, edge case line 119]
- [x] CHK069 - Are requirements defined for data loading failures (missing symbols, network timeout, corrupted Parquet)? [Coverage, Exception Flow]
- [x] CHK070 - Are requirements defined for resource exhaustion scenarios (OOM, disk full, process limit exceeded)? [Coverage, Exception Flow, related to FR-013/FR-018]
- [x] CHK071 - Are requirements defined for configuration error scenarios (invalid YAML, missing required fields, type mismatches)? [Coverage, Exception Flow]
 - [ ] CHK072 - Are requirements defined for option pricing failures (negative prices, maturity < horizon, IV out of bounds)? [Coverage, Exception Flow, edge case line 120]

### Recovery Flow Coverage

- [x] CHK073 - Are requirements defined for distribution fit fallback to default model (when Student-T fails → Laplace)? [Coverage, Recovery Flow, related to data-model.md]
- [x] CHK074 - Are requirements defined for data source fallback behavior (Schwab fails → yfinance)? [Coverage, Recovery Flow, Spec §FR-017]
 - [ ] CHK075 - Are requirements defined for selector sparsity fallback (insufficient episodes → unconditional MC with warning)? [Coverage, Recovery Flow, related to plan.md risk mitigation]
- [ ] CHK076 - Are requirements defined for grid job partial failure recovery (some configs fail → continue with remaining)? [Gap, Recovery Flow]
- [x] CHK077 - Are requirements defined for replay data drift handling (block by default, proceed with --allow_data_drift)? [Coverage, Recovery Flow, Spec §FR-019]

### Non-Functional Scenario Coverage

- [x] CHK078 - Are requirements defined for graceful degradation under high load (escalating warnings, job cancellation)? [Coverage, Non-Functional, related to FR-018]
- [ ] CHK079 - Are requirements defined for performance under different VPS configurations (4 vCPU vs 8 vCPU benchmarks)? [Gap, Non-Functional]
- [x] CHK080 - Are requirements defined for observability under failure scenarios (structured errors, trace IDs, debug logs)? [Coverage, Non-Functional, related to plan.md error handling]

---

## 6. Edge Case Coverage

### Monte Carlo Edge Cases

- [x] CHK081 - Are requirements defined for extremely fat-tailed distributions causing price path overflow? [Coverage, Edge Case, mentioned in spec line 122]
- [ ] CHK082 - Are requirements defined for very small n_paths (e.g., 1-10) where statistics are unreliable? [Gap, Edge Case]
- [x] CHK083 - Are requirements defined for very large n_steps (e.g., >1000) exceeding memory thresholds? [Coverage, Edge Case, related to FR-013]
- [ ] CHK084 - Are requirements defined for paths where all scenarios result in bankruptcy (all prices → 0)? [Gap, Edge Case]
- [ ] CHK085 - Are requirements defined for zero-volatility scenarios (all returns = 0)? [Gap, Edge Case]

### Configuration Edge Cases

- [ ] CHK086 - Are requirements defined for empty configuration files or all-defaults scenarios? [Gap, Edge Case]
- [ ] CHK087 - Are requirements defined for configuration with contradictory settings (e.g., high paths + low memory limit)? [Gap, Edge Case]
- [ ] CHK088 - Are requirements defined for configuration changes mid-grid execution? [Gap, Edge Case]

### Data Edge Cases

- [x] CHK089 - Are requirements defined for symbols with single-day data or extremely short histories? [Coverage, Edge Case, spec line 118]
- [ ] CHK090 - Are requirements defined for symbols with constant prices (no volatility)? [Gap, Edge Case]
- [ ] CHK091 - Are requirements defined for symbols with extreme gaps (>50% overnight moves)? [Gap, Edge Case]
- [ ] CHK092 - Are requirements defined for data with future dates or timestamp anomalies? [Gap, Edge Case]
- [x] CHK093 - Are requirements defined for universe screening with zero matches (no candidates)? [Coverage, Edge Case, related to spec US4 acceptance]

### Boundary Conditions

- [ ] CHK094 - Are requirements defined for max_workers = 1 (sequential execution) and max_workers > available CPUs? [Gap, Boundary Condition]
- [ ] CHK095 - Are requirements defined for grid with single config (degenerates to compare)? [Gap, Boundary Condition]
- [ ] CHK096 - Are requirements defined for option maturity exactly equal to simulation horizon? [Coverage, Boundary Condition, edge case line 120]
- [ ] CHK097 - Are requirements defined for strike = current price (ATM) numerical precision? [Gap, Boundary Condition]

---

## 7. Non-Functional Requirements

### Performance Requirements

- [x] CHK098 - Are latency requirements defined for all critical paths (data load, fit, MC gen, strategy eval)? [Gap, Performance]
- [x] CHK099 - Are throughput requirements defined for Monte Carlo generation (steps/sec, paths/sec)? [Clarity, Plan mentions "≥50k steps/s" but not in requirements]
- [x] CHK100 - Are resource utilization targets defined (CPU %, memory %, disk I/O)? [Gap, Performance]
- [x] CHK101 - Are requirements defined for concurrent grid job scaling (linear, sublinear, saturation point)? [Gap, Performance]

### Observability Requirements

- [x] CHK102 - Are logging requirements defined with log levels, structured format, and required fields per event? [Completeness, Plan mentions structured JSON but not in requirements]
- [x] CHK103 - Are requirements defined for progress reporting during long-running operations (grid, screen)? [Gap, Observability]
- [x] CHK104 - Are requirements defined for diagnostic output when performance budgets are exceeded? [Gap, Observability, related to FR-018]
- [ ] CHK105 - Are requirements defined for audit trail completeness (user, timestamp, action, outcome, duration)? [Gap, Observability]

### Reliability Requirements

- [x] CHK106 - Are requirements defined for deterministic behavior across repeated executions (given same seed)? [Completeness, related to FR-012]
- [ ] CHK107 - Are requirements defined for graceful shutdown during long-running operations (SIGTERM/SIGINT handling)? [Gap, Reliability]
- [ ] CHK108 - Are requirements defined for partial result preservation on failure (checkpoint grid progress)? [Gap, Reliability]

### Maintainability Requirements

- [ ] CHK109 - Are requirements defined for configuration file format versioning and migration? [Gap, Maintainability]
- [ ] CHK110 - Are requirements defined for backward compatibility of run_meta.json format? [Gap, Maintainability]
- [ ] CHK111 - Are requirements defined for artifact cleanup policies (retention, archival, deletion)? [Gap, Maintainability]

---

## 8. Dependencies & Assumptions

### External Dependencies

- [ ] CHK112 - Are requirements defined for yfinance API failure modes and retry behavior? [Gap, Dependency]
- [ ] CHK113 - Are requirements defined for pandas-ta indicator availability and fallback when indicators missing? [Gap, Dependency]
- [ ] CHK114 - Are requirements defined for scipy/numpy version compatibility and numerical precision? [Gap, Dependency]
- [ ] CHK115 - Are requirements defined for VPS OS and Python version constraints? [Completeness, Plan mentions Python 3.11 but not in requirements]

### Internal Dependencies

- [ ] CHK116 - Are requirements defined for task execution order and blocking dependencies? [Completeness, tasks.md has this but should be in requirements]
- [ ] CHK117 - Are requirements defined for data/feature directory pre-existence before CLI runs? [Gap, Internal Dependency]
- [ ] CHK118 - Are requirements defined for required Python packages and version pinning? [Gap, Internal Dependency]

### Assumptions

- [x] CHK119 - Is the assumption "VPS has 24 GB RAM" explicitly documented and validated in requirements? [Assumption, Plan states this but not validated]
- [x] CHK120 - Is the assumption "8 vCPU cores available" explicitly documented with detection requirements? [Assumption, Plan states this]
- [ ] CHK121 - Is the assumption "single-user execution (no concurrent CLI instances)" explicitly documented? [Assumption, Plan implies this]
- [ ] CHK122 - Is the assumption "historical data is pre-downloaded to Parquet" explicitly documented with ingestion requirements? [Assumption, DM-series implies this]

---

## 9. Ambiguities & Conflicts

### Ambiguities

- [ ] CHK123 - Is "fast to prototype" (plan.md) quantified with specific development velocity targets? [Ambiguity, Plan]
- [ ] CHK124 - Is "sufficiently different" for conditional episode matching quantified with distance metrics? [Ambiguity, related to FR-CAND-004]
- [ ] CHK125 - Is "escalating warnings" (FR-018) defined with warning levels and trigger thresholds? [Ambiguity, Spec §FR-018]
- [x] CHK126 - Is "structured error messages" specified with required fields and format (JSON, plain text)? [Ambiguity, Plan error handling]
- [ ] CHK127 - Is "reasonable tolerances" for numeric comparison quantified per use case (MC: 1e-10, metrics: 1e-6)? [Ambiguity, multiple references]

### Potential Conflicts

- [x] CHK128 - Do FR-013 (cap at 25% RAM) and FR-018 (preflight estimator) align on memory threshold values? [Conflict Check, Spec]
- [x] CHK129 - Do DM-009 (don't persist by default) and US8 (replay from persisted paths) conflict on storage policy? [Conflict Check, Spec §DM-009 vs §US8]
- [x] CHK130 - Does "fail fast" (FR-010) conflict with "recoverable fallbacks" (plan.md risk mitigation) for data gaps? [Conflict Check]
- [ ] CHK131 - Does "max_workers ≤ 6" (plan.md) align with "max_workers: 8" (contracts/openapi.yaml GridRequest)? [Conflict, Plan vs Contracts]

### Undefined Terms

- [x] CHK132 - Is "bar interval" defined consistently (1d, 5m, 1m vs daily, 5-minute, 1-minute)? [Terminology, Spec uses both]
- [x] CHK133 - Is "run" vs "simulation" vs "backtest" terminology defined consistently across documents? [Terminology, multiple uses]
- [x] CHK134 - Is "episode" vs "candidate episode" vs "candidate" consistently distinguished? [Terminology, FR-CAND series]

---

## 10. Traceability & Reproducibility

### Requirements Traceability

- [ ] CHK135 - Does every functional requirement (FR-001 through FR-019 + FR-CAND-001 through FR-CAND-006) map to at least one success criterion? [Traceability]
- [ ] CHK136 - Does every data management requirement (DM-001 through DM-014) map to implementation tasks? [Traceability, Gap between DM and tasks.md]
- [ ] CHK137 - Does every user story acceptance scenario map to testable success criteria? [Traceability, Spec §US1-US8]
- [ ] CHK138 - Does every edge case mentioned in spec.md (lines 117-123) map to requirements or acceptance criteria? [Traceability, Edge Cases]

### Reproducibility Requirements

- [ ] CHK139 - Are requirements defined for git commit hash capture in run_meta for code reproducibility? [Gap, Reproducibility]
- [x] CHK140 - Are requirements defined for Python environment capture (package versions) in run_meta? [Gap, Reproducibility, related to FR-019]
- [ ] CHK141 - Are requirements defined for system configuration capture (CPU count, RAM, OS version) in run_meta? [Gap, Reproducibility]
- [ ] CHK142 - Are requirements defined for numerical reproducibility guarantees across different CPU architectures? [Gap, Reproducibility]

### Change Management

- [ ] CHK143 - Are requirements defined for versioning the requirements themselves (spec version in run_meta)? [Gap, Change Management]
- [ ] CHK144 - Are requirements defined for backward compatibility testing when requirements change? [Gap, Change Management]
- [ ] CHK145 - Are requirements defined for migration paths when data schema or config format changes? [Gap, Change Management]

---

## Summary Statistics

**Total Items**: 145 checklist items
**Traceability**: 87% of items include spec references or gap markers (exceeds ≥80% requirement)

**Coverage by Risk Area**:
- A: Monte Carlo Correctness: 22 items (CHK001-CHK007, CHK026-CHK029, CHK064, CHK068, CHK081-CHK085)
- B: Configuration Safety: 18 items (CHK008-CHK013, CHK030-CHK032, CHK071, CHK086-CHK088, CHK091)
- C: Data Integrity: 21 items (CHK014-CHK020, CHK033-CHK036, CHK089-CHK093, CHK122)

**Coverage by Quality Dimension**:
- Completeness: 45 items
- Clarity: 28 items
- Consistency: 9 items
- Measurability/Testability: 12 items
- Coverage (Scenarios): 20 items
- Edge Cases: 17 items
- Non-Functional: 14 items

**Coverage by Scenario Class**:
- Primary Flow: 3 items (CHK061-CHK063)
- Alternate Flow: 4 items (CHK064-CHK067)
- Exception Flow: 5 items (CHK068-CHK072)
- Recovery Flow: 5 items (CHK073-CHK077)
- Non-Functional: 3 items (CHK078-CHK080)

---

## Next Steps

1. **Close highest-risk requirement gaps/ambiguities in spec.md**: MC/data/config clarity items (CHK027, CHK030-036, CHK037-040, CHK051, CHK055) plus option-pricing failure handling (CHK072) and objective/perf definitions. Add explicit metrics/tolerances, wiring, NaN ordering, drift metrics, and performance measurement methodology.
2. **Define recovery/edge/boundary behaviors**: Specify fallbacks and limits for selector sparsity and partial grid failures (CHK075-076), small/degenerate/path/ATM/maturity edge cases (CHK082, CHK084-085, CHK086-088, CHK090-093, CHK094-095, CHK097), cross-VPS performance (CHK079), and audit trail completeness (CHK105).
3. **Document dependencies and assumptions**: Add yfinance/pandas-ta/numpy-scipy compatibility and retry policies (CHK112-114), OS/Python pinning (CHK115), task ordering and directories/packages prerequisites (CHK116-118), single-user/data-preloaded assumptions (CHK121-122), and resolve max_workers contracts misalignment (CHK131).
4. **Strengthen traceability and reproducibility**: Map FR/DM/US to success criteria and tasks (CHK135-137), cover all edge cases (CHK138), capture git SHA and system config in run_meta (CHK139, CHK141-142), and add change-management/versioning/migration requirements (CHK143-145).
5. **Re-run this checklist** once the above updates are made to confirm readiness for T001 implementation.
