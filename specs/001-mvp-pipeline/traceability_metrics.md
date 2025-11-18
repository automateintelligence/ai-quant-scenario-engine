# Traceability Matrices

### Data Management Requirements → Success Criteria Mapping

| DM Requirement | Success Criteria | Verification Method |
|---|---|---|
| DM-001 (Daily bars) | SC-001 (baseline run completes) | Run with daily data, verify completion |
| DM-002 (5-min bars) | SC-001 (run with 5-min interval) | CLI with `--interval 5m` |
| DM-003 (1-min bars) | SC-001 (run with 1-min interval) | CLI with `--interval 1m` |
| DM-004 (Parquet format) | SC-013 (schema preserved) | Export/import cycle test |
| DM-005 (feature separation) | SC-013 (features in separate files) | Check `data/features/` directory |
| DM-006 (directory layout) | SC-045 (auto-create directories) | First run creates structure |
| DM-007 (universe tiers) | SC-010 (screen ≥100 symbols) | Screen command on SP500 universe |
| DM-008 (in-memory default) | SC-014 (auto memmap fallback) | Run with <25% RAM usage |
| DM-009 (reproducibility without paths) | SC-007 (seeded reproducibility) | Repeat run with same seed |
| DM-010 (persistent MC conditions) | SC-014 (memmap when >25% RAM) | Large MC run triggers persistence |
| DM-011 (memmap for >50% RAM) | SC-014 (memmap fallback) | Very large MC run uses memmap |
| DM-012 (MC storage format) | SC-014 (.npz or memmap) | Check artifact format |
| DM-013 (historical retention) | SC-027 (drift detection on replay) | Replay after data change |
| DM-014 (MC ephemeral) | SC-015 (run_meta sufficient) | Verify no MC files by default |
| DM-015 (schema validation) | SC-026 (schema drift detection) | Load drifted Parquet file |
| DM-016 (data fingerprinting) | SC-027 (drift blocks replay) | Replay with changed data |
| DM-017 (missing data tolerances) | SC-020 (data gap warnings) | Run with gapped data |
| DM-018 (metadata durability) | SC-015 (run_meta complete) | Verify atomic writes |

### Functional Requirements → Success Criteria Mapping

| FR Requirement | Success Criteria | Notes |
|---|---|---|
| FR-001 (OHLCV loading) | SC-068, SC-069, SC-070 | On-demand fetch, staleness, incremental |
| FR-002 (distribution fitting) | SC-019 (fit failure handling) | Laplace default, convergence |
| FR-003 (stock strategy) | SC-002 (stock metrics in output) | Equity curves + metrics |
| FR-004 (option strategy) | SC-002 (option metrics in output) | Option pricing + P&L |
| FR-005 (CLI interface) | SC-001 (baseline run completes) | All CLI commands |
| FR-006 (feature injection) | SC-002 (indicators in features) | Pandas-ta integration |
| FR-007 (grid execution) | SC-003 (ranked configs) | Parallel grid jobs |
| FR-008 (artifacts) | SC-002, SC-015 (output artifacts) | JSON/CSV + run_meta |
| FR-009 (config swapping) | SC-006 (component swap via config) | Data source, pricer swap |
| FR-010 (missing data) | SC-020 (data gap warnings) | Deterministic fallbacks |
| FR-011 (stubs) | SC-006 (yfinance default) | Schwab stub, pricer stubs |
| FR-012 (seeding) | SC-004, SC-007 (reproducibility) | Deterministic RNG |
| FR-013 (memory cap) | SC-014 (auto memmap fallback) | 25% RAM threshold |
| FR-014 (macro alignment) | Not explicitly tested in SC | Feature alignment |
| FR-015 (performance targets) | SC-001 (baseline ≤10s) | VPS benchmarks |
| FR-016 (option pricing) | SC-002, SC-036, SC-073 | BS + IV fallback, ATM, maturity=horizon |
| FR-017 (data sources) | SC-006, SC-068 (yfinance/Schwab) | Fallback chain |
| FR-018 (resource limits) | SC-001 (time budgets) | Grid ≤15min |
| FR-019 (replay drift) | SC-027 (drift blocks replay) | Data version check |
| FR-020 (distribution validation) | SC-019 (fit failure) | Parameter bounds |
| FR-021 (reproducibility) | SC-046, SC-047, SC-048 | Cross-run, cross-arch |
| FR-022 (overflow handling) | SC-031 (bankruptcy error) | Negative price rejection |
| FR-023 (memory estimator) | SC-014 (preflight check) | Storage policy |
| FR-024 (config precedence) | Logged in run_meta | CLI > ENV > YAML |
| FR-025 (config validation) | SC-022 (fail fast) | Invalid config error |
| FR-026 (component logging) | Logged in run_meta | Audit trail |
| FR-027 (schema validation) | SC-026 (schema drift) | Parquet schema check |
| FR-028 (fingerprinting) | SC-027 (drift detection) | SHA256 hashing |
| FR-029 (missing tolerances) | SC-020 (gap warnings) | 3× bar interval |
| FR-030 (atomic writes) | SC-015 (metadata integrity) | run_meta durability |
| FR-031 (storage policy) | SC-014 (persistence flag) | Default non-persistent |
| FR-032 (minimum samples) | SC-019 (fit failure) | ≥60 bars |
| FR-033 (CLI validation) | SC-022 (param validation) | OpenAPI contract |
| FR-034 (artifact formats) | SC-002, SC-015 (schemas) | Metrics + run_meta |
| FR-035 (episode capture) | SC-010 (candidate list) | Episode metadata |
| FR-036 (conditional MC methods) | SC-008 (conditional backtest) | Bootstrap + refit |
| FR-037 (parameter thresholds) | SC-019 (implausible params) | Per-model bounds |
| FR-038 (fail-fast default) | SC-019 (structured errors) | Abort on invalid |
| FR-039 (structured logging) | SC-021 (trace logs) | JSON logs |
| FR-040 (performance budgets) | SC-001 (threshold warnings) | Tiered INFO/WARN/ERROR |
| FR-041 (invalid config) | SC-022 (7 error conditions) | Exhaustive enumeration |
| FR-042 (error messages) | SC-022 (field/value/fix) | Required fields |
| FR-043 (component wiring) | SC-006 (factory pattern) | Registry-based |
| FR-044 (config precedence) | Logged in run_meta | Override tracking |
| FR-045 (defaults) | SC-024 (empty config) | Built-in defaults |
| FR-046 (drift quantification) | SC-027 (3 drift types) | Schema/distribution/count |
| FR-047 (NaN handling) | SC-028 (4-step priority) | Drop/forward/backward/abort |
| FR-048 (source version format) | SC-046 (run_meta capture) | provider_ver_date_sha |
| FR-049-087 (edge cases) | SC-030-038, SC-073 | Boundary conditions |
| FR-CAND-001 (selectors) | SC-010 (screening) | Gap/volume + DSL |
| FR-CAND-002 (episodes) | SC-010 (episode construction) | symbol, t0, horizon |
| FR-CAND-003 (conditional backtest) | SC-011 (metrics comparison) | Episode-level metrics |
| FR-CAND-004 (conditional MC) | SC-008 (state conditioning) | Distance metrics |
| FR-CAND-005 (sampling methods) | SC-009 (bootstrap + refit) | Non-parametric + parametric |
| FR-CAND-006 (default selector) | SC-012 (selector swap) | Gap/volume default |

### User Story Acceptance Scenarios → Success Criteria Mapping

| User Story | Acceptance Scenario | Success Criteria |
|---|---|---|
| US1 (compare) | US1-1 (baseline run) | SC-001, SC-002, SC-004 |
| US1 (compare) | US1-2 (distribution swap) | SC-006, SC-016 |
| US2 (grid) | US2-1 (grid metrics) | SC-003, SC-007 |
| US2 (grid) | US2-2 (parallel execution) | SC-003, SC-018 (resource limits) |
| US3 (features) | US3-1 (indicator injection) | SC-002 (features in output) |
| US3 (features) | US3-2 (missing macro) | SC-020 (warnings) |
| US4 (screening) | US4-1 (candidate list) | SC-010 |
| US4 (screening) | US4-2 (missing data) | SC-020 |
| US4 (screening) | US4-3 (selector swap) | SC-012 |
| US5 (strategy screen) | US5-1 (unconditional ranking) | SC-003, SC-011 |
| US5 (strategy screen) | US5-2 (conditional filtering) | SC-011 |
| US5 (strategy screen) | US5-3 (Mode A selector-only) | SC-010 |
| US5 (strategy screen) | US5-4 (sparse episodes) | SC-011 (low-confidence flag) |
| US5 (strategy screen) | US5-5 (selector change) | SC-012 |
| US5 (strategy screen) | US5-6 (mode distinction) | SC-011 (artifact naming) |
| US6 (conditional MC) | US6-1 (state conditioning) | SC-008, SC-009 |
| US6 (conditional MC) | US6-2 (reproducibility) | SC-007 |
| US6 (conditional MC) | US6-3 (fallback) | SC-008 (sparsity warning) |
| US7 (config swap) | US7-1 (component selection) | SC-006 |
| US7 (config swap) | US7-2 (pricer swap) | SC-017 |
| US7 (config swap) | US7-3 (invalid config) | SC-022 |
| US8 (replay) | US8-1 (metadata capture) | SC-015, SC-046 |
| US8 (replay) | US8-2 (replay methods) | SC-007 (seeded) or SC-014 (persisted) |
| US8 (replay) | US8-3 (data drift) | SC-027 |

### Edge Cases → Requirements Mapping

| Edge Case (spec.md lines 162-167) | Functional Requirement | Success Criterion |
|---|---|---|
| Missing/insufficient historical data | FR-010, FR-032, DM-013 | SC-019, SC-020 |
| Distribution fit fails or implausible params | FR-002, FR-020, FR-037 | SC-019 |
| Option maturity < horizon | FR-004, FR-016, FR-087 | SC-073 (maturity=horizon), implied for maturity<horizon |
| Zero/negative prices, NaN in data | FR-022, FR-047, DM-017 | SC-028, SC-031, SC-038 |
| Fat tails causing overflow | FR-002, FR-022, FR-037 | SC-019, SC-031 |
| Parallel grid exceeding limits | FR-013, FR-018, FR-058 | SC-001, SC-033, SC-034 |

---

**Traceability Notes**:
- **DM-to-Tasks**: All DM requirements map to implementation tasks in `tasks.md`. Cross-reference via task descriptions (e.g., T008-T025 implement data loading per DM-001/004/006; T060-T063 implement MC storage per DM-008-012).
- **FR Coverage**: All FR requirements (FR-001 through FR-087 + FR-CAND-001 through FR-CAND-006) map to at least one SC. Unmapped FRs are logged/documented requirements without explicit test criteria (e.g., FR-014 macro alignment, FR-024/026/044 logging).
- **US Coverage**: All user story acceptance scenarios map to testable success criteria. Independent tests per US validate core functionality.
- **Edge Case Coverage**: All 6 edge cases from spec.md lines 162-167 map to explicit FR requirements and SC validation tests.
