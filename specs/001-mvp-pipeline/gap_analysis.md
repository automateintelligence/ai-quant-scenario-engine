# Gap Analysis: Specification, Plan, and Tasks Review
**Date**: 2025-11-16
**Purpose**: Identify ambiguities, gaps, and potential blockers for coding agents
**Scope**: spec.md, plan.md, data-model.md, contracts/openapi.yaml, tasks.md

---

## 🔴 CRITICAL GAPS (Must resolve before implementation)

### G1: Strategy.generate_signals() Interface Mismatch
**Location**: data-model.md lines 10-11, 83-98
**Issue**: Two conflicting signatures for `Strategy.generate_signals()`:
- Line 10: `generate_signals(price_paths: np.ndarray, features: DataFrame|dict, params: StrategyParams)`
- Lines 83-98: `generate_signals(prices: pd.DataFrame, features: pd.DataFrame, params: Dict[str, Any])`

**Impact**: Coding agents will not know which signature to implement
**Resolution Needed**:
- Clarify if `price_paths` is a 2D array `[n_paths, n_steps]` or `prices` is a DataFrame
- Specify if `params` is a `StrategyParams` dataclass or `Dict[str, Any]`
- Update all references consistently across data-model.md, spec.md, and tasks

---

### G2: Missing OptionSpec Strike Parsing Logic
**Location**: data-model.md line 60, spec.md FR-016
**Issue**: `strike` field allows both absolute float and relative string (`"atm"`, `"+0.05"`) but no parsing specification provided
**Impact**: Coding agents won't know how to:
- Parse `"atm"` → current spot price
- Parse `"+0.05"` → spot * 1.05 or spot + 0.05?
- Handle invalid formats

**Resolution Needed**:
- Define exact parsing rules for relative strike strings
- Specify error handling for malformed strikes
- Add validation task to Phase 2 or document in data-model.md

**Example Needed**:
```python
# Is "+0.05" interpreted as:
strike = spot * 1.05  # 5% OTM
# OR
strike = spot + 0.05  # $0.05 OTM (makes no sense)
# OR
strike = spot + (spot * 0.05)  # 5% OTM (clearer)
```

---

### G3: CandidateSelector.score() vs .select() Relationship Undefined
**Location**: spec.md lines 254-264, data-model.md
**Issue**: Two methods defined but relationship unclear:
- `score(features) -> Series` returns float scores
- `select(features, threshold) -> Series` returns boolean mask
- Is `select()` just `score() >= threshold`?
- Can users call `score()` separately to tune thresholds?
- What's the threshold range and default value?

**Impact**: Agents won't know if they need to implement both or if one calls the other
**Resolution Needed**: Clarify in data-model.md with example implementation

---

### G4: Missing "Normal" Distribution in Contracts but Referenced in Tasks
**Location**: contracts/openapi.yaml line 106, tasks.md T093
**Issue**: OpenAPI enum lists `[laplace, student_t, garch_t]` but:
- T093 says "Implement NormalDistribution for config comparison"
- FR-002 implies Normal is available
- research.md mentions "Normal-only (insufficient tails)"

**Impact**: CLI validation will reject `distribution=normal` if not in enum
**Resolution Needed**: Either add `normal` to OpenAPI enum or remove T093

---

### G5: Conditional MC Fallback Chain Not Specified
**Location**: spec.md US6 acceptance scenario 3, FR-036, tasks.md T082-T084a
**Issue**: States fallback order as "bootstrap → refit → unconditional" but details missing:
- What triggers each fallback? (exact threshold: `n_episodes < 30`?)
- Do we try bootstrap first then refit, or pick one based on config?
- What gets logged? Just the method used or why fallback happened?
- Does fallback change the `run_meta` schema?

**Impact**: Agents won't know how to implement the decision tree
**Resolution Needed**: Add explicit decision logic to data-model.md or research.md

**Example Needed**:
```python
if n_matching_episodes >= 30:
    return bootstrap_sample(episodes)
elif n_matching_episodes >= 10:
    log.warning("Insufficient episodes for bootstrap, using parametric refit")
    return parametric_refit(episodes)
else:
    log.warning("Insufficient episodes, falling back to unconditional MC")
    return unconditional_sample()
```

---

### G6: Missing Schema for `run_meta.json`
**Location**: FR-019, FR-008, US8 acceptance scenario 1, tasks.md T044, T108
**Issue**: `run_meta.json` referenced extensively but no JSON schema provided:
- What are the exact required fields?
- What's the structure for nested objects?
- Version field? Format version?
- Is it append-only or single write?

**Impact**: Agents will invent inconsistent schemas
**Resolution Needed**: Add complete JSON schema example to data-model.md

**Minimum Required**:
```json
{
  "run_id": "string",
  "timestamp": "ISO8601",
  "symbol": "string",
  "config": {
    "n_paths": "int",
    "n_steps": "int",
    "seed": "int",
    "distribution": "string",
    "data_source": "string"
  },
  "distribution_params": {},
  "data_fingerprint": "SHA256",
  "library_versions": {},
  "git_commit": "string",
  "storage_policy": "memory|npz|memmap",
  "schema_version": "0.1.0"
}
```

---

### G7: Missing Error Code Mapping for Structured Exceptions
**Location**: plan.md lines 104-108, spec.md SC-005, SC-019
**Issue**: 5 exception classes defined but no mapping to exit codes or error structures:
- `DistributionFitError`
- `DataSourceError`
- `ResourceLimitError`
- `PricingError`
- `EpisodeGenerationError`

**Impact**: CLI error handling will be inconsistent; no clear user-facing error messages
**Resolution Needed**: Define error code schema and example messages

**Example**:
```python
class DistributionFitError(Exception):
    code = "DIST_FIT_001"
    message_template = "Failed to fit {model} distribution: {reason}. Try increasing min_samples or using fallback model."
```

---

### G8: Unclear Feature Pipeline Integration with Strategies
**Location**: US3, FR-006, tasks.md T099-T107
**Issue**: US3 says "augment simulations with indicators" but unclear:
- Are features computed once per symbol or per path?
- For Monte Carlo paths, do we compute indicators on each simulated path?
- How do features align with `[n_paths, n_steps]` price arrays?
- Is `features` a 2D array `[n_paths, n_steps]` or 1D `[n_steps]`?

**Impact**: Major architectural ambiguity affecting entire feature system
**Resolution Needed**: Clarify in data-model.md whether features are:
1. **Historical only**: Features computed on actual OHLCV, same values broadcast to all paths
2. **Path-specific**: Features recomputed for each Monte Carlo path (expensive)

**Likely Intent**: Historical features only, but needs explicit statement

---

## 🟡 HIGH PRIORITY GAPS (Will cause confusion but workarounds exist)

### G9: Missing Parquet Compression Specification
**Location**: DM-004, tasks.md T019, T020a
**Issue**: DM-004 says "Snappy or ZSTD recommended" but doesn't specify:
- Default compression algorithm
- Compression level (if ZSTD)
- When to use which algorithm
- How to configure via CLI/config

**Impact**: Inconsistent file formats, performance variability
**Resolution Needed**: Pick default (suggest ZSTD level 3) and document in data-model.md

---

### G10: Undefined "Implausible Parameter" Thresholds
**Location**: FR-037, FR-020, tasks.md T028
**Issue**: FR-037 lists examples but no exact thresholds:
- "scale > 0 and finite" - OK
- "Student-T df ∈ [2, 100]" - Why 100? Why not 30?
- "GARCH parameters within stationarity bounds" - What are the bounds?

**Impact**: Agents will pick arbitrary validation rules
**Resolution Needed**: Document exact validation rules in data-model.md per distribution

---

### G11: Missing max_workers Default and Bounds
**Location**: plan.md line 101, FR-018, contracts missing
**Issue**: States "default max_workers=6 on 8 vCPU" but:
- No config schema for max_workers
- No validation rule (1 ≤ max_workers ≤ cores-2?)
- Not in RunConfig or GridRequest schemas
- No task to implement worker pool management

**Impact**: Grid execution won't have worker controls
**Resolution Needed**: Add to RunConfig schema and GridRequest; add validation task

---

### G12: Atomic Write Mechanism for run_meta.json Not Specified
**Location**: FR-030, DM-018, tasks.md T020c, T043a
**Issue**: "Atomic, append-only writes" required but no implementation guidance:
- Write to temp file then `os.rename()`?
- Use file locking?
- What if write fails partway?
- How to detect corrupt/partial files?

**Impact**: Race conditions in parallel runs, corrupted metadata
**Resolution Needed**: Add implementation pattern to plan.md or data-model.md

---

### G13: Missing Data Drift Detection Algorithm
**Location**: FR-019, FR-027, FR-028, tasks.md T020a, T109, T112
**Issue**: States "detect drift" and "fingerprint" but no algorithm:
- SHA256 of what exactly? Raw bytes, sorted rows, schema JSON?
- Does order matter for Parquet files?
- Do we hash the entire file or just metadata?
- Schema drift vs data drift - same fingerprint or separate?

**Impact**: Agents will implement inconsistent fingerprinting
**Resolution Needed**: Specify exact fingerprint algorithm in data-model.md

**Suggested**:
```python
def compute_fingerprint(parquet_path: str) -> str:
    df = pd.read_parquet(parquet_path)
    schema_hash = hashlib.sha256(str(df.dtypes.to_dict()).encode()).hexdigest()[:8]
    data_hash = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:8]
    return f"{schema_hash}:{data_hash}"
```

---

### G14: Universe/Watchlist/Live Set Configuration Schema Missing
**Location**: DM-007, tasks.md T020g
**Issue**: T020g says "Create config schema" but no specification of:
- File format (YAML, JSON, Python dict?)
- Structure (flat list, nested dict, separate files?)
- How to specify per-symbol resolution requirements
- How CLI loads and validates this config

**Impact**: Agent will invent arbitrary format
**Resolution Needed**: Add example YAML schema to data-model.md

**Example**:
```yaml
universe:
  symbols: [AAPL, MSFT, ...]
  resolutions: [1d, 5m]

watchlist:
  symbols: [AAPL, TSLA]
  resolutions: [1d, 5m]
  features: true

live:
  symbols: [AAPL]
  resolutions: [1d, 5m, 1m]
```

---

### G15: Missing Objective Function Definition for Grid Ranking
**Location**: US2, FR-007, tasks.md T074
**Issue**: US2 says "ranked by objective function" but no definition:
- Is it Sharpe ratio? CVaR? User-defined?
- Configurable or hardcoded?
- How to handle multi-objective optimization?
- Single scalar output or Pareto frontier?

**Impact**: T074 can't implement ranking without knowing the objective
**Resolution Needed**: Add objective function specification to data-model.md or GridRequest schema

---

### G16: Unclear EpisodeGenerationError vs Warning Behavior
**Location**: plan.md line 107, US5 acceptance scenario 3, tasks.md T070
**Issue**: States "minimum episode threshold triggers warning" but also lists `EpisodeGenerationError`:
- When do we warn vs error?
- Is min_episodes=30 a hard requirement or soft suggestion?
- Does warning allow continuation with fewer episodes?

**Impact**: Inconsistent error handling between backtest and conditional MC
**Resolution Needed**: Clarify threshold behavior (suggest: error if <10, warn if <30)

---

## 🟢 MEDIUM PRIORITY GAPS (Best practices, not blockers)

### G17: Missing Black-Scholes Greeks Specification
**Location**: FR-016, data-model.md OptionPricer interface
**Issue**: BlackScholesPricer only specifies `price()` method:
- Greeks (delta, gamma, vega, theta) are required.
- Needed for position sizing or P&L attribution?
- If not MVP, should interface reserve methods?

**Impact**: May need interface changes later

---

### G18: Incomplete Test Coverage Guidance ✅ RESOLVED
**Location**: spec.md (tests not explicitly requested), tasks.md line 6, plan.md line 17
**Issue**: Constitution requires ≥80% coverage but:
- No test tasks in tasks.md
- "Not explicitly requested" note created confusion
- Which modules require property-based tests?
- Contract test scope unclear

**Impact**: Agents may skip testing entirely
**Resolution**: Added comprehensive "Testing Strategy" section to tasks.md with:
- Test-during-implementation pattern (not deferred to Phase 12)
- Directory structure (unit/integration/contract)
- Coverage requirements by module type (100% critical, 90% core, 80% infrastructure)
- Property-based testing guidance with hypothesis
- Merge criteria requiring ≥80% coverage gate
- Test execution commands and fixtures
- Updated T007 to include pytest, pytest-cov, hypothesis, pytest-mock

---

### G19: Missing Progress Reporting Specification
**Location**: FR-039, long-running grid jobs
**Issue**: "Long-running jobs MUST emit progress updates" but no specification:
- What's "long-running"? (>30s?)
- Progress format (percentage, ETA, items completed?)
- Where to emit (stdout, log file, both?)
- Frequency (every N seconds? Every N configs?)

**Impact**: Inconsistent user experience
**Resolution Needed**: Add progress reporting spec to plan.md

---

### G20: Undefined Position Sizing Implementation
**Location**: data-model.md StrategyParams line 55, position_sizing enum
**Issue**: Enum lists `[fixed_notional, percent_equity]` but no specification:
- What's the equity curve? Starting capital?
- How to specify notional amount (parameter where?)
- Does percent_equity rebalance each step or fixed from start?
- Leverage limits?

**Impact**: Strategy implementation will be inconsistent
**Resolution Needed**: Add position sizing logic to simulation documentation

---

### G21: Missing Logging Configuration Specification
**Location**: FR-039, tasks.md T009
**Issue**: T009 says "structured JSON logging" but missing:
- Log level defaults
- Log rotation policy
- Log destination (file, stdout, both?)
- Sensitive data filtering (seeds in logs?)

**Impact**: Verbose or insufficient logging
**Resolution Needed**: Add logging config example to plan.md

---

### G22: Unclear Fees and Slippage Application
**Location**: data-model.md StrategyParams line 55
**Issue**: `fees` and `slippage` fields defined but no specification:
- Applied per trade or per share?
- Percentage or absolute value?
- Bid-ask spread modeling?
- When applied (entry, exit, both?)

**Impact**: P&L calculations will be inconsistent
**Resolution Needed**: Document fee/slippage model in data-model.md

---

### G23: Missing YAML Config File Schema
**Location**: US7, FR-024, tasks.md T088-T091
**Issue**: US7 requires YAML config but no schema provided:
- What's the top-level structure?
- How to nest distribution params, strategy configs?
- Defaults handling?
- Validation against OpenAPI schemas?

**Impact**: Agent will create arbitrary YAML structure
**Resolution Needed**: Add example config YAML to data-model.md or quickstart.md

---

### G24: Incomplete ConditionalRequest Schema
**Location**: contracts/openapi.yaml, tasks.md T068-T069
**Issue**: ConditionalRequest schema not fully shown in spec review:
- Does it support both backtest and MC modes?
- How to specify conditioning method?
- CandidateEpisode specification?

**Impact**: T068-T069 can't implement without full schema
**Resolution Needed**: Review contracts/openapi.yaml ConditionalRequest definition

---

## 📋 AMBIGUITIES (Interpretation needed)

### A1: "Broadcast" vs "Compute Per Path" for Features
**Location**: G8 (see critical gaps)
**Clarification Needed**: Confirm features are historical-only for MVP

---

### A2: Storage Policy Selection Boundary
**Location**: DM-008, DM-010, DM-011
**Issue**: Three thresholds mentioned:
- <25% RAM → in-memory
- ≥25% RAM → npz/memmap
- >50% RAM → memmap required

**Question**: What happens between 25-50%? User choice? npz default?
**Clarification Needed**: Simplify to two tiers or define middle range behavior

---

### A3: "Episode Horizon" Definition
**Location**: data-model.md CandidateEpisode line 109
**Issue**: `horizon` (int bars) defined but unclear:
- Forward-looking window from t0?
- Fixed for all episodes or episode-specific?
- Must match simulation n_steps?

**Clarification Needed**: Add example showing t0, horizon, and simulation relationship

---

### A4: "Payoff-Only Mode" for Option Pricing
**Location**: data-model.md line 12
**Issue**: "Supports payoff-only mode for expiry pricing" mentioned but:
- What's the interface? Separate method?
- When to use vs full pricing?
- Does it affect Greeks?

**Clarification Needed**: Specify if this is an optional optimization or required feature

---

## ✅ RECOMMENDED ADDITIONS

### R1: Add Glossary Section to spec.md
**Reason**: Terms like "episode," "horizon," "conditional," "broadcast" used without definition
**Content**: 1-page glossary of domain terms

---

### R2: Add Decision Matrix for Storage Policy Selection
**Reason**: DM-008/010/011 thresholds need visualization
**Content**: Flowchart or decision table in data-model.md

---

### R3: Add Error Handling Flowchart
**Reason**: Multiple fallback chains need clear visualization
**Content**: Mermaid diagram showing: data load → fit → MC → strategy → error paths

---

### R4: Add Example run_meta.json Files
**Reason**: Three scenarios needed:
1. Successful unconditional run
2. Conditional backtest with episodes
3. Failed run with fallback

---

### R5: Add CLI Usage Examples to quickstart.md
**Reason**: OpenAPI contracts alone don't show CLI syntax
**Content**: Examples for all 5 commands with realistic flags

---

## 📊 SUMMARY

| Category | Count | Severity |
|----------|-------|----------|
| 🔴 Critical Gaps | 8 | MUST FIX before implementation |
| 🟡 High Priority | 8 | FIX before Phase 3 (US1) |
| 🟢 Medium Priority | 7 | FIX before Phase 12 (Polish) |
| ✅ Resolved | 1 | G18 - Testing strategy added |
| 📋 Ambiguities | 4 | CLARIFY in next spec revision |
| ✅ Recommendations | 5 | NICE TO HAVE |

**Top Blockers for Immediate Resolution**:
1. G1 - Strategy interface signature mismatch
2. G6 - Missing run_meta.json schema
3. G8 - Feature pipeline integration unclear
4. G5 - Conditional MC fallback chain unspecified
5. G2 - OptionSpec strike parsing undefined

**Estimated Impact**: Without resolving critical gaps, coding agents will:
- Make inconsistent architectural decisions (30% rework risk)
- Implement incompatible interfaces (requires refactoring)
- Skip validation/error handling (technical debt)
- Create untestable components (blocks Phase 12)

**Recommendation**: Resolve 🔴 critical gaps before starting Phase 3 (US1 implementation). Can defer 🟢 medium priority gaps to relevant phases.
