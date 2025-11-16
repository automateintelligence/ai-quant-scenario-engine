# Implementation Plan: Backtesting & Strategy Spec Authoring (Quant Scenario Engine)

**Branch**: `001-mvp-pipeline` | **Date**: 2025-11-16 | **Spec**: [/specs/001-mvp-pipeline/spec.md](/specs/001-mvp-pipeline/spec.md)
**Input**: Feature specification from `/specs/001-mvp-pipeline/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

MVP delivers a CPU-only Quant Scenario Engine that ingests historical OHLCV, fits heavy-tailed return models (Laplacian default; alternatives like Student-T), generates Monte Carlo price paths, and compares paired stock vs option strategies via a CLI. It must support parameter grids, conditional candidate episodes, and modular component swaps (data source, distribution, option pricer) while producing reproducible artifacts (run metadata, metrics, optional plots) on the `001-mvp-pipeline` branch.

## Technical Context

**Language/Version**: Python 3.11 (CPU-only VPS)  
**Primary Dependencies**: numpy, pandas, scipy, numba, statsmodels/arch (fat-tail fits), pandas-ta (technical indicators), quantstats/plotly for reporting, yfinance (default data), Schwab API stub, CLI via typer; option pricer layer using Black–Scholes with per-strike implied volatility by default, with support for advanced pricers (e.g., Heston/QuantLib) via configuration.  
**Storage**: Parquet for canonical OHLCV/features; `.npz` or `numpy.memmap` for Monte Carlo arrays that exceed in-memory thresholds; run metadata as JSON (or YAML); configs via YAML/CLI/env.  
**Testing**: pytest + property-based tests (where useful) for distributions; contract tests for CLI/config/schema; coverage target ≥80% per constitution.  
**Target Platform**: Linux VPS (8 vCPU, 24 GB RAM), no GPU.  
**Project Type**: Single-package research/CLI toolchain.  
**Performance Goals**: Baseline CLI (1,000 paths × 60 steps for one config) SHOULD complete in ≤10s on the target VPS; grid runs up to ~50 configs SHOULD complete in ≤15m; Monte Carlo memory usage SHOULD stay under ~25% of RAM when in-memory, otherwise fall back to `.npz`/memmap with clear logging.  
**Constraints**: CPU-only; avoid >50% RAM for Monte Carlo arrays; robust to missing data; deterministic seeded runs; fail fast on invalid configs; option pricer accuracy vs speed is controlled via configuration (Black–Scholes+IV as default; advanced pricers opt-in).  
**Scale/Scope**: Single user; screen ≥100 symbols daily; 5–20 live symbols; up to thousands of Monte Carlo paths per run; batch grids up to ~50 configs.
## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Specification-driven**: Plan/research/data-model/contracts/quickstart must align to spec.md; no implementation outside documented scope (Constitution II, XVIII).
- **Contracts & reproducibility**: Define explicit schemas for CLIs/configs/artifacts and guarantee seeded reruns + run metadata (Constitution VIII, XIX).
- **Testing discipline**: Commit to boundary tests (CLI + data adapters + pricers), contract tests, and ≥80% coverage; fail fast on violations (Constitution XV, V).
- **Simplicity & separation**: Prefer minimal vectorized pipeline; avoid speculative subsystems (e.g., live trading stack) until justified; keep component swaps via config (Constitution III, XVI).

## Project Structure

### Documentation (this feature)

```text
specs/001-mvp-pipeline/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
backtesting/
├── data/               # data source adapters (yfinance, Schwab stub, loaders)
├── features/           # indicator + macro enrichment
├── distributions/      # fit/sample abstractions for Normal/Laplace/Student-T/GARCH
├── mc/                 # Monte Carlo generators + memmap/npz persistence helpers
├── pricing/            # option pricers (Black-Scholes default; alt pricers TBD)
├── strategies/         # stock + option strategies, param schemas
├── simulation/         # core simulator, grid runner, conditional episode logic
└── cli/                # typer entrypoints for compare, grid, screening

tests/
├── unit/
├── integration/        # CLI + data/pricer wiring + persistence paths
└── contract/           # CLI/config/schema/contracts
```

**Structure Decision**: Single-package layout rooted in `backtesting/` with sibling `tests/`; keeps CLI + engine co-located for fast iteration while preserving clear submodules for data, MC, pricers, strategies, and simulations.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |

## Phases and Outputs (alignment with /speckit workflow)

- **Phase 0 (Research)**: Decision log captured in `research.md` (option pricer interface per FR-016, data sourcing per FR-017, MC storage per FR-013), unresolved clarifications flagged. Gate: constitution compliance + clarified unknowns.
- **Phase 1 (Design & Contracts)**: Artifacts `data-model.md`, `contracts/openapi.yaml`, `quickstart.md`; explicit interfaces (below) and config/CLI schemas; re-run constitution check.
- **Phase 2 (Implementation planning)**: `/speckit.tasks` to derive ordered tasks; wire Typer CLIs (`compare`, `grid`, `screen`, `conditional`, `replay`), add tests and resource guards.
- **Architecture Diagram**: Generate a simple flow (DataSource → Distribution → MC → Strategies → Simulation → Metrics/artifacts) to live alongside `plan.md` (image or Mermaid) for onboarding.

## Interfaces & Public Functions (per module)

- `backtesting.data`: `load_ohlcv(symbol, start, end, interval) -> DataFrame`; adapters `YFinanceSource`, `SchwabStub` chosen via config.
- `backtesting.distributions`: `ReturnDistribution.fit(returns)`, `.sample(n_paths, n_steps, seed)`, models: Laplace (default), Student-T, optional GARCH-T flag (FR-002).
- `backtesting.mc`: `generate_paths(dist, n_paths, n_steps, s0, seed, storage_policy)` with memmap/npz fallback (FR-013/DM-011).
- `backtesting.pricing`: `OptionPricer.price(path_slice, option_spec)`; `BlackScholesPricer` default; `PyVollibPricer` optional; adapter slot for QuantLib/Heston later (FR-016).
- `backtesting.strategies`: `Strategy.generate_signals(price_paths, features, params) -> StrategySignals` (stock + option signals + OptionSpec); param schemas validated against `StrategyParams`.
- `backtesting.simulation`: `run_compare`, `run_grid`, `run_conditional_backtest`, `run_conditional_mc`, all producing `SimulationRun` + artifacts.
- `backtesting.cli`: Typer commands invoking above, enforcing config validation and run_meta persistence (FR-009, FR-019).

## Concurrency Model

- Use `concurrent.futures.ProcessPoolExecutor` (default max_workers=6 on 8 vCPU) for grid evaluations; single-process vectorized NumPy/Numba for path generation.
- Per FR-018, preflight resource estimator aborts when predicted time/memory > thresholds; long-running jobs emit warnings and stop remaining tasks.

## Error Handling Policies (map to SC-019..SC-021)

- Structured exceptions: `DistributionFitError`, `DataSourceError`, `ResourceLimitError`, `PricingError`, `EpisodeGenerationError` with actionable messages.
- Fail-fast on invalid configs/data gaps; log warnings for recoverable cases (fallback data source, selector with insufficient episodes triggers unconditional MC fallback + warning).
- Each run writes `run_meta.json` + logs for traceability; replay refused on data drift unless forced (FR-019).

## Performance Budget (targets for validation)

| Path/Step Budget | Metric | Target |
|------------------|--------|--------|
| Baseline compare (1k×60) | wall time | ≤10s on 8 vCPU VPS |
| Grid (≤50 configs, 1k×60) | wall time | ≤15m |
| Laplace/Student-T fit | runtime | ≤1s per symbol window |
| MC generation | throughput | ≥50k steps/s aggregate |
| Memory | footprint | <25% RAM in-memory; auto memmap beyond |

## Risk Mitigation

- **Data gaps/schema drift**: schema validation on load; versioned Parquet; downgrade/skip with warnings (FR-010, DM-series).
- **Distribution instability**: parameter sanity checks + fallback to Laplace when fits fail; seed persistence for reproducibility.
- **OOM / resource overrun**: estimator enforces FR-013/FR-018 thresholds; memmap chunking; worker cap.
- **Selector sparsity**: minimum episode threshold; fallback to unconditional MC with warning.

## Onboarding & Workflow

- Branch model: work on `001-mvp-pipeline`; PRs require spec alignment, tests, lint/type checks (black/ruff/mypy), constitution check.
- Preferred tooling: `python -m venv .venv`, `pip install -r requirements-dev.txt` (to be added); `pytest` for tests; `ruff` + `mypy` gating CI.
- Contracts in `specs/001-mvp-pipeline/contracts/` are source of truth for CLI/config validation; update alongside code changes.


## Phases & Milestones

**Phase 0 – Research & Architecture Decisions**  
- Finalize distribution/pricing choices (e.g., Laplace vs Student-T; Black–Scholes default vs advanced pricers).  
- Lock data storage policy (Parquet + `.npz`/memmap) and run provenance requirements.  
- Produce `research.md` and initial architecture notes (this plan, data-model, quickstart sketches).

**Phase 1 – Core Engine Skeleton (MVP)**  
- Implement data layer (yfinance adapter, Parquet loaders, feature engineering) and RunConfig schema.  
- Implement distribution interfaces and at least two concrete models (Normal, Laplace or Student-T).  
- Implement Monte Carlo engine, stock/option strategy interfaces, and MarketSimulator for a single symbol.  
- Implement basic CLI for “stock vs option comparison” using seeded runs and writing run artifacts.

**Phase 2 – Conditional Episodes, Grids, and Screening**  
- Implement CandidateSelector abstraction and candidate episode builder.  
- Implement conditional backtesting and (at least one) conditional Monte Carlo method.  
- Implement grid runner for parameter exploration and stock screening CLI for a universe of symbols.  
- Harden error-handling, resource limits, and reproducibility (replay) in line with spec and success criteria.

Future phases may extend to more advanced pricers, additional distribution models, and portfolio-level strategies once MVP is stable.

## Concurrency & Resource Management

- Concurrency will be implemented via Python's `concurrent.futures` (thread or process pool) with a configurable `max_workers` (default ≤6 on the 8-core VPS).  
- Grid jobs will use task-level parallelism (different parameter configs as independent tasks) while keeping Monte Carlo path generation vectorized inside each task.  
- Resource limits will be enforced by:
  - Estimating memory needs based on `n_paths × n_steps × n_fields × 8 bytes` and comparing to configured RAM thresholds.  
  - Capping path counts or refusing jobs when estimated usage exceeds configured limits (FR-013, FR-018).  
  - Logging warnings and failing fast when jobs would exceed runtime or memory budgets.

## Error Handling & Observability

- Define explicit exception classes (e.g., `DataSourceError`, `DistributionFitError`, `PricingError`, `ResourceLimitError`, `EpisodeGenerationError`) and raise them from the appropriate layers.  
- Ensure CLIs:
  - Exit with non-zero status on unrecoverable errors.  
  - Emit structured error messages (including symbol, run_id, component, and cause).  
- Logging:
  - Use a standard logging configuration (e.g., Python `logging` with INFO/DEBUG levels) and write logs into each run directory alongside metrics and metadata.  
  - Record key events: data loading, model fits, MC generation, simulation progress, grid job status.

## Performance Budget

The following table captures initial performance targets for the MVP on the CPU-only VPS:

| Operation                                 | Target (approx)                  | Notes                                      |
|-------------------------------------------|----------------------------------|--------------------------------------------|
| Single run (1,000 paths × 60 steps)       | ≤ 10 seconds                     | Includes data load, fit, MC, strategies    |
| Grid run (≤ 50 configs, same scale)       | ≤ 15 minutes                     | With `max_workers` tuned for 8 vCPU        |
| Distribution fit (per model, per symbol)  | ≤ 1 second (Normal/Laplace)      | Heavier models (e.g., GARCH) are optional  |
| Run artifact write (metrics/metadata)     | ≤ 1 second per run               | Parquet/JSON writes amortized              |

These budgets are guidance for implementation and testing; they can be refined after profiling but MUST remain documented so SC-001–SC-003 are objectively testable.

## Reference Interfaces & Code Snippets (Non-normative)

The following reference snippets are non-normative but illustrate expected interfaces and shapes for key components.

### Monte Carlo Path Generator

```python
# engine/mc.py

import numpy as np

def generate_price_paths(
    s0: float,
    distribution,
    n_paths: int,
    n_steps: int,
) -> np.ndarray:
    """Generate price paths from a fitted return distribution.

    Returns:
        np.ndarray: price matrix of shape (n_paths, n_steps)
    """
    r = distribution.sample(n_paths, n_steps)  # log returns
    log_s = np.log(s0) + np.cumsum(r, axis=1)
    return np.exp(log_s)
```

### RunConfig & Run Metadata Schema

```python
# schema/run_config.py

from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class RunConfig:
    symbol: str
    data_source: str
    distribution: str
    strategy_name: str
    strategy_params: Dict[str, Any]
    mc_paths: int
    mc_steps: int
    random_seed: int
    resource_limits: Dict[str, Any] = field(default_factory=dict)
```

Example `run_meta.json` written per run:

```json
{
  "run_id": "2025-11-16_15-30-22",
  "symbol": "AAPL",
  "data_source": "yfinance",
  "distribution": "student_t",
  "strategy": "MeanReversionCall",
  "strategy_params": { "threshold": 0.03, "contracts": 1 },
  "mc_paths": 1000,
  "mc_steps": 60,
  "random_seed": 42,
  "version": "QuantScenarioEngine v0.1.0"
}
```

These references guide implementation and contract tests but do not constrain internal module layout beyond the requirements defined in `spec.md` and `data-model.md`.
