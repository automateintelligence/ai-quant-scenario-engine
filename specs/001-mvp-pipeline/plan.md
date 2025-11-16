# Implementation Plan: Backtesting & Strategy Spec Authoring (Quant Scenario Engine)

**Branch**: `001-mvp-pipeline` | **Date**: 2025-11-16 | **Spec**: [/specs/001-mvp-pipeline/spec.md](/specs/001-mvp-pipeline/spec.md)
**Input**: Feature specification from `/specs/001-mvp-pipeline/spec.md`
**Hierarchy**: Parent spec above per `.specify/memory/CONSTITUTION.md`; this plan is parent to `research.md`, `data-model.md`, `contracts/`, and `quickstart.md` in the same directory.

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
quant-scenario-engine/
├── data/                   # data source adapters (yfinance, Schwab API, PyData loaders)
├── features/               # indicator + macro enrichment
├── schema/
├── models/
├── interfaces/
├── runs/
├── backtesting/
|   ├── distributions/      # fit/sample abstractions for Normal/Laplace/Student-T/GARCH
|   └── mc/                 # Monte Carlo generators + memmap/npz persistence helpers
├── optimizer/
├── pricing/                # option pricers (Black-Scholes default; alt pricers Heston, TBD)
├── strategies/             # stock + option strategies, param schemas
├── simulation/             # core simulator, grid runner, conditional episode logic
├── cli/                    # typer entrypoints for compare, grid, screening
├── config/
|     main.py
├── tests/
    ├── unit/
    ├── integration/        # CLI + data/pricer wiring + persistence paths
    └── contract/           # CLI/config/schema/contracts
```

**Structure Decision**: Single-package layout rooted in `quant-scenario-engine/`; keeps CLI + engine co-located for fast iteration while preserving clear submodules for data, backtesting, MC, pricers, strategies, and simulations.

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
- `backtesting.pricing`: `OptionPricer.price(path_slice, option_spec)`; `BlackScholesPricer` default; `PyVollibPricer` optional; adapter slot for QuantLib/Heston 'HestonPricer' later(FR-016).
- `backtesting.strategies`: `Strategy.generate_signals(price_paths, features, params) -> StrategySignals` (stock + option signals + OptionSpec); param schemas validated against `StrategyParams`.
- `backtesting.simulation`: `run_compare`, `run_grid`, `run_conditional_backtest`, `run_conditional_mc`, all producing `SimulationRun` + artifacts.
- `quant-scenario-engine.cli`: Typer commands invoking above, enforcing config validation and run_meta persistence (FR-009, FR-019).

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

## Component Wiring & Dependency Injection

**Factory Pattern Implementation** (per FR-043):

```python
# config/factories.py

from typing import Protocol, Dict, Type

class DataSourceFactory:
    """Registry for data source implementations."""
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register data source implementation."""
        def wrapper(impl_class):
            cls._registry[name] = impl_class
            return impl_class
        return wrapper

    @classmethod
    def create(cls, name: str, **kwargs):
        """Create data source instance from config key."""
        if name not in cls._registry:
            raise ComponentNotFoundError(
                f"Data source '{name}' not registered. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

# Usage in data/yfinance_source.py:
@DataSourceFactory.register("yfinance")
class YFinanceSource(DataSource):
    def load_ohlcv(self, symbol, start, end, interval):
        # implementation
        pass

# Similar factories for:
# - DistributionFactory: "laplace", "student_t", "normal", "garch_t"
# - PricerFactory: "black_scholes", "pyvollib", "heston" (stub)
# - SelectorFactory: "gap_volume", "custom_dsl"
```

**Component Swap Logging** (per FR-043):
- All factory `.create()` calls MUST log at INFO level: `Component loaded: type=DataSource, name=yfinance, prior=schwab`
- Log format: `{"event": "component_swap", "component_type": "DataSource", "prior": "schwab", "new": "yfinance", "timestamp": "...", "run_id": "..."}`

## Concurrency & Resource Management

**Process Pool Execution Model** (per FR-084):

```python
# optimizer/grid_runner.py

from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def run_grid(configs: List[RunConfig], max_workers: int = None):
    """Execute grid with process-level parallelism."""
    # Clamp max_workers per FR-058/FR-084
    if max_workers is None:
        max_workers = min(6, os.cpu_count() or 1)
    else:
        max_workers = min(max_workers, os.cpu_count() - 2)
        if max_workers != config.max_workers:
            log.warning(f"max_workers clamped to {max_workers}")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all configs
        futures = {
            executor.submit(run_single_config, cfg): idx
            for idx, cfg in enumerate(configs)
        }

        # Collect results as they complete (FR-062: partial results)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append({"config_index": idx, "status": "success", **result})
                write_partial_result(result)  # FR-062: immediate write
            except Exception as e:
                log.error(f"Config {idx} failed: {e}")
                results.append({"config_index": idx, "status": "failed", "error": str(e)})
                # FR-081: continue with remaining configs

    return results
```

**Concurrency Model Details**:
- **Process pool**: `ProcessPoolExecutor` for grid jobs (config-level parallelism).
- **Shared memory**: None; each worker has independent memory space (no shared arrays).
- **IPC**: Parent collects via return values; no pipes/queues for simplicity.
- **Thread pool**: NOT used (Python GIL prevents CPU-bound parallelism).
- **Max workers**: `min(max_workers_config, os.cpu_count() - 2)` reserves 2 cores for OS.
- **MC generation**: Vectorized NumPy/Numba within each worker process (no intra-MC parallelism).

**Resource Limits Enforcement**:
- **Memory estimator** (FR-023): `estimated_gb = (n_paths * n_steps * 8 * 1.1) / 1e9` (10% overhead).
- **Thresholds**:
  - In-memory: `estimated_gb < 0.25 * total_ram_gb` (≈6 GB on 24 GB VPS).
  - Memmap fallback: `0.25 * total_ram_gb ≤ estimated_gb < 0.5 * total_ram_gb`.
  - Abort: `estimated_gb ≥ 0.5 * total_ram_gb` → raise `ResourceLimitError`.
- **Time budgets** (FR-018):
  - Baseline run: ≤10s (enforced via timeout; abort if exceeded).
  - Grid run: ≤15m (emit warnings at 50%, 90%; abort remaining tasks at 100%).
- **Preflight checks**: Run estimator before starting; fail fast if limits exceeded.

## Error Handling & Observability

**Exception Hierarchy** (per FR-041, FR-042):

```python
# exceptions.py

class QuantScenarioEngineError(Exception):
    """Base exception for all QSE errors."""
    pass

class ConfigError(QuantScenarioEngineError):
    """Configuration validation errors."""
    pass

class ConfigValidationError(ConfigError):
    """Invalid configuration field/value (FR-041)."""
    def __init__(self, field: str, value: Any, constraint: str, fix: str):
        self.field = field
        self.value = value
        self.constraint = constraint
        self.fix = fix
        super().__init__(
            f"Field '{field}' has invalid value '{value}'. "
            f"Constraint: {constraint}. Fix: {fix}"
        )

class ConfigConflictError(ConfigError):
    """Contradictory configuration settings (FR-053)."""
    pass

class ConfigVersionError(ConfigError):
    """Unsupported config schema version (FR-063)."""
    pass

class ComponentNotFoundError(ConfigError):
    """Component not registered in factory (FR-043)."""
    pass

class DataError(QuantScenarioEngineError):
    """Data loading/validation errors."""
    pass

class DataSourceError(DataError):
    """Data source fetch failures."""
    pass

class SchemaError(DataError):
    """Parquet schema drift (FR-027)."""
    pass

class InsufficientDataError(DataError):
    """Below minimum bar count (FR-032)."""
    pass

class TimestampAnomalyError(DataError):
    """Future dates or non-monotonic index (FR-057)."""
    pass

class DistributionFitError(QuantScenarioEngineError):
    """Distribution fitting failures (FR-020)."""
    pass

class PricingError(QuantScenarioEngineError):
    """Option pricing failures."""
    pass

class ResourceLimitError(QuantScenarioEngineError):
    """Memory or time budget exceeded (FR-023, FR-018)."""
    pass

class BankruptcyError(QuantScenarioEngineError):
    """All MC paths → 0 (FR-050)."""
    pass

class EpisodeGenerationError(QuantScenarioEngineError):
    """Candidate episode construction failures."""
    pass

class DependencyError(QuantScenarioEngineError):
    """Missing or incompatible dependencies (FR-066, FR-067)."""
    pass
```

**Structured Logging** (per FR-039):

```python
# logging_config.py

import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Format log records as JSON."""
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", None),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "duration_ms"):
            log_obj["duration_ms"] = record.duration_ms
        # Add custom fields from record
        for key in ["symbol", "config_index", "n_paths", "n_steps"]:
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)
        return json.dumps(log_obj)

# Setup per run
def setup_run_logging(run_id: str, output_dir: Path):
    logger = logging.getLogger("quant_scenario_engine")
    logger.setLevel(logging.INFO)

    # File handler with JSON format
    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(JSONFormatter())
    logger.addHandler(fh)

    # Console handler with human-readable format
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logger.addHandler(ch)

    return logger
```

**CLI Error Handling** (per SC-019, SC-022):

```python
# cli/main.py

import sys
from exceptions import *

def cli_main():
    try:
        result = run_pipeline(config)
        return 0
    except ConfigValidationError as e:
        log.error(str(e))
        return 1
    except InsufficientDataError as e:
        log.error(f"Data validation failed: {e}")
        return 2
    except DistributionFitError as e:
        log.error(f"Distribution fitting failed: {e}")
        return 3
    except ResourceLimitError as e:
        log.error(f"Resource limit exceeded: {e}")
        return 4
    except KeyboardInterrupt:
        log.info("Shutdown requested. Finishing current tasks...")
        cleanup_partial_results()
        return 130  # Standard SIGINT exit code
    except Exception as e:
        log.exception("Unhandled exception")
        return 255

if __name__ == "__main__":
    sys.exit(cli_main())
```

**Progress Reporting** (per FR-039):

```python
# For long-running grid jobs
for i, config in enumerate(configs):
    log.info(f"Grid progress: {i+1}/{len(configs)} configs completed")
    # Emit progress every 10 configs or 1 minute
```

## VPS Configuration Variability & Benchmarking

**Benchmark Protocol** (addresses CHK079):

1. **Baseline VPS** (8 vCPU, 24 GB RAM, Ubuntu 22.04, Python 3.11):
   - Run `pytest tests/benchmark/` to execute:
     - `test_baseline_run()`: 1,000 paths × 60 steps, single config, measure wall time.
     - `test_grid_run()`: 50 configs × 1,000 paths × 60 steps, `max_workers=6`, measure wall time.
     - `test_distribution_fit()`: Fit Laplace/Student-T on 252-bar window, measure time per fit.
     - `test_mc_throughput()`: Generate 1M steps across 10,000 paths, measure steps/second.
   - Record results in `benchmarks/baseline_8vcpu_24gb.json`.

2. **4 vCPU VPS** (4 vCPU, 12 GB RAM):
   - Run same benchmark suite with `max_workers=3`.
   - Expected: 1.5× slower baseline run (≤15s), 2× slower grid run (≤30m).
   - Record results in `benchmarks/low_spec_4vcpu_12gb.json`.

3. **16 vCPU VPS** (16 vCPU, 48 GB RAM):
   - Run same benchmark suite with `max_workers=14`.
   - Expected: ~1× baseline run (still I/O bound), ~2× faster grid run (≤8m).
   - Record results in `benchmarks/high_spec_16vcpu_48gb.json`.

4. **Validation**:
   - Compare actual vs expected performance (within 20% tolerance).
   - If variance >20%, investigate bottlenecks (CPU vs I/O vs memory).
   - Update performance targets in spec.md if consistent deviation found.

**Benchmark Automation**:
```bash
# scripts/run_benchmarks.sh
pytest tests/benchmark/ -v --benchmark-json=benchmarks/$(hostname)_$(date +%Y%m%d).json
python scripts/compare_benchmarks.py benchmarks/*.json
```

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
