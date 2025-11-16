# Implementation Plan: Backtesting & Strategy Spec Authoring (Quant Scenario Engine)

**Branch**: `001-mvp-pipeline`  
**Date**: 2025-11-16  
**Spec**: `specs/001-mvp-pipeline/spec.md` (authoritative)  
**Parents/Children**: Follows constitution; parent is spec.md, siblings include research.md, data-model.md, quickstart.md, contracts/, tasks.md.

## Summary
Build a CPU-only Quant Scenario Engine that loads Parquet OHLCV, fits heavy-tailed return models (Laplace default; Student-T/GARCH-T options), generates Monte Carlo paths, runs stock and option strategies, and emits reproducible artifacts. Support CLI commands `compare`, `grid`, `screen`, `conditional`, and `replay`, with deterministic seeding, resource-aware storage policy, and candidate-based backtesting/MC per spec FR-001..FR-040 and FR-CAND-001..006.

## Technical Context & Constraints
- **Runtime**: Python 3.11 on 8 vCPU / 24 GB RAM VPS; CPU-only.
- **Data**: Parquet canonical storage (DM-004..014), 1d/5m/1m bars; features stored separately. Default yfinance, Schwab stub optional; schema validation on load (FR-027), fingerprint + drift detection (FR-019, FR-028, DM-013).
- **MC**: Laplace default; Student-T/GARCH-T optional. Memory estimator (`n_paths*n_steps*8*1.1`) and storage policy thresholds per FR-013/FR-023; non-positive/overflow rejection per FR-022; implausible parameter bounds per FR-020/FR-037.
- **Pricing**: Black–Scholes default, swap-friendly (FR-016); option maturity/ATM edge handling per spec.
- **Config**: CLI > ENV > YAML precedence with defaults and incompatible-combo fail-fast (FR-009, FR-024, FR-025); component swap logging (FR-026).
- **Performance**: Budgets per FR-018 (≤10s baseline, ≤15m grid), throughput targets captured; resource limit enforcement + warnings/abort.
- **Observability**: Structured JSON logs, progress, diagnostics (FR-039, FR-040); run_meta immutability and atomic writes (FR-030).
- **Reproducibility**: Seeds applied everywhere (FR-012/021), capture package versions/system config/git SHA, data fingerprints (FR-019, FR-028, FR-034).
- **Assumptions**: Single-user execution, pre-downloaded Parquet, 8 vCPU/24 GB RAM (FR-018 context); revisit if violated.

## Workstreams
1) **Data & Schema**  
   - Implement DataSource adapters (yfinance default, Schwab stub) with retries and drift detection (FR-001, FR-017, FR-027, FR-028).  
   - Enforce missing-data tolerances and gap handling (FR-010, FR-029); align macro series (FR-014).  
   - Persist fingerprints and schema metadata in run_meta (FR-019, FR-034).

2) **MC Models & Storage Policy**  
   - Implement ReturnDistribution interface + Laplace/Student-T/GARCH-T fits with bounds, convergence limits, and implausible-parameter checks (FR-002, FR-020, FR-037).  
   - Log-return → price transform with overflow/non-positive rejection (FR-022).  
   - Memory estimator + policy: in-memory <25% RAM; memmap/npz ≥25%; abort ≥50% (FR-013, FR-023); record in run_meta.

3) **Strategies & Pricing**  
   - Stock/option strategy interfaces; option pricer abstraction with Black–Scholes default and plug-ins (FR-004, FR-016).  
   - Handle option-specific edge cases (maturity vs horizon, ATM precision, invalid IV) with structured errors (FR-022).

4) **CLI & Config**  
   - Typer CLIs for `compare`, `grid`, `screen`, `conditional`, `replay` with parameter validation against contracts (FR-005, FR-033).  
   - Config precedence (FR-024) and defaulting/incompatibility checks (FR-009, FR-025); audit component swaps (FR-026).  
   - Fail-fast + recoverable fallbacks logged per FR-038.

5) **Candidate Selection & Conditional Flows**  
   - CandidateSelector abstraction and default gap/volume rule (FR-CAND-001, -006).  
   - Episode construction `(symbol, t0, horizon, state_features)` (FR-CAND-002, FR-035); screening outputs per US4.  
   - Conditional backtest + conditional MC with bootstrap + parametric refit; fallback order logged (FR-CAND-004/005/036).  
   - Selector sparsity/zero-candidate and replay data-drift handling (FR-019, SC-011/012/020).

6) **Resource Limits, Observability, Reproducibility**  
   - Enforce time/memory budgets and worker caps (FR-018, FR-023).  
   - Structured logging + progress + diagnostics (FR-039/040); audit trail completeness.  
   - run_meta: seeds, versions (Python/pkg), git SHA, system config, data fingerprints, storage policy, fallbacks (FR-019, FR-021, FR-030, FR-034).

## Phases & Milestones
- **Phase 0: Architecture & Contracts**  
  - Finalize interfaces (ReturnDistribution, OptionPricer, CandidateSelector, RunConfig) and storage policy rules.  
  - Draft contracts/openapi for CLI/config schemas; update data-model.md and quickstart.md with resolved defaults.

- **Phase 1: Core Engine & CLI (compare)**  
  - Data adapters + schema validation + fingerprints.  
  - Laplace + Student-T fit/sample; MC engine with estimator/policy; stock vs option simulator; run_meta emission.  
  - `compare` CLI with config precedence, seed handling, artifacts (metrics JSON/CSV, run_meta).

- **Phase 2: Grids, Screening, Conditional**  
  - Grid runner with resource caps; progress + partial failure handling.  
  - Candidate selector + screening CLI; episode builder.  
  - Conditional backtest + conditional MC (bootstrap + parametric) with fallbacks; `replay` CLI enforcing drift checks.

- **Phase 3: Hardening & Edge Cases**  
  - Option-pricing edge handling, small/degenerate path cases, bankruptcy/zero-vol paths, max_workers boundaries, single-config grid.  
  - Performance validation vs budgets; observability/audit completeness; change/versioning hooks.

## Deliverables (Definition of Done)
- Updated `plan.md`, `data-model.md`, `quickstart.md`, `contracts/` aligning to spec; `tasks.md` with executable backlog.  
- Implemented CLIs (`compare`, `grid`, `screen`, `conditional`, `replay`) meeting FR-005/033 and US1–US8 acceptance scenarios.  
- MC + storage policy + option pricer per FR-002/013/016/020/022/023/037.  
- Candidate flows per FR-CAND-001..006; episode artifacts and conditional metrics per SC-010/011/012.  
- run_meta content: seeds, versions, git SHA, system config, data fingerprints, storage policy, fallbacks, drift status.  
- Tests: unit for distributions/pricers/config validation; integration for CLI commands; property/boundary tests for MC reproducibility and resource thresholds; coverage ≥80%.

## Risks & Mitigations
- **Performance budget miss**: profile MC and pricer hotspots; reduce paths/steps or swap pricer; ensure memmap fallback works.
- **Data drift/quality**: strict schema checks, fingerprints, and drift blocking; clear warnings with override flag.
- **Resource exhaustion**: preflight estimator + hard caps; partial results persisted for grids; fail-fast with structured errors.
- **Reproducibility gaps**: capture full environment (git SHA, packages, system config) and seeds in run_meta; deterministic config precedence.
- **Config complexity**: contracts as single source; defaults documented; conflicts rejected early.

## Data Loading & Caching Architecture

**On-Demand Fetching with Local Cache** (per FR-001, FR-085, FR-086):

```python
# data/data_loader.py

from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

class DataLoader:
    """Load OHLCV with automatic on-demand fetching and caching."""

    def __init__(self, cache_dir: Path, data_source_factory):
        self.cache_dir = cache_dir
        self.data_source_factory = data_source_factory

    def load_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
        allow_stale: bool = False,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """Load OHLCV with cache-aside pattern and corporate action detection (FR-085)."""

        # Determine cache paths
        cache_path = self.cache_dir / interval / f"{symbol}.parquet"
        cache_meta_path = cache_path.with_suffix(".meta.json")

        # 0. Force refresh bypasses cache entirely (FR-085)
        if force_refresh:
            log.info(f"Force refresh requested for {symbol}, bypassing cache...")
            df = self._fetch_from_source(symbol, start, end, interval)
            self._write_cache(cache_path, df, symbol, interval)
            return df

        # 1. Check cache

        if cache_path.exists() and cache_meta_path.exists():
            # Load cache metadata
            cache_meta = json.loads(cache_meta_path.read_text())
            cached_start = pd.Timestamp(cache_meta["start"])
            cached_end = pd.Timestamp(cache_meta["end"])
            fetched_at = pd.Timestamp(cache_meta["fetched_at"])

            # Check staleness (FR-085)
            staleness_threshold = timedelta(days=1 if interval == "1d" else 0)
            is_stale = (datetime.now() - fetched_at) > staleness_threshold

            # Check coverage
            requested_start = pd.Timestamp(start)
            requested_end = pd.Timestamp(end)
            has_full_coverage = (cached_start <= requested_start and
                                 cached_end >= requested_end)

            # Use cache if fresh and complete
            if has_full_coverage and not is_stale:
                log.info(f"Using cached data for {symbol} ({start} to {end})")
                df = pd.read_parquet(cache_path)
                return df.loc[start:end]

            # Fetch incremental if partial coverage
            elif not is_stale and cached_end < requested_end:
                log.info(f"Fetching incremental data for {symbol} "
                         f"({cached_end} to {end})")

                # Corporate action detection (FR-085)
                # Fetch overlapping bar to detect stock splits/dividends
                overlap_start = cached_end
                overlap_end = cached_end + timedelta(days=1)
                overlap_data = self._fetch_from_source(
                    symbol, overlap_start, overlap_end, interval
                )

                if len(overlap_data) > 0:
                    cached_last_close = cache_meta.get("last_close")
                    fresh_close = overlap_data.iloc[0]["close"]

                    if cached_last_close is not None:
                        divergence = abs(fresh_close - cached_last_close) / cached_last_close
                        if divergence > 0.01:  # 1% threshold
                            log.warning(
                                f"Historical prices adjusted for {symbol} "
                                f"(cached: {cached_last_close:.2f}, "
                                f"fresh: {fresh_close:.2f}, "
                                f"divergence: {divergence:.2%}). "
                                f"Likely stock split/dividend adjustment. "
                                f"Triggering full refresh."
                            )
                            # Full refresh instead of incremental
                            df = self._fetch_from_source(symbol, start, end, interval)
                            self._write_cache(cache_path, df, symbol, interval)
                            return df

                # No corporate action detected, proceed with incremental append
                new_data = self._fetch_from_source(
                    symbol, cached_end, end, interval
                )
                df = pd.read_parquet(cache_path)
                df = pd.concat([df, new_data]).drop_duplicates()
                self._write_cache(cache_path, df, symbol, interval)
                return df.loc[start:end]

            # Stale cache: re-fetch or use if allowed
            elif is_stale:
                try:
                    log.info(f"Cache stale for {symbol}, re-fetching...")
                    df = self._fetch_from_source(symbol, start, end, interval)
                    self._write_cache(cache_path, df, symbol, interval)
                    return df
                except DataSourceError as e:
                    if allow_stale:
                        log.warning(f"Using stale cache due to fetch failure: {e}")
                        df = pd.read_parquet(cache_path)
                        return df.loc[start:end]
                    else:
                        raise

        # 2. Cache miss: fetch from source
        log.info(f"Cache miss for {symbol}, fetching from data source...")
        df = self._fetch_from_source(symbol, start, end, interval)
        self._write_cache(cache_path, df, symbol, interval)
        return df

    def _fetch_from_source(self, symbol, start, end, interval) -> pd.DataFrame:
        """Fetch with retries and exponential backoff (FR-086)."""
        source = self.data_source_factory.create(self.config.data_source)

        for attempt in range(3):
            try:
                df = source.fetch(symbol, start, end, interval)
                self._validate_schema(df)
                return df
            except Exception as e:
                if attempt < 2:
                    backoff = 2 ** attempt  # 1s, 2s, 4s
                    log.warning(f"Fetch attempt {attempt+1} failed: {e}. "
                                f"Retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    raise DataSourceError(
                        f"Unable to fetch {symbol} from {source.name} "
                        f"after 3 retries. Check network/API status."
                    )

    def _write_cache(self, path: Path, df: pd.DataFrame, symbol: str, interval: str):
        """Write Parquet + metadata atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write data
        df.to_parquet(path, compression="snappy")

        # Write metadata (FR-085: include last_close for corporate action detection)
        meta = {
            "symbol": symbol,
            "interval": interval,
            "start": str(df.index[0]),
            "end": str(df.index[-1]),
            "source": self.config.data_source,
            "fetched_at": datetime.utcnow().isoformat(),
            "last_close": float(df.iloc[-1]["close"])  # For corporate action detection
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

        log.info(f"Cached {len(df)} bars for {symbol} to {path}")
```

**Cache Management**:
- **Directory structure**: `data/historical/{interval}/{symbol}.parquet` + `{symbol}.meta.json`
- **Staleness detection**: Compare `fetched_at` to current time with interval-specific thresholds
- **Incremental updates**: Append new data to existing cache rather than full re-download
- **Corporate action detection** (FR-085): Before incremental append, fetch overlapping bar to detect stock splits/dividends via >1% price divergence. Triggers full refresh with warning if detected.
- **Force refresh**: `--force_refresh` flag bypasses cache entirely for manual control after known corporate actions
- **Atomic writes**: Write data + metadata together to prevent partial corruption
- **Graceful degradation**: Use stale cache on fetch failure if `--allow_stale_cache` flag set

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

| Operation | Target | Notes |
|-----------|--------|-------|
| Baseline compare (1k×60) | ≤10s on 8 vCPU VPS | Includes data load, fit, MC, strategies |
| Grid (≤50 configs, 1k×60) | ≤15m | With `max_workers` tuned for 8 vCPU |
| Laplace/Student-T fit | ≤1s per symbol window | Heavier models (GARCH) optional |
| MC generation | ≥50k steps/s aggregate | Vectorized NumPy/Numba |
| Memory footprint | <25% RAM in-memory | Auto memmap beyond threshold |
| Run artifact write | ≤1s per run | Parquet/JSON writes amortized |

## Reference Interfaces & Code Snippets

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
