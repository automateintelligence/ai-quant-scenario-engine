# Data Model

Parent: `plan.md` (under `spec.md` per constitution). Children: contracts consume schemas; quickstart/CLI examples rely on these entities.

## Relationships Overview
- `DataSource` → `ReturnDistribution` → `PricePath` → `StrategySignals` → `SimulationRun` → `MetricsReport`.
- `CandidateSelector` → `CandidateEpisode` feeds both conditional backtests and conditional Monte Carlo sampling.
- `RunConfig` ties all components together and is persisted for replay.

## Core Interfaces
- **Strategy**: `generate_signals(price_paths: np.ndarray, features: DataFrame|dict, params: StrategyParams) -> StrategySignals` (must emit both stock and option signals plus OptionSpec when option legs are used).
- **StrategySignals**: `{signals_stock: np.ndarray[int8][n_paths,n_steps], signals_option: np.ndarray[int8][n_paths,n_steps], option_spec: OptionSpec|null, features_used: list[str]}`.
- **OptionPricer**: `price(path_slice: np.ndarray[n_steps], option_spec: OptionSpec) -> np.ndarray[n_steps]` for mark-to-market; supports payoff-only mode for expiry pricing. Implementations: `BlackScholesPricer` (default), `PyVollibPricer` optional, QuantLib/Heston adapter later.
- **ReturnDistribution**: `fit(returns: np.ndarray, min_samples: int=60) -> params` and `sample(n_paths, n_steps, seed) -> np.ndarray[float64][n_paths,n_steps]` with model choices Laplace (default), Student-T, optional GARCH-T.
- **StoragePolicy**: decide `memory` vs `npz` vs `memmap` based on estimated footprint and reuse flag.

## Entities

### DataSource
- **Fields**: `name` (enum: yfinance, schwab_stub), `symbols` (list[str]), `start` (date), `end` (date), `interval` (enum: 1d, 5m, 1m), `source_version` (str), `path` (abs path to Parquet partition).
- **Defaults**: `name=yfinance`, `interval=1d`.
- **Validation**: Must provide OHLCV columns; interval must match requested frequency; warn/drop symbols with insufficient coverage; Parquet partitioning ensures schema consistency.
- **Relationships**: Supplies data for `ReturnDistribution` and `CandidateSelector`.

### ReturnDistribution
- **Fields**: `model` (enum: laplace [default], student_t, garch_t), `params` (dict), `fit_window` (int bars), `seed` (int), `aic`/`bic` (float, optional), `fit_status` (enum: success, warn, fail), `min_samples` (int ≥60).
- **Validation**: Parameter dict must match model; `len(returns) >= min_samples`; seed required for reproducibility; failure triggers `DistributionFitError` and fallback model.
- **Relationships**: Generates `PricePath` samples; params + seed written to run_meta.

```python
# interfaces/distribution.py

from abc import ABC, abstractmethod
import numpy as np

class ReturnDistribution(ABC):
    """Base class for all unconditional or state-conditioned return models."""

    @abstractmethod
    def fit(self, returns: np.ndarray) -> None:
        """Fit parameters from 1D array of log returns."""

    @abstractmethod
    def sample(self, n_paths: int, n_steps: int) -> np.ndarray:
        """
        Produce log-return matrix of shape (n_paths, n_steps).
        """
```

### PricePath
- **Fields**: `paths` (np.ndarray|memmap) shape `[n_paths, n_steps]`, `n_paths` (int), `n_steps` (int), `s0` (float), `storage` (enum: memory, npz, memmap), `seed` (int).
- **Validation**: `n_paths * n_steps * 8 bytes` must respect RAM thresholds (<25% in-memory; ≥25% triggers memmap/npz); seed mandatory for replay; memmap requires file handle in run dir.
- **Relationships**: Consumed by `Strategy` evaluations and option pricers.

### StrategyParams
- **Fields**: `name` (str), `kind` (enum: stock, option), `params` (typed dict: floats/ints/enums per strategy), `position_sizing` (enum: fixed_notional, percent_equity), `fees` (float ≥0), `slippage` (float ≥0).
- **Validation**: Strategy-specific schema enforced; no negative sizing; DTE/strike offsets valid for option strategies.
- **Relationships**: Drives `StrategySignals`; ties to `OptionSpec` when `kind=option`.

### OptionSpec
- **Fields**: `option_type` (enum: call, put), `strike` (float or relative offset string like `atm`, `+0.05`), `maturity_days` (int), `implied_vol` (float >0), `risk_free_rate` (float), `contracts` (int !=0).
- **Validation**: `maturity_days >= simulation horizon`; IV > 0; strike positive; contracts non-zero; warning when IV source missing.
- **Relationships**: Passed to `OptionPricer`; linked from `StrategyParams`/`StrategySignals`.

### StrategySignals
- **Fields**: `signals_stock` (int8 array `[n_paths, n_steps]` in {-1,0,1}), `signals_option` (int8 array `[n_paths, n_steps]`), `option_spec` (OptionSpec|null), `features_used` (list[str]).
- **Validation**: Shapes must match `PricePath`; `option_spec` required when option signals non-empty; `features_used` recorded for reproducibility.
- **Relationships**: Output of `Strategy`; consumed by `SimulationRun`.

```python
# interfaces/strategy.py

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class Strategy(ABC):
    """
    A pluggable trading strategy that consumes price and feature data and
    emits timestamp-aligned signals for stock and/or option positions.
    """

    @abstractmethod
    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Returns something like:
        {
          "stock": np.ndarray,
          "option": {
             "position": np.ndarray,
             "spec": OptionSpec
          }
        }
        """
```

### CandidateSelector
- **Fields**: `name` (str), `rules` (list of predicates e.g., gap %, volume z-score), `feature_requirements` (list[str]), `min_lookback` (int), `min_episodes` (int default 30).
- **Validation**: Rules only use info available at time t; feature dependencies resolved before evaluation; warn on missing features; must produce ≥min_episodes or trigger fallback.
- **Relationships**: Generates `CandidateEpisode` sets; conditions both historical backtests and conditional MC.

### CandidateEpisode
- **Fields**: `symbol` (str), `t0` (timestamp), `horizon` (int bars), `state_features` (dict[str, float]), `selector_name` (str).
- **Validation**: `t0` exists in historical data; `horizon > 0`; `state_features` keys match selector requirements; `maturity_days >= horizon` when paired with option specs.
- **Relationships**: Feeds conditional backtests and conditional MC sampling.

### SimulationRun
- **Fields**: `run_id` (uuid/str), `symbol` (str or list), `config` (RunConfig), `distribution` (ReturnDistribution), `price_paths` (PricePath), `strategies` (list[StrategyParams]), `episodes` (list[CandidateEpisode]|null), `artifacts_path` (abs path).
- **Validation**: Unique run_id; seed in config; conditional runs require non-empty episodes or documented fallback; artifacts path writable.
- **Relationships**: Orchestrates MC generation, strategy execution, metrics, and artifact emission.

### RunConfig
- **Fields**: `n_paths` (int), `n_steps` (int), `seed` (int), `distribution_model` (enum), `data_source` (enum), `selector` (CandidateSelector|null), `grid` (list[StrategyParams]|null), `resource_limits` ({`max_workers` int, `mem_threshold` float, `runtime_budget_s` int}).
- **Validation**: Enforce FR-018 limits; reject configs exceeding RAM/time estimates; seed required; `max_workers <= 6` on 8-core VPS by default.
- **Relationships**: Stored within `SimulationRun` and `run_meta.json` for replay (FR-019).

### MetricsReport
- **Fields**: `per_config_metrics` (list of `{config_id, pnl_stats, drawdown, var, cvar, sharpe, sortino, objective}`), `comparison` (stock vs option summary), `conditional_metrics` (episode-level stats), `logs_path` (str), `plots` (optional paths).
- **Validation**: Objective function defined; metrics arrays align with configs; conditional metrics only when episodes provided.
- **Relationships**: Generated from `SimulationRun`; persisted as JSON/CSV; referenced by quickstart/CLI outputs.

## Relationships Overview
- `DataSource` → `ReturnDistribution` → `PricePath` → `StrategySignals` → `SimulationRun` → `MetricsReport`.
- `CandidateSelector` → `CandidateEpisode` feeds both conditional backtests and conditional MC sampling.
- `RunConfig` ties all components together and is persisted for replay.

## State & Lifecycle
1. Load OHLCV from `DataSource` (validate schema).
2. Fit `ReturnDistribution` (persist params/seed).
3. Generate `PricePath` (select storage policy based on RAM threshold).
4. Produce `StrategySignals` (feature-enriched as configured; includes OptionSpec when needed).
5. Execute `SimulationRun` (stock + option strategies, grid optional) over unconditional or conditional episodes.
6. Emit `MetricsReport` + artifacts + `run_meta.json` for reproducibility.
7. Optional replay uses `run_meta.json` and persisted MC data if available; otherwise regenerates with same seed/params.
## Example SimulationRun (JSON)

```json
{
  "run_id": "2024-11-16T01-compare-aapl",
  "symbol": "AAPL",
  "config": {
    "n_paths": 1000,
    "n_steps": 60,
    "seed": 42,
    "distribution_model": "laplace",
    "data_source": "yfinance",
    "selector": null,
    "resource_limits": {"max_workers": 6, "mem_threshold": 0.25, "runtime_budget_s": 900}
  },
  "distribution": {"model": "laplace", "params": {"loc": 0.0, "scale": 0.012}, "fit_window": 252, "seed": 42},
  "price_paths": {"storage": "memory", "n_paths": 1000, "n_steps": 60, "s0": 190.0, "seed": 42},
  "strategies": [
    {"name": "stock_basic", "kind": "stock", "params": {"threshold": 0.01}, "position_sizing": "fixed_notional", "fees": 0.0, "slippage": 0.0},
    {"name": "call_basic", "kind": "option", "params": {"entry": "atm"}, "position_sizing": "fixed_notional", "fees": 0.0, "slippage": 0.0,
     "option_spec": {"option_type": "call", "strike": "atm", "maturity_days": 60, "implied_vol": 0.25, "risk_free_rate": 0.05, "contracts": 1}}
  ],
  "episodes": null,
  "artifacts_path": "runs/2024-11-16T01-compare-aapl"
}
```
