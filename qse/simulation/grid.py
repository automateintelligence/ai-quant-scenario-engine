"""Grid runner for parameter exploration (US2).

This module expands strategy parameter grids, performs preflight resource
checks, and executes configurations in parallel with a process pool. Results
are normalized and ranked using the FR-083 objective function.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from typing import Literal, Sequence

import numpy as np

from qse.exceptions import ConfigValidationError, ResourceLimitError
from qse.interfaces.distribution import ReturnDistribution
from qse.models.options import OptionSpec
from qse.schema.strategy import StrategyKind, StrategyParams
from qse.simulation.metrics import MetricsReport
from qse.simulation.run import run_compare
from qse.utils.logging import get_logger
from qse.utils.resources import estimate_footprint_gb

log = get_logger(__name__, component="simulation.grid")

ObjectiveWeights = dict[str, float]


@dataclass(slots=True)
class StrategyGridDefinition:
    """Grid definition for a single strategy.

    Attributes:
        name: Strategy name passed to the strategy factory.
        kind: Strategy kind (stock|option).
        grid: Parameter ranges keyed by parameter name.
        shared: Optional shared defaults (fees, slippage, position_sizing).
        option_spec: OptionSpec required for option strategies.
    """

    name: str
    kind: StrategyKind
    grid: dict[str, Sequence[object]]
    shared: dict[str, object] | None = None
    option_spec: OptionSpec | None = None


@dataclass(slots=True)
class GridCombination:
    """Concrete parameter combination to evaluate."""

    index: int
    stock_params: StrategyParams
    option_params: StrategyParams
    option_spec: OptionSpec


@dataclass(slots=True)
class GridResult:
    """Result for a single grid configuration."""

    index: int
    status: Literal["success", "failed"]
    stock_params: StrategyParams | None
    option_params: StrategyParams | None
    metrics: MetricsReport | None
    objective_score: float | None = None
    normalized_metrics: dict[str, float] | None = None
    error: str | None = None


def _clamp_workers(max_workers: int | None) -> int:
    cpu_count = os.cpu_count() or 1
    if max_workers is None:
        return min(6, cpu_count)
    return max(1, min(max_workers, 6, max(cpu_count - 2, 1)))


def _detect_total_ram_gb() -> float:
    try:  # pragma: no cover - optional dependency
        import psutil

        return float(psutil.virtual_memory().total / 1e9)
    except Exception:
        # Fallback to a conservative default (24 GB VPS per plan)
        return 24.0


def _preflight_resources(
    *,
    n_paths: int,
    n_steps: int,
    config_count: int,
    max_workers: int | None,
    total_ram_gb: float | None = None,
    time_budget_seconds: float = 15 * 60,
) -> None:
    """Validate resource limits before launching the pool."""

    worker_count = _clamp_workers(max_workers)
    estimated_gb = estimate_footprint_gb(n_paths, n_steps)
    concurrent_gb = estimated_gb * worker_count
    ram = total_ram_gb or _detect_total_ram_gb()

    if concurrent_gb >= 0.5 * ram:
        raise ResourceLimitError(
            f"Estimated concurrent footprint {concurrent_gb:.3f} GB exceeds 50% of RAM ({ram:.3f} GB)."
        )
    if concurrent_gb >= 0.25 * ram:
        log.warning(
            "Memory footprint in memmap band",
            extra={"estimated_gb": concurrent_gb, "ram_gb": ram},
        )

    estimated_seconds = config_count * n_paths * n_steps / 1e6
    if estimated_seconds >= time_budget_seconds:
        raise ResourceLimitError(
            f"Estimated runtime {estimated_seconds:.1f}s exceeds budget of {time_budget_seconds:.0f}s"
        )
    if estimated_seconds >= 0.9 * time_budget_seconds:
        log.warning(
            "Grid runtime estimate above 90% of budget",
            extra={"estimated_seconds": estimated_seconds},
        )
    elif estimated_seconds >= 0.5 * time_budget_seconds:
        log.warning(
            "Grid runtime estimate above 50% of budget",
            extra={"estimated_seconds": estimated_seconds},
        )


def expand_strategy_grid(strategy: StrategyGridDefinition) -> list[StrategyParams]:
    """Expand a single strategy grid into StrategyParams combinations."""

    if not strategy.name:
        raise ConfigValidationError("strategy name is required")
    if strategy.kind not in {"stock", "option"}:
        raise ConfigValidationError("strategy kind must be 'stock' or 'option'")

    shared = strategy.shared or {}
    base_params = {
        key: value
        for key, value in shared.items()
        if key not in {"option_spec", "fees", "slippage", "position_sizing"}
    }
    keys = list(strategy.grid.keys())
    values: list[Sequence[object]] = []
    for key in keys:
        value = strategy.grid[key]
        values.append(value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else [value])

    combinations: list[StrategyParams] = []
    if not values:
        combinations.append(
            StrategyParams(
                name=strategy.name,
                kind=strategy.kind,
                params=base_params,
                position_sizing=shared.get("position_sizing", "fixed_notional"),
                fees=float(shared.get("fees", 0.0005)),
                slippage=float(shared.get("slippage", 0.65)),
            )
        )
    else:
        for combo in product(*values):
            params = base_params | dict(zip(keys, combo))
            combinations.append(
                StrategyParams(
                    name=strategy.name,
                    kind=strategy.kind,
                    params=params,
                    position_sizing=shared.get("position_sizing", "fixed_notional"),
                    fees=float(shared.get("fees", 0.0005)),
                    slippage=float(shared.get("slippage", 0.65)),
                )
            )
    return combinations


def _build_combinations(strategy_grids: Sequence[StrategyGridDefinition]) -> list[GridCombination]:
    expanded: list[tuple[StrategyGridDefinition, list[StrategyParams]]] = []
    for grid in strategy_grids:
        expanded.append((grid, expand_strategy_grid(grid)))

    combos: list[GridCombination] = []
    for idx, params_combo in enumerate(product(*(item[1] for item in expanded))):
        stock_params = next((p for p in params_combo if p.kind == "stock"), None)
        option_params = next((p for p in params_combo if p.kind == "option"), None)
        if stock_params is None or option_params is None:
            raise ConfigValidationError("Grid requires both a stock and option strategy")

        option_grid = next(
            (item[0] for item in expanded if item[0].kind == "option"), None
        )
        if option_grid is None or option_grid.option_spec is None:
            raise ConfigValidationError("Option strategies require an option_spec")

        combos.append(
            GridCombination(
                index=idx,
                stock_params=stock_params,
                option_params=option_params,
                option_spec=option_grid.option_spec,
            )
        )
    return combos


def _z_scores(values: list[float]) -> list[float]:
    mean = float(np.mean(values)) if values else 0.0
    std = float(np.std(values)) if values else 0.0
    if std == 0:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def _score_results(results: list[GridResult], weights: ObjectiveWeights) -> list[GridResult]:
    successes = [r for r in results if r.status == "success" and r.metrics]
    if not successes:
        return results

    mean_scores = [r.metrics.mean_pnl for r in successes]  # type: ignore[union-attr]
    sharpe_scores = [r.metrics.sharpe for r in successes]  # type: ignore[union-attr]
    drawdown_scores = [-r.metrics.max_drawdown for r in successes]  # type: ignore[union-attr]
    cvar_scores = [-r.metrics.cvar for r in successes]  # type: ignore[union-attr]

    mean_z = _z_scores(mean_scores)
    sharpe_z = _z_scores(sharpe_scores)
    drawdown_z = _z_scores(drawdown_scores)
    cvar_z = _z_scores(cvar_scores)

    for result, mz, sz, dz, cz in zip(successes, mean_z, sharpe_z, drawdown_z, cvar_z):
        result.normalized_metrics = {
            "mean_pnl_z": mz,
            "sharpe_z": sz,
            "max_drawdown_z": dz,
            "cvar_z": cz,
        }
        result.objective_score = (
            weights.get("w1", 0.3) * mz
            + weights.get("w2", 0.3) * sz
            + weights.get("w3", 0.2) * dz
            + weights.get("w4", 0.2) * cz
        )

    return sorted(results, key=lambda r: (r.objective_score or -math.inf), reverse=True)


def _execute_combination(
    combination: GridCombination,
    *,
    distribution: ReturnDistribution,
    n_paths: int,
    n_steps: int,
    base_seed: int,
    s0: float,
    var_method: str,
    covariance_estimator: str,
    lookback_window: int | None,
) -> GridResult:
    try:
        run_result = run_compare(
            s0=s0,
            distribution=distribution,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=base_seed + combination.index,
            stock_strategy=combination.stock_params.name,
            option_strategy=combination.option_params.name,
            option_spec=combination.option_spec,
            stock_params=combination.stock_params.params,
            option_params=combination.option_params.params,
            var_method=var_method,
            covariance_estimator=covariance_estimator,
            lookback_window=lookback_window,
            compute_features=True,
        )
        return GridResult(
            index=combination.index,
            status="success",
            stock_params=combination.stock_params,
            option_params=combination.option_params,
            metrics=run_result.metrics,
        )
    except Exception as exc:  # pragma: no cover - defensive catch
        log.exception("Grid config failed", extra={"index": combination.index})
        return GridResult(
            index=combination.index,
            status="failed",
            stock_params=combination.stock_params,
            option_params=combination.option_params,
            metrics=None,
            error=str(exc),
        )


def run_grid(
    *,
    strategy_grids: Sequence[StrategyGridDefinition],
    distribution: ReturnDistribution,
    s0: float,
    n_paths: int,
    n_steps: int,
    seed: int,
    objective_weights: ObjectiveWeights | None = None,
    max_workers: int | None = None,
    var_method: str = "historical",
    covariance_estimator: str = "sample",
    lookback_window: int | None = None,
    total_ram_gb: float | None = None,
    time_budget_seconds: float = 15 * 60,
) -> list[GridResult]:
    """Run the parameter grid and return ranked results."""

    combinations = _build_combinations(strategy_grids)
    _preflight_resources(
        n_paths=n_paths,
        n_steps=n_steps,
        config_count=len(combinations),
        max_workers=max_workers,
        total_ram_gb=total_ram_gb,
        time_budget_seconds=time_budget_seconds,
    )

    worker_count = _clamp_workers(max_workers)
    results: list[GridResult] = []

    if worker_count == 1:
        results = [
            _execute_combination(
                combo,
                distribution=distribution,
                n_paths=n_paths,
                n_steps=n_steps,
                base_seed=seed,
                s0=s0,
                var_method=var_method,
                covariance_estimator=covariance_estimator,
                lookback_window=lookback_window,
            )
            for combo in combinations
        ]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    _execute_combination,
                    combo,
                    distribution=distribution,
                    n_paths=n_paths,
                    n_steps=n_steps,
                    base_seed=seed,
                    s0=s0,
                    var_method=var_method,
                    covariance_estimator=covariance_estimator,
                    lookback_window=lookback_window,
                ): combo.index
                for combo in combinations
            }
            for future in as_completed(future_map):
                results.append(future.result())

    weights = objective_weights or {"w1": 0.3, "w2": 0.3, "w3": 0.2, "w4": 0.2}
    return _score_results(results, weights)


__all__ = [
    "GridResult",
    "StrategyGridDefinition",
    "run_grid",
    "expand_strategy_grid",
]
