"""Grid runner for strategy parameter exploration (US2).

Implements parameter expansion, resource-aware execution with optional
process-level parallelism, and objective scoring per FR-083.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from qse.exceptions import ConfigConflictError, ConfigValidationError
from qse.models.options import OptionSpec
from qse.schema.strategy import StrategyParams
from qse.simulation.compare import run_compare
from qse.simulation.metrics import MetricsReport
from qse.utils.logging import get_logger
from qse.utils.resources import select_storage_policy

log = get_logger(__name__, component="grid")


@dataclass(slots=True)
class ObjectiveWeights:
    mean_pnl: float = 0.3
    sharpe: float = 0.3
    max_drawdown: float = 0.2
    cvar: float = 0.2

    def normalized(self) -> "ObjectiveWeights":
        total = self.mean_pnl + self.sharpe + self.max_drawdown + self.cvar
        if total <= 0:
            raise ConfigValidationError("objective_weights must sum to a positive value")
        return ObjectiveWeights(
            mean_pnl=self.mean_pnl / total,
            sharpe=self.sharpe / total,
            max_drawdown=self.max_drawdown / total,
            cvar=self.cvar / total,
        )


@dataclass(slots=True)
class GridEvaluationConfig:
    index: int
    stock: StrategyParams
    option: StrategyParams
    option_spec: OptionSpec


@dataclass(slots=True)
class GridResult:
    config_index: int
    stock_params: StrategyParams
    option_params: StrategyParams
    metrics: MetricsReport | None = None
    status: str = "pending"
    error: str | None = None
    normalized_metrics: dict[str, float] | None = None
    objective_score: float | None = None


def _expand_param_grid(grid: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values: list[list[Any]] = []
    for key in keys:
        raw_val = grid[key]
        if isinstance(raw_val, (str, bytes)):
            vals = [raw_val]
        elif isinstance(raw_val, Iterable):
            vals = list(raw_val)
        else:
            vals = [raw_val]
        if not vals:
            raise ConfigValidationError(f"grid parameter '{key}' has no values")
        values.append(vals)
    combos: list[dict[str, Any]] = []
    for combo in product(*values):
        combos.append({k: v for k, v in zip(keys, combo)})
    return combos


def _strategy_params_from_grid(cfg: dict[str, Any]) -> list[StrategyParams]:
    name = cfg.get("name")
    kind = cfg.get("kind")
    grid = cfg.get("grid") or {}
    shared = cfg.get("shared") or {}
    if not name or not kind:
        raise ConfigValidationError("strategy grid entries require name and kind")

    base_params = shared.get("params") or {}
    sizing = shared.get("sizing", "fixed_notional")
    fees = shared.get("fees", 0.0005)
    slippage = shared.get("slippage", 0.65)

    params: list[StrategyParams] = []
    for combo in _expand_param_grid(grid):
        merged = {**base_params, **combo}
        params.append(
            StrategyParams(
                name=name,
                kind=kind,
                params=merged,
                position_sizing=sizing,
                fees=float(fees),
                slippage=float(slippage),
            )
        )
    return params


def _build_option_spec(shared: dict[str, Any] | None, defaults: dict[str, Any]) -> OptionSpec:
    option_spec_cfg = (shared or {}).get("option_spec", {})
    merged = {**defaults, **option_spec_cfg}
    required = ["option_type", "strike", "maturity_days", "implied_vol", "risk_free_rate", "contracts"]
    missing = [k for k in required if k not in merged]
    if missing:
        raise ConfigValidationError(f"option_spec missing required fields: {missing}")
    return OptionSpec(
        option_type=merged["option_type"],
        strike=merged["strike"],
        maturity_days=merged["maturity_days"],
        implied_vol=merged["implied_vol"],
        risk_free_rate=merged["risk_free_rate"],
        contracts=merged["contracts"],
        iv_source=merged.get("iv_source", "config_default"),
        early_exercise=merged.get("early_exercise", False),
    )


def build_grid_configs(
    strategy_grids: Sequence[dict[str, Any]],
    *,
    default_stock: str,
    default_option: str,
    option_spec_defaults: dict[str, Any],
) -> list[GridEvaluationConfig]:
    stock_param_sets: list[StrategyParams] = []
    option_param_sets: list[tuple[StrategyParams, OptionSpec]] = []

    for cfg in strategy_grids:
        if cfg.get("kind") == "stock":
            stock_param_sets.extend(_strategy_params_from_grid(cfg))
        elif cfg.get("kind") == "option":
            shared = cfg.get("shared") or {}
            option_spec = _build_option_spec(shared, option_spec_defaults)
            for params in _strategy_params_from_grid(cfg):
                option_param_sets.append((params, option_spec))
        else:
            raise ConfigValidationError(f"unrecognized strategy kind: {cfg.get('kind')}")

    if not stock_param_sets:
        stock_param_sets.append(StrategyParams(name=default_stock, kind="stock", params={}))

    if not option_param_sets:
        default_spec = _build_option_spec({}, option_spec_defaults)
        option_param_sets.append(
            (StrategyParams(name=default_option, kind="option", params={}), default_spec)
        )

    configs: list[GridEvaluationConfig] = []
    for idx, (stock_params, (option_params, option_spec)) in enumerate(
        product(stock_param_sets, option_param_sets)
    ):
        configs.append(
            GridEvaluationConfig(
                index=idx,
                stock=stock_params,
                option=option_params,
                option_spec=option_spec,
            )
        )
    return configs


def _run_single_config(
    cfg: GridEvaluationConfig,
    *,
    distribution,
    s0: float,
    n_paths: int,
    n_steps: int,
    seed: int,
    var_method: str,
    covariance_estimator: str,
) -> GridResult:
    res = GridResult(config_index=cfg.index, stock_params=cfg.stock, option_params=cfg.option)
    try:
        result = run_compare(
            s0=s0,
            distribution=distribution,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            stock_strategy=cfg.stock.name,
            option_strategy=cfg.option.name,
            stock_params=cfg.stock.params,
            option_params=cfg.option.params,
            option_spec=cfg.option_spec,
            var_method=var_method,
            covariance_estimator=covariance_estimator,
        )
        res.metrics = result.metrics
        res.status = "success"
    except Exception as exc:  # pragma: no cover - defensive error path
        res.status = "failed"
        res.error = str(exc)
    return res


def _zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0:
        return [0.0 for _ in values]
    return [float((v - mean) / std) for v in values]


def apply_objective_scores(
    results: list[GridResult], weights: ObjectiveWeights | None = None
) -> list[GridResult]:
    weights = (weights or ObjectiveWeights()).normalized()
    successful = [r for r in results if r.metrics is not None]
    if not successful:
        raise ConfigConflictError("All grid configurations failed; no metrics to score")

    mean_pnl = [r.metrics.mean_pnl for r in successful]
    sharpe_vals = [r.metrics.sharpe for r in successful]
    mdd_vals = [r.metrics.max_drawdown for r in successful]
    cvar_vals = [r.metrics.cvar for r in successful]

    z_mean = _zscore(mean_pnl)
    z_sharpe = _zscore(sharpe_vals)
    z_mdd = _zscore(mdd_vals)
    z_cvar = _zscore(cvar_vals)

    for r, zm, zs, zmd, zc in zip(successful, z_mean, z_sharpe, z_mdd, z_cvar):
        r.normalized_metrics = {
            "mean_pnl_z": zm,
            "sharpe_z": zs,
            "max_drawdown_z": zmd,
            "cvar_z": zc,
        }
        r.objective_score = (
            weights.mean_pnl * zm
            + weights.sharpe * zs
            + weights.max_drawdown * (-zmd)
            + weights.cvar * (-zc)
        )

    return sorted(results, key=lambda r: (r.objective_score or float("-inf")), reverse=True)


def write_grid_results(path: Path, results: list[GridResult], *, weights: ObjectiveWeights) -> None:
    payload: list[dict[str, Any]] = []
    for r in results:
        entry: dict[str, Any] = {
            "config_index": r.config_index,
            "status": r.status,
            "error": r.error,
            "stock_params": asdict(r.stock_params),
            "option_params": asdict(r.option_params),
            "objective_score": r.objective_score,
            "normalized_metrics": r.normalized_metrics,
        }
        if r.metrics:
            entry["metrics"] = asdict(r.metrics)
        payload.append(entry)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            {
                "weights": asdict(weights.normalized()),
                "results": payload,
            },
            indent=2,
        )
    )
    tmp.replace(path)


def run_grid(
    *,
    distribution,
    s0: float,
    n_paths: int,
    n_steps: int,
    seed: int,
    strategy_grids: Sequence[dict[str, Any]],
    option_spec_defaults: dict[str, Any],
    default_stock_strategy: str = "stock_basic",
    default_option_strategy: str = "call_basic",
    max_workers: int | None = None,
    objective_weights: ObjectiveWeights | None = None,
    var_method: str = "historical",
    covariance_estimator: str = "sample",
    output_path: Path | None = None,
) -> list[GridResult]:
    # Resource preflight (FR-018/FR-023)
    policy, estimated_gb = select_storage_policy(n_paths, n_steps)
    log.info("Storage policy selected", extra={"policy": policy, "estimated_gb": round(estimated_gb, 4)})

    configs = build_grid_configs(
        strategy_grids,
        default_stock=default_stock_strategy,
        default_option=default_option_strategy,
        option_spec_defaults=option_spec_defaults,
    )

    worker_default = min(6, os.cpu_count() or 1)
    if max_workers is None:
        worker_count = worker_default
    else:
        worker_count = max(1, min(max_workers, (os.cpu_count() or 3) - 2))
        if worker_count != max_workers:
            log.warning(
                "max_workers clamped to reserve capacity",
                extra={"requested": max_workers, "clamped": worker_count},
            )

    start_time = time.time()
    halfway_warned = False
    near_limit_warned = False

    results: list[GridResult] = []

    def _maybe_warn_time() -> None:
        nonlocal halfway_warned, near_limit_warned
        elapsed = time.time() - start_time
        if not halfway_warned and elapsed >= 450:  # 50% of 15m budget
            log.warning("Grid runtime exceeded 50% of budget", extra={"elapsed_seconds": round(elapsed, 2)})
            halfway_warned = True
        if not near_limit_warned and elapsed >= 810:  # 90% of 15m budget
            log.error("Grid runtime exceeded 90% of budget", extra={"elapsed_seconds": round(elapsed, 2)})
            near_limit_warned = True

    if worker_count <= 1 or len(configs) == 1:
        for cfg in configs:
            results.append(
                _run_single_config(
                    cfg,
                    distribution=distribution,
                    s0=s0,
                    n_paths=n_paths,
                    n_steps=n_steps,
                    seed=seed,
                    var_method=var_method,
                    covariance_estimator=covariance_estimator,
                )
            )
            _maybe_warn_time()
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    _run_single_config,
                    cfg,
                    distribution=distribution,
                    s0=s0,
                    n_paths=n_paths,
                    n_steps=n_steps,
                    seed=seed,
                    var_method=var_method,
                    covariance_estimator=covariance_estimator,
                ): cfg.index
                for cfg in configs
            }
            for future in as_completed(future_map):
                result = future.result()
                if result.status == "failed":
                    log.error("Config failed", extra={"config_index": result.config_index, "error": result.error})
                results.append(result)
                _maybe_warn_time()

    # Apply objective scoring + ranking (FR-083, SC-003)
    results = apply_objective_scores(results, weights=objective_weights)

    if output_path:
        write_grid_results(output_path, results, weights=objective_weights or ObjectiveWeights())

    return results
