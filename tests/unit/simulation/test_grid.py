from pathlib import Path

from qse.distributions.laplace import LaplaceDistribution
from qse.schema.strategy import StrategyParams
from qse.simulation.grid import (
    GridResult,
    ObjectiveWeights,
    apply_objective_scores,
    build_grid_configs,
    run_grid,
)
from qse.simulation.metrics import MetricsReport


def _option_defaults() -> dict:
    return {
        "option_type": "call",
        "strike": 100.0,
        "maturity_days": 10,
        "implied_vol": 0.2,
        "risk_free_rate": 0.01,
        "contracts": 1,
    }


def test_build_grid_configs_cartesian_expansion():
    strategy_grids = [
        {
            "name": "stock_basic",
            "kind": "stock",
            "grid": {"short_window": [5, 10], "long_window": [20, 30]},
            "shared": {"fees": 0.001, "slippage": 0.5},
        },
        {
            "name": "option_call",
            "kind": "option",
            "grid": {"momentum_window": [1, 2]},
            "shared": {"option_spec": {"strike": 105}},
        },
    ]

    configs = build_grid_configs(
        strategy_grids,
        default_stock="stock_basic",
        default_option="option_call",
        option_spec_defaults=_option_defaults(),
    )

    # 2 x 2 stock combos crossed with 2 option combos = 8 configs
    assert len(configs) == 8
    assert all(cfg.option_spec.strike in {100.0, 105} for cfg in configs)


def test_apply_objective_scores_assigns_scores_and_sorting():
    base_stock = StrategyParams(name="stock_basic", kind="stock", params={})
    base_option = StrategyParams(name="option_call", kind="option", params={})
    metrics_high = MetricsReport(
        mean_pnl=5.0,
        median_pnl=4.0,
        max_drawdown=-0.1,
        sharpe=1.0,
        sortino=1.0,
        var=-0.02,
        cvar=-0.03,
        var_method="historical",
        lookback_window=None,
        covariance_estimator="sample",
        bankruptcy_rate=0.0,
        early_exercise_events=0,
    )
    metrics_low = MetricsReport(
        mean_pnl=1.0,
        median_pnl=0.5,
        max_drawdown=-0.3,
        sharpe=0.2,
        sortino=0.2,
        var=-0.05,
        cvar=-0.06,
        var_method="historical",
        lookback_window=None,
        covariance_estimator="sample",
        bankruptcy_rate=0.0,
        early_exercise_events=0,
    )

    results = [
        GridResult(config_index=0, stock_params=base_stock, option_params=base_option, metrics=metrics_high, status="success"),
        GridResult(config_index=1, stock_params=base_stock, option_params=base_option, metrics=metrics_low, status="success"),
    ]

    ranked = apply_objective_scores(results, weights=ObjectiveWeights())
    assert ranked[0].objective_score is not None
    assert ranked[0].objective_score >= ranked[1].objective_score
    assert ranked[0].normalized_metrics is not None


def test_run_grid_sequential_produces_ranked_results(tmp_path: Path):
    dist = LaplaceDistribution()
    dist.loc = 0.0
    dist.scale = 0.02

    strategy_grids = [
        {"name": "stock_basic", "kind": "stock", "grid": {"short_window": [3, 4]}},
        {"name": "option_call", "kind": "option", "grid": {"momentum_window": [1]}},
    ]

    results = run_grid(
        distribution=dist,
        s0=100.0,
        n_paths=6,
        n_steps=30,
        seed=123,
        strategy_grids=strategy_grids,
        option_spec_defaults=_option_defaults(),
        max_workers=1,
        objective_weights=ObjectiveWeights(),
        output_path=tmp_path / "grid_results.json",
    )

    assert results
    assert results == sorted(results, key=lambda r: r.objective_score or float("-inf"), reverse=True)
    assert all(r.metrics is not None for r in results if r.status == "success")
    payload = (tmp_path / "grid_results.json").read_text()
    assert "objective_score" in payload
