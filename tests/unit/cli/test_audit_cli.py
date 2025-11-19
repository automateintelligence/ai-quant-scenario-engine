import pandas as pd
import pytest

pytest.importorskip("typer")
from typer.testing import CliRunner

from quant_scenario_engine.cli.main import app
from quant_scenario_engine.distributions.distribution_audit import (
    DistributionAuditResult,
    FitResult,
    ModelScore,
    ModelSpec,
    SimulationMetrics,
    TailMetrics,
    VarBacktestResult,
)


def _result(symbol: str) -> DistributionAuditResult:
    fit = FitResult(
        model_name="laplace",
        log_likelihood=-1.0,
        aic=1.0,
        bic=1.4,
        params={"loc": 0.1, "scale": 0.02},
        n=200,
        converged=True,
        heavy_tailed=True,
        fit_success=True,
        warnings=[],
    )
    tail = TailMetrics(
        model_name="laplace",
        var_emp_95=-0.02,
        var_emp_99=-0.03,
        var_model_95=-0.021,
        var_model_99=-0.031,
        tail_error_95=0.005,
        tail_error_99=0.01,
    )
    var_bt = VarBacktestResult(
        model_name="laplace",
        level=0.99,
        n_obs=60,
        n_breaches=1,
        expected_breaches=0.6,
        kupiec_pvalue=0.8,
        christoffersen_pvalue=0.9,
        passed=True,
    )
    sim = SimulationMetrics(
        model_name="laplace",
        mean_annualized_vol=0.15,
        acf_sq_returns_lag1=0.05,
        mean_max_drawdown=-0.1,
        freq_gt_3pct_move=0.01,
        freq_gt_5pct_move=0.002,
    )
    score = ModelScore(model_name="laplace", total_score=0.95, components={"aic": 0.2, "tail": 0.3, "var": 0.25, "cluster": 0.2})
    return DistributionAuditResult(
        symbol=symbol,
        models=[ModelSpec(name="laplace", cls=None, config={})],
        fit_results=[fit],
        tail_metrics=[tail],
        var_backtests=[var_bt],
        simulation_metrics=[sim],
        scores=[score],
        best_model=ModelSpec(name="laplace", cls=None, config={}),
        best_fit=fit,
    )


def test_audit_cli_formats_output(monkeypatch, tmp_path):
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=40, freq="D"),
        "close": [100 + i * 0.5 for i in range(40)],
    })

    monkeypatch.setattr(
        "quant_scenario_engine.cli.commands.audit_distributions.load_or_fetch",
        lambda symbol, start, end, interval, target: df,
    )
    monkeypatch.setattr(
        "quant_scenario_engine.cli.commands.audit_distributions.audit_distributions_for_symbol",
        lambda **kwargs: _result(kwargs["symbol"]),
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "audit-distributions",
            "--symbol",
            "TEST",
            "--lookback-days",
            "30",
            "--end-date",
            "2024-02-01",
            "--target",
            str(tmp_path),
        ],
    )

    assert res.exit_code == 0
    assert "Distribution Audit for TEST" in res.stdout
    assert "Best Model: laplace" in res.stdout
