"""
distribution_audit.py

Tools for fitting multiple return distribution models (Laplace, Student-T, GARCH-T),
evaluating their suitability for financial returns, and selecting a "best" model
for Monte Carlo simulation based on tail behavior, VaR backtests, and volatility
clustering.

Intended to be callable both as a library module and via a CLI wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from quant_scenario_engine.distributions.base import ReturnDistribution
from quant_scenario_engine.distributions.laplace import LaplaceDistribution
from quant_scenario_engine.distributions.student_t import StudentTDistribution
from quant_scenario_engine.distributions.garch_student_t import GarchStudentTDistribution


# ---------------------------------------------------------------------------
# Data classes for audit results
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """Identifier and configuration used to build a distribution model."""
    name: str                  # "laplace", "student_t", "garch_t"
    cls: type[ReturnDistribution]
    config: Dict[str, object]  # hyperparams or fit settings


@dataclass
class FitResult:
    """Basic fit diagnostics for a model on a given return series."""
    model_name: str
    log_likelihood: float
    aic: float
    bic: float
    params: Dict[str, float]
    excess_kurtosis: float
    heavy_tailed: bool
    fit_success: bool
    fit_message: str = ""


@dataclass
class TailMetrics:
    """Tail-focused diagnostics for empirical vs model-implied returns."""
    model_name: str
    var_emp_95: float
    var_emp_99: float
    var_model_95: float
    var_model_99: float
    tail_error_95: float
    tail_error_99: float


@dataclass
class VarBacktestResult:
    """VaR backtest results on an out-of-sample segment."""
    model_name: str
    level: float               # e.g. 0.95, 0.99
    n_obs: int
    n_breaches: int
    expected_breaches: float
    kupiec_pvalue: float       # unconditional coverage test
    christoffersen_pvalue: float  # independence test
    passed: bool


@dataclass
class SimulationMetrics:
    """
    Summary statistics from Monte Carlo simulations for realism checks:
    volatility, volatility clustering, drawdowns, and extreme-move frequency.
    """
    model_name: str
    mean_annualized_vol: float
    acf_sq_returns_lag1: float
    mean_max_drawdown: float
    freq_gt_3pct_move: float
    freq_gt_5pct_move: float


@dataclass
class ModelScore:
    """Aggregated score summarizing how suitable a model is for MC usage."""
    model_name: str
    total_score: float
    components: Dict[str, float]  # e.g. {"aic": ..., "tail": ..., "var": ..., "cluster": ...}


@dataclass
class DistributionAuditResult:
    """Full audit of all candidate models for a single symbol/timeframe."""
    symbol: str
    models: List[ModelSpec]
    fit_results: List[FitResult]
    tail_metrics: List[TailMetrics]
    var_backtests: List[VarBacktestResult]
    simulation_metrics: List[SimulationMetrics]
    scores: List[ModelScore]
    best_model: Optional[ModelSpec] = None


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------


def fit_candidate_models(
    returns: np.ndarray,
    candidate_models: Sequence[ModelSpec],
    heavy_tail_excess_kurtosis_threshold: float = 1.0,
    heavy_tail_warning_threshold: float = 0.5,
) -> List[FitResult]:
    """
    Fit each candidate model to the return series and compute AIC/BIC and
    heavy-tailed diagnostics.

    Returns a list of FitResult instances.
    """
    n = len(returns)
    results: List[FitResult] = []

    for spec in candidate_models:
        model = spec.cls(**spec.config)
        try:
            model.fit(returns)
            log_like = model.log_likelihood(returns)  # Optional: define on interface
            k = model.num_params()                   # Optional: define on interface

            aic = 2 * k - 2 * log_like
            bic = k * np.log(n) - 2 * log_like

            # Compute empirical excess kurtosis for fitted residuals or returns
            # TODO: define whether to use raw returns or model residuals
            r_centered = returns - np.mean(returns)
            m2 = np.mean(r_centered ** 2)
            m4 = np.mean(r_centered ** 4)
            kurtosis = m4 / (m2 ** 2) if m2 > 0 else np.nan
            excess_kurtosis = kurtosis - 3.0 if np.isfinite(kurtosis) else np.nan

            heavy_tailed = bool(
                np.isfinite(excess_kurtosis) and excess_kurtosis >= heavy_tail_excess_kurtosis_threshold
            )

            results.append(
                FitResult(
                    model_name=spec.name,
                    log_likelihood=log_like,
                    aic=aic,
                    bic=bic,
                    params=model.get_params(),
                    excess_kurtosis=excess_kurtosis,
                    heavy_tailed=heavy_tailed,
                    fit_success=True,
                )
            )

        except Exception as exc:  # noqa: BLE001
            results.append(
                FitResult(
                    model_name=spec.name,
                    log_likelihood=float("nan"),
                    aic=float("inf"),
                    bic=float("inf"),
                    params={},
                    excess_kurtosis=float("nan"),
                    heavy_tailed=False,
                    fit_success=False,
                    fit_message=str(exc),
                )
            )

    return results


def compute_tail_metrics(
    returns: np.ndarray,
    fitted_models: Sequence[ModelSpec],
    levels: Sequence[float] = (0.95, 0.99),
    mc_paths: int = 50_000,
    mc_steps: int = 1,
) -> List[TailMetrics]:
    """
    Compare empirical tail quantiles to model-implied tail quantiles using
    Monte Carlo or closed-form quantiles where available.
    """
    tail_results: List[TailMetrics] = []

    # Empirical quantiles
    emp_q = {
        lvl: np.quantile(returns, 1.0 - lvl)  # for losses in right tail; adjust convention as needed
        for lvl in levels
    }

    for spec in fitted_models:
        model = spec.cls(**spec.config)
        # NOTE: assume model is already fitted or re-fit as needed
        # TODO: decide whether to pass fitted instance instead of spec

        # For simplicity, approximate tail quantiles via simulation of one-step returns
        simulated = model.sample(n_paths=mc_paths, n_steps=mc_steps).reshape(-1)

        var_model_95 = np.quantile(simulated, 1.0 - 0.95)
        var_model_99 = np.quantile(simulated, 1.0 - 0.99)

        var_emp_95 = emp_q[0.95]
        var_emp_99 = emp_q[0.99]

        def tail_err(emp: float, mod: float) -> float:
            if emp == 0:
                return 0.0
            return abs(mod - emp) / (abs(emp) + 1e-12)

        tail_results.append(
            TailMetrics(
                model_name=spec.name,
                var_emp_95=var_emp_95,
                var_emp_99=var_emp_99,
                var_model_95=var_model_95,
                var_model_99=var_model_99,
                tail_error_95=tail_err(var_emp_95, var_model_95),
                tail_error_99=tail_err(var_emp_99, var_model_99),
            )
        )

    return tail_results


def run_var_backtests(
    returns_train: np.ndarray,
    returns_test: np.ndarray,
    fitted_models: Sequence[ModelSpec],
    levels: Sequence[float] = (0.95, 0.99),
) -> List[VarBacktestResult]:
    """
    Perform VaR backtests on an out-of-sample segment using each model.

    This is a stub: fill in Kupiec and Christoffersen tests as needed.
    """
    results: List[VarBacktestResult] = []

    for spec in fitted_models:
        model = spec.cls(**spec.config)
        # TODO: ensure model is fitted on returns_train

        for level in levels:
            # TODO: for each day in returns_test, compute predictive VaR_t
            # Here we just illustrate aggregate breach calculation.
            # Implement proper one-step-ahead forecast logic per model.

            # Placeholder: assume static VaR from train distribution
            simulated = model.sample(n_paths=100_000, n_steps=1).reshape(-1)
            var_level = np.quantile(simulated, 1.0 - level)

            breaches = returns_test < var_level
            n_breaches = int(breaches.sum())
            n_obs = len(returns_test)
            expected_breaches = (1.0 - level) * n_obs

            # TODO: implement Kupiec and Christoffersen tests
            kupiec_pvalue = 1.0  # placeholder
            christoffersen_pvalue = 1.0  # placeholder

            passed = True  # TODO: apply thresholds on p-values

            results.append(
                VarBacktestResult(
                    model_name=spec.name,
                    level=level,
                    n_obs=n_obs,
                    n_breaches=n_breaches,
                    expected_breaches=expected_breaches,
                    kupiec_pvalue=kupiec_pvalue,
                    christoffersen_pvalue=christoffersen_pvalue,
                    passed=passed,
                )
            )

    return results


def simulate_paths_and_metrics(
    symbol: str,
    s0: float,
    fitted_models: Sequence[ModelSpec],
    paths: int = 10_000,
    steps: int = 252,
) -> List[SimulationMetrics]:
    """
    Use each model to generate Monte Carlo paths and compute high-level
    realism metrics: volatility, clustering, drawdowns, extreme-move frequency.
    """
    sim_results: List[SimulationMetrics] = []

    for spec in fitted_models:
        model = spec.cls(**spec.config)
        r = model.sample(n_paths=paths, n_steps=steps)  # log returns
        # Price paths
        log_s = np.log(s0) + np.cumsum(r, axis=1)
        prices = np.exp(log_s)

        # Daily returns on prices for metrics
        px = prices
        ret = np.diff(px, axis=1) / px[:, :-1]

        annualized_vol = np.std(ret, axis=1, ddof=1) * np.sqrt(252.0)
        mean_annual_vol = float(np.mean(annualized_vol))

        # Volatility clustering: ACF of squared returns at lag 1
        sq = ret ** 2
        sq_mean = sq.mean(axis=1, keepdims=True)
        num = np.mean((sq[:, 1:] - sq_mean) * (sq[:, :-1] - sq_mean), axis=1)
        den = np.mean((sq - sq_mean) ** 2, axis=1)
        acf1 = float(np.mean(num / (den + 1e-12)))

        # Max drawdown per path
        roll_max = np.maximum.accumulate(px, axis=1)
        dd = (px - roll_max) / roll_max
        max_dd = dd.min(axis=1)
        mean_max_dd = float(np.mean(max_dd))

        # Extreme move frequencies
        freq_3pct = float(np.mean(np.any(ret <= -0.03, axis=1)))
        freq_5pct = float(np.mean(np.any(ret <= -0.05, axis=1)))

        sim_results.append(
            SimulationMetrics(
                model_name=spec.name,
                mean_annualized_vol=mean_annual_vol,
                acf_sq_returns_lag1=acf1,
                mean_max_drawdown=mean_max_dd,
                freq_gt_3pct_move=freq_3pct,
                freq_gt_5pct_move=freq_5pct,
            )
        )

    return sim_results


def score_models(
    fit_results: Sequence[FitResult],
    tail_metrics: Sequence[TailMetrics],
    var_backtests: Sequence[VarBacktestResult],
    sim_metrics: Sequence[SimulationMetrics],
    weights: Optional[Dict[str, float]] = None,
) -> List[ModelScore]:
    """
    Aggregate multiple diagnostics into a single score per model.

    weights: e.g. {"aic": 0.2, "tail": 0.4, "var": 0.2, "cluster": 0.2}
    """
    if weights is None:
        weights = {"aic": 0.2, "tail": 0.4, "var": 0.2, "cluster": 0.2}

    # Index metrics by model_name for convenience
    fit_by_name = {fr.model_name: fr for fr in fit_results}
    tail_by_name = {tm.model_name: tm for tm in tail_metrics}
    sim_by_name = {sm.model_name: sm for sm in sim_metrics}

    # For VaR tests, aggregate per model across levels
    var_pass_rate: Dict[str, float] = {}
    for vb in var_backtests:
        var_pass_rate.setdefault(vb.model_name, []).append(1.0 if vb.passed else 0.0)
    var_pass_rate = {k: float(np.mean(v)) for k, v in var_pass_rate.items()}

    scores: List[ModelScore] = []

    model_names = {fr.model_name for fr in fit_results}
    for name in model_names:
        fr = fit_by_name[name]
        tm = tail_by_name.get(name)
        sm = sim_by_name.get(name)

        # Normalize / invert metrics as needed (lower AIC is better, lower tail error better, etc.)
        aic_component = -fr.aic if np.isfinite(fr.aic) else -1e9

        tail_component = 0.0
        if tm is not None:
            tail_component = - (tm.tail_error_95 + tm.tail_error_99) / 2.0

        var_component = var_pass_rate.get(name, 0.0)

        cluster_component = 0.0
        if sm is not None:
            # You can design cluster_component as closeness of acf1 to empirical target;
            # here we just reward positive clustering signal.
            cluster_component = sm.acf_sq_returns_lag1

        components = {
            "aic": aic_component,
            "tail": tail_component,
            "var": var_component,
            "cluster": cluster_component,
        }

        total = (
            weights["aic"] * aic_component
            + weights["tail"] * tail_component
            + weights["var"] * var_component
            + weights["cluster"] * cluster_component
        )

        scores.append(
            ModelScore(
                model_name=name,
                total_score=float(total),
                components=components,
            )
        )

    return scores


def select_best_model(
    candidate_models: Sequence[ModelSpec],
    scores: Sequence[ModelScore],
    require_heavy_tails: bool,
    fit_results: Sequence[FitResult],
) -> Optional[ModelSpec]:
    """
    Select the best model by score, optionally requiring it to be heavy-tailed.
    """
    score_by_name = {s.model_name: s for s in scores}
    fit_by_name = {fr.model_name: fr for fr in fit_results}

    # Filter on heavy tail if required
    eligible_names: List[str] = []
    for name in score_by_name.keys():
        if not require_heavy_tails:
            eligible_names.append(name)
        else:
            fr = fit_by_name.get(name)
            if fr and fr.heavy_tailed and fr.fit_success:
                eligible_names.append(name)

    if not eligible_names:
        return None

    best_name = max(eligible_names, key=lambda n: score_by_name[n].total_score)
    for spec in candidate_models:
        if spec.name == best_name:
            return spec
    return None


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------


def audit_distributions_for_symbol(
    symbol: str,
    price_series: pd.Series,
    train_fraction: float = 0.8,
    candidate_models: Optional[Sequence[ModelSpec]] = None,
    s0_override: Optional[float] = None,
    require_heavy_tails: bool = True,
) -> DistributionAuditResult:
    """
    Run the full audit pipeline for a single symbol:

    1. Compute log returns and split into train/test.
    2. Fit Laplace, Student-T, GARCH-T (or provided candidates).
    3. Compute AIC/BIC + heavy-tail diagnostics.
    4. Compute tail metrics (VaR 95/99).
    5. Run VaR backtests on test window.
    6. Simulate paths and realism metrics.
    7. Score models and select the best.
    """
    prices = price_series.dropna().astype(float)
    if len(prices) < 100:
        raise ValueError(f"Not enough data to audit distributions for {symbol}")

    log_returns = np.log(prices / prices.shift(1)).dropna().values

    n = len(log_returns)
    n_train = max(50, int(train_fraction * n))
    r_train = log_returns[:n_train]
    r_test = log_returns[n_train:]

    if candidate_models is None:
        candidate_models = [
            ModelSpec(name="laplace", cls=LaplaceDistribution, config={}),
            ModelSpec(name="student_t", cls=StudentTDistribution, config={}),
            ModelSpec(name="garch_t", cls=GarchStudentTDistribution, config={}),
        ]

    fit_results = fit_candidate_models(r_train, candidate_models)

    tail_metrics = compute_tail_metrics(r_train, candidate_models)

    var_backtests = run_var_backtests(r_train, r_test, candidate_models)

    s0 = float(s0_override if s0_override is not None else prices.iloc[-1])
    sim_metrics = simulate_paths_and_metrics(
        symbol=symbol,
        s0=s0,
        fitted_models=candidate_models,
        paths=10_000,
        steps=252,
    )

    scores = score_models(
        fit_results=fit_results,
        tail_metrics=tail_metrics,
        var_backtests=var_backtests,
        sim_metrics=sim_metrics,
    )

    best_model = select_best_model(
        candidate_models=candidate_models,
        scores=scores,
        require_heavy_tails=require_heavy_tails,
        fit_results=fit_results,
    )

    return DistributionAuditResult(
        symbol=symbol,
        models=list(candidate_models),
        fit_results=fit_results,
        tail_metrics=tail_metrics,
        var_backtests=var_backtests,
        simulation_metrics=sim_metrics,
        scores=scores,
        best_model=best_model,
    )
