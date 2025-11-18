"""
distribution_audit.py

Tools for fitting multiple return distribution models (Laplace, Student-T, GARCH-T),
evaluating their suitability for financial returns, and selecting a "best" model
for Monte Carlo simulation based on tail behavior, VaR backtests, and volatility
clustering.

ADAPTATION STATUS (Per T137 - User Story 6a):
This sophisticated stub must be adapted to align with spec.md US6a (lines 131-271)
and tasks.md Phase 7a (T137-T182). The core architecture is sound, but specific
modifications are needed to meet all acceptance scenarios (AS1-AS12).

KEY ADAPTATIONS NEEDED:
1. Preprocessing pipeline (AS1, T138-T142): Add stationarity checks, outlier detection
2. Goodness-of-fit enhancements (AS2, T143-T145): Add excess kurtosis validation (≥1.0)
3. Tail diagnostics (AS3, T146-T149): Add 99.5% quantile, comprehensive Q-Q plots
4. VaR backtesting (AS4, T150-T155): Implement Kupiec/Christoffersen statistical tests
5. Model selection scoring (AS6, T164-T168): Update weights to (0.2, 0.4, 0.3, 0.1)
6. Caching layer (AS7, T169-T171): Add 30-day TTL cache with proper key structure
7. Reproducibility (AS8, T172): Add deterministic seeding throughout
8. Integration hooks (AS9-10, T173-T176): Auto-load validated models for US1/US6
9. CLI command (T177-T179): Wire to audit-distributions command
10. Enhanced error handling (AS11-12, T180-T182): Add convergence diagnostics

See tasks.md Phase 7a for detailed implementation tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from quant_scenario_engine.distributions.fitters.garch_t_fitter import GarchTFitter
from quant_scenario_engine.distributions.fitters.laplace_fitter import LaplaceFitter
from quant_scenario_engine.distributions.fitters.student_t_fitter import StudentTFitter
from quant_scenario_engine.distributions.metrics.information_criteria import aic as calc_aic, bic as calc_bic
from quant_scenario_engine.distributions.models import FitResult
from quant_scenario_engine.exceptions import DistributionFitError
from quant_scenario_engine.utils.logging import get_logger

log = get_logger(__name__, component="distribution_audit")


# ---------------------------------------------------------------------------
# Data classes for audit results
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """Identifier and configuration used to build a distribution model."""
    name: str                  # "laplace", "student_t", "garch_t"
    cls: object
    config: Dict[str, object]  # hyperparams or fit settings


@dataclass
class TailMetrics:
    """
    Tail-focused diagnostics for empirical vs model-implied returns.

    TODO [T146, AS3]: Add 99.5% quantile fields per spec.md US6a AS3:
    - var_emp_995: float
    - var_model_995: float
    - tail_error_995: float
    """
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
    fit_results: List[FitResult]  # Fixed: was FitSummary (typo)
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
) -> List[FitResult]:
    """
    Fit each candidate model to the return series and compute AIC/BIC and
    heavy-tailed diagnostics.

    Returns a list of FitResult instances.

    TODO [T138-T142, AS1]: Add preprocessing pipeline before fitting:
    - T138: Stationarity checks (ADF test, warn if p > 0.05)
    - T139: Outlier detection (clip at ±5σ or flag high leverage points)
    - T140: Return normalization (ensure zero mean, log returns)
    - T141: Minimum sample size validation (warn if n < 252 for GARCH-t)

    TODO [T143-T145, AS2]: Enhance goodness-of-fit validation:
    - T143: Add excess kurtosis check (must be ≥1.0 per spec.md US6a AS2)
    - T144: Generate Q-Q plots for diagnostic output
    - T145: Add KS test for distribution fit quality
    """
    n = len(returns)
    results: List[FitResult] = []

    for spec in candidate_models:
        try:
            fitter = spec.cls if hasattr(spec.cls, "fit") else spec.cls()
            fitted: FitResult = fitter.fit(returns)  # type: ignore[call-arg]
            results.append(fitted)
        except DistributionFitError as exc:
            log.warning("model fit failed", extra={"model": spec.name, "error": str(exc)})
            results.append(
                FitResult(
                    model_name=spec.name,
                    log_likelihood=float("nan"),
                    aic=float("inf"),
                    bic=float("inf"),
                    params={},
                    n=len(returns),
                    heavy_tailed=False,
                    fit_success=False,
                    converged=False,
                    error=str(exc),
                    warnings=[str(exc)],
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

    TODO [T146-T149, AS3]: Enhance tail diagnostics per spec.md US6a AS3:
    - T146: Add 99.5% quantile to TailMetrics dataclass (currently only 95%, 99%)
    - T147: Generate comprehensive Q-Q plots (not just error metrics)
    - T148: Add tail error normalization: |model - empirical| / |empirical|
    - T149: Create tail diagnostics report with visual plots and interpretation
    - Update levels parameter default to (0.95, 0.99, 0.995)
    """
    tail_results: List[TailMetrics] = []

    # Empirical quantiles
    emp_q = {
        lvl: np.quantile(returns, 1.0 - lvl)  # for losses in right tail; adjust convention as needed
        for lvl in levels
    }

    for spec in fitted_models:
        try:
            fitter = spec.cls if isinstance(spec.cls, object) else spec.cls()
            # For simplicity, approximate tail quantiles via simulation of one-step returns
            simulated = fitter.sample(n_paths=mc_paths, n_steps=mc_steps).reshape(-1)  # type: ignore[attr-defined]

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
        except Exception as exc:  # noqa: BLE001
            log.warning("tail metric calculation failed", extra={"model": spec.name, "error": str(exc)})

    return tail_results


def run_var_backtests(
    returns_train: np.ndarray,
    returns_test: np.ndarray,
    fitted_models: Sequence[ModelSpec],
    levels: Sequence[float] = (0.95, 0.99),
) -> List[VarBacktestResult]:
    """
    Perform VaR backtests on an out-of-sample segment using each model.

    TODO [T150-T155, AS4]: Implement proper VaR backtesting per spec.md US6a AS4:
    - T150: Implement 70/30 train/test splitter (currently train_fraction parameter)
    - T151: Implement Kupiec unconditional coverage test:
            H₀: breach frequency = expected
            Test statistic: -2×log(L_restricted/L_unrestricted)
            Reject if p < 0.05 (model mis-calibrated)
    - T152: Implement Christoffersen independence test:
            H₀: breaches are serially independent
            Test statistic based on transition matrix
            Reject if p < 0.05 (breaches cluster)
    - T153: Implement one-step-ahead VaR predictor (not static VaR from train!)
            For Student-t/Laplace: use μ_train, σ_train
            For GARCH-t: use dynamic conditional volatility σ_t
    - T154: Implement breach counter with proper test statistics
    - T155: Create backtest results aggregator with pass/fail logic:
            Pass if p ≥ 0.01 on at least one test (Kupiec OR Christoffersen)
            Catastrophic failure if p < 0.01 on BOTH tests
    """
    results: List[VarBacktestResult] = []

    for spec in fitted_models:
        try:
            fitter = spec.cls if isinstance(spec.cls, object) else spec.cls()

            for level in levels:
                # TODO [T153]: Implement proper one-step-ahead forecast logic per model
                # Current implementation uses static VaR - needs dynamic forecast!

                # Placeholder: assume static VaR from train distribution
                simulated = fitter.sample(n_paths=100_000, n_steps=1).reshape(-1)  # type: ignore[attr-defined]
                var_level = np.quantile(simulated, 1.0 - level)

                breaches = returns_test < var_level
                n_breaches = int(breaches.sum())
                n_obs = len(returns_test)
                expected_breaches = (1.0 - level) * n_obs

                # TODO [T151]: Implement Kupiec test with proper likelihood ratio statistic
                kupiec_pvalue = 1.0  # placeholder

                # TODO [T152]: Implement Christoffersen test with transition matrix
                christoffersen_pvalue = 1.0  # placeholder

                # TODO [T155]: Apply proper pass/fail logic based on p-values
                # Pass if p ≥ 0.01 on at least one test (not both < 0.01)
                passed = True  # placeholder

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
        except Exception as exc:  # noqa: BLE001
            log.warning("VaR backtest failed", extra={"model": spec.name, "error": str(exc)})

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

    TODO [T156-T163, AS5]: Enhance simulation realism validation per spec.md US6a AS5:
    - T156: Ensure MC path generator uses deterministic seeding (reproducibility)
    - T157: Annualized volatility calculator already implemented (line 315)
    - T158: Autocorrelation calculator already implemented (lines 319-323)
    - T159: Maximum drawdown calculator already implemented (lines 326-329)
    - T160: Extreme move counter already implemented (lines 332-333)
    - T161: Add historical metrics calculator on 252-day rolling windows:
            Compute same statistics (vol, acf, drawdown, extreme moves) on historical data
            Store as comparison baseline for model validation
    - T162: Add distributional comparator:
            Compare simulated vs historical metric distributions
            Use KS test or chi-square for distribution similarity
            Report systematic biases (model consistently over/under-estimates)
    - T163: Create simulation realism report:
            Generate histograms of simulated vs historical metrics
            Flag models with systematic biases (e.g., always underestimates volatility)
            Include visual diagnostics for interpretation
    """
    sim_results: List[SimulationMetrics] = []

    for spec in fitted_models:
        try:
            fitter = spec.cls if isinstance(spec.cls, object) else spec.cls()
            r = fitter.sample(n_paths=paths, n_steps=steps)  # type: ignore[attr-defined]
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
        except Exception as exc:  # noqa: BLE001
            log.warning("simulation metrics failed", extra={"model": spec.name, "error": str(exc)})

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

    TODO [T164-T168, AS6]: Update scoring to match spec.md US6a AS6:
    - T164: Implement AIC normalization:
            AIC_norm = (AIC - AIC_min) / (AIC_max - AIC_min)
            Currently using raw -AIC (line 386)
    - T165: Update composite scoring formula per spec:
            score = w₁×(-AIC_norm) + w₂×(-tail_error_99) + w₃×(-VaR_backtest_penalty) + w₄×(-vol_cluster_error)
            Note: Use tail_error_99 specifically (not average of 95% and 99%)
            Note: VaR backtest penalty is NOT simple pass rate (see T155 for penalty calculation)
    - T166: Implement constraint validator:
            Excess kurtosis ≥ 1.0 (heavy-tailed requirement from AS2)
            VaR backtest must not catastrophically fail BOTH tests (p < 0.01 on both)
            Models failing constraints get score = -inf
    - T167: Select highest-scoring model that meets all constraints
    - T168: Generate selection report with rationale, evidence, and warnings

    weights: Default per spec.md US6a AS6 should be {"aic": 0.2, "tail": 0.4, "var": 0.3, "cluster": 0.1}
             (NOT 0.2, 0.4, 0.2, 0.2 as currently implemented)
    """
    if weights is None:
        weights = {"aic": 0.2, "tail": 0.4, "var": 0.3, "cluster": 0.1}  # Updated per spec

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
    train_fraction: float = 0.8,  # TODO [T150]: Update default to 0.7 per spec.md US6a AS4
    candidate_models: Optional[Sequence[ModelSpec]] = None,
    s0_override: Optional[float] = None,
    require_heavy_tails: bool = True,
) -> DistributionAuditResult:
    """
    Run the full audit pipeline for a single symbol:

    1. Compute log returns and split into train/test.
    2. Fit Laplace, Student-T, GARCH-T (or provided candidates).
    3. Compute AIC/BIC + heavy-tail diagnostics.
    4. Compute tail metrics (VaR 95/99/99.5).
    5. Run VaR backtests on test window.
    6. Simulate paths and realism metrics.
    7. Score models and select the best.

    TODO [T169-T172, AS7-8]: Add caching and reproducibility:
    - T169: Implement cache manager with 30-day TTL
            Cache key: (symbol, lookback_days, end_date, data_source)
            Store in ~/.cache/quant_scenario_engine/distribution_audits/
    - T170: Create audit result serializer (JSON format)
            Include fitted parameters, validation metrics, selection decision
    - T171: Add --force-refit flag support to bypass cache
    - T172: Implement deterministic seeding throughout:
            Set np.random.seed() and random.seed() at start of audit
            Ensure reproducibility within 1e-6 tolerance

    TODO [T173-T176, AS9-10]: Add integration hooks for US1/US6:
    - T173: Create model loader that auto-loads cached validated models
            Called from US1/US6 Monte Carlo workflows
    - T174: Implement model metadata logger:
            Record model type, fit date, validation scores in run_meta.json
            Per FR-034 provenance requirements
    - T175: Add cache age warning (suggest re-audit when > 30 days old)
    - T176: Implement fallback handler:
            If no audit exists, emit warning and use Laplace default
            Mark run_meta with model_validated: false

    TODO [T177-T179]: CLI command integration:
    - T177: Create audit-distributions CLI command
            Flags: --symbol, --lookback-days, --end-date, --force-refit
    - T178: Wire CLI to this orchestrator function
    - T179: Add audit results formatter (ranked models, scores, recommendation)

    TODO [T180-T182, AS11-12]: Enhanced error handling:
    - T180: Implement insufficient data handler:
            Skip models when sample size < minimum (GARCH-t ≥252)
            Emit warning, continue with other models
    - T181: Create convergence failure handler:
            Log diagnostics (optimizer state, gradient norms, Hessian condition)
            Mark model "FAILED", continue with others
    - T182: Implement audit failure logic:
            Fail entire audit only if ALL three models fail to converge
            Otherwise return partial results with warnings
    """
    prices = price_series.dropna().astype(float)
    if len(prices) < 100:
        raise ValueError(f"Not enough data to audit distributions for {symbol}")

    log_returns = np.log(prices / prices.shift(1)).dropna().values

    # TODO [T150]: Update train_fraction default to 0.7 (70/30 split per spec.md US6a AS4)
    # Currently defaults to 0.8 (80/20 split)
    n = len(log_returns)
    n_train = max(50, int(train_fraction * n))
    r_train = log_returns[:n_train]
    r_test = log_returns[n_train:]

    if candidate_models is None:
        candidate_models = [
            ModelSpec(name="laplace", cls=LaplaceFitter(), config={}),
            ModelSpec(name="student_t", cls=StudentTFitter(), config={}),
            ModelSpec(name="garch_t", cls=GarchTFitter(), config={}),
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
