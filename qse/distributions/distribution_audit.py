"""
distribution_audit.py

Tools for fitting multiple return distribution models (Laplace, Student-T, GARCH-T),
evaluating their suitability for financial returns, and selecting a "best" model
for Monte Carlo simulation based on tail behavior, VaR backtests, and volatility
clustering.

ADAPTATION STATUS:
This sophisticated stub must be adapted to align with spec.md US6a (lines 131-271)
and tasks.md Phase 7a (T137-T182). The core architecture is sound, but specific
modifications are needed to meet all acceptance scenarios (AS1-AS12).

KEY ADAPTATIONS NEEDED:
X 1. Preprocessing pipeline (AS1, T138-T142): Add stationarity checks, outlier detection
X 2. Goodness-of-fit enhancements (AS2, T143-T145): Add excess kurtosis validation (≥1.0)
3. Tail diagnostics (AS3, T146-T149): Add 99.5% quantile, comprehensive Q-Q plots
4. VaR backtesting (AS4, T150-T155): Implement Kupiec/Christoffersen statistical tests
5. Model selection scoring (AS6, T164-T168): Update weights to (0.2, 0.4, 0.3, 0.1)
X 6. Caching layer (AS7, T169-T171): Add 30-day TTL cache with proper key structure
X 7. Reproducibility (AS8, T172): Add deterministic seeding throughout
X 8. Integration hooks (AS9-10, T173-T176): Auto-load validated models for US1/US6
9. CLI command (T177-T179): Wire to audit-distributions command
X 10. Enhanced error handling (AS11-12, T180-T182): Add convergence diagnostics

See tasks.md Phase 7a for detailed implementation tasks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from qse.distributions.backtesting.breach_counter import count_breaches
from qse.distributions.backtesting.christoffersen_test import christoffersen_pvalue
from qse.distributions.backtesting.data_split import train_test_split
from qse.distributions.backtesting.kupiec_test import kupiec_pvalue
from qse.distributions.backtesting.var_predictor import predict_var_from_samples
from qse.distributions.cache.cache_manager import get_cache_path, is_fresh, load_cache, save_cache, TTL_DAYS
from qse.distributions.cache.serializer import deserialize_payload, serialize_payload
from qse.distributions.diagnostics.tail_report import build_tail_report
from qse.distributions.diagnostics.tail_metrics import tail_error
from qse.distributions.errors import (
    handle_insufficient_data,
    has_minimum_samples,
    record_convergence_failure,
)
from qse.distributions.fitters.garch_t_fitter import GarchTFitter
from qse.distributions.fitters.laplace_fitter import LaplaceFitter
from qse.distributions.fitters.student_t_fitter import StudentTFitter
from qse.distributions.integration.cache_checker import warn_if_stale
from qse.distributions.metrics.information_criteria import aic as calc_aic, bic as calc_bic
from qse.distributions.models import FitResult
from qse.distributions.selection.selection_report import build_selection_report
from qse.distributions.selection.scorer import composite_score
from qse.distributions.validation.clustering_calc import autocorr_squared_returns
from qse.distributions.validation.drawdown_calc import max_drawdown
from qse.distributions.validation.extreme_moves import extreme_move_frequencies
from qse.distributions.validation.historical_metrics import compute_historical_metrics
from qse.distributions.validation.realism_report import build_realism_report
from qse.distributions.validation.volatility_calc import annualized_volatility
from qse.distributions.validation.stationarity import MIN_SAMPLES
from qse.exceptions import DistributionFitError
from qse.interfaces.distribution import ReturnDistribution
from qse.utils.logging import get_logger

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
    """Tail-focused diagnostics for empirical vs model-implied returns."""

    model_name: str
    var_emp_95: float
    var_emp_99: float
    var_emp_995: float
    var_model_95: float
    var_model_99: float
    var_model_995: float
    tail_error_95: float
    tail_error_99: float
    tail_error_995: float


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
    best_fit: Optional[FitResult] = None
    tail_reports: Dict[str, dict] = field(default_factory=dict)
    realism_reports: Dict[str, dict] = field(default_factory=dict)
    selection_report: Dict[str, object] = field(default_factory=dict)


def _derive_seed(base_seed: int | None, model_name: str, stage: str) -> Optional[int]:
    """Derive a deterministic seed per model/stage from the base seed."""

    if base_seed is None:
        return None
    mix = sum(ord(ch) for ch in f"{model_name}:{stage}")
    return int((base_seed * 31 + mix) % (2**31 - 1))


def _rehydrate_model_spec(raw: ModelSpec | dict | None) -> Optional[ModelSpec]:
    if raw is None:
        return None
    if isinstance(raw, ModelSpec):
        return raw
    return ModelSpec(name=raw.get("name"), cls=raw.get("cls"), config=raw.get("config", {}))


def _rehydrate_fit_results(raw_items: Sequence[FitResult | dict]) -> List[FitResult]:
    fits: List[FitResult] = []
    for item in raw_items:
        if isinstance(item, FitResult):
            fits.append(item)
        else:
            fits.append(FitResult(**item))
    return fits


def _rehydrate_scores(raw_items: Sequence[ModelScore | dict]) -> List[ModelScore]:
    scores: List[ModelScore] = []
    for item in raw_items:
        if isinstance(item, ModelScore):
            scores.append(item)
        else:
            scores.append(ModelScore(**item))
    return scores


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------


def fit_candidate_models(
    returns: np.ndarray,
    candidate_models: Sequence[ModelSpec],
    *,
    symbol: str | None = None,
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

    sample_count = len(returns)

    for spec in candidate_models:
        required = MIN_SAMPLES.get(spec.name, 60)
        if not has_minimum_samples(spec.name, sample_count, required):
            results.append(
                handle_insufficient_data(
                    spec.name,
                    sample_count,
                    min_required=required,
                    symbol=symbol,
                )
            )
            continue

        try:
            fitter = spec.cls if hasattr(spec.cls, "fit") else spec.cls()
            fitted: FitResult = fitter.fit(returns)  # type: ignore[call-arg]
            if not fitted.fit_success or not fitted.converged:
                log.warning(
                    "model convergence incomplete",
                    extra={
                        "model": spec.name,
                        "message": fitted.fit_message or "; ".join(fitted.warnings) or "non-converged",
                    },
                )
            results.append(fitted)
        except DistributionFitError as exc:
            results.append(
                record_convergence_failure(
                    spec.name,
                    error=exc,
                    n_samples=sample_count,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                record_convergence_failure(
                    spec.name,
                    error=exc,
                    n_samples=sample_count,
                    stage="unexpected",
                )
            )

    return results


def compute_tail_metrics(
    returns: np.ndarray,
    fitted_models: Sequence[ModelSpec],
    fit_results: Sequence[FitResult],
    levels: Sequence[float] = (0.95, 0.99, 0.995),
    mc_paths: int = 50_000,
    mc_steps: int = 1,
    seed: int | None = None,
) -> Tuple[List[TailMetrics], Dict[str, np.ndarray]]:
    """
    Compare empirical tail quantiles to model-implied tail quantiles using
    Monte Carlo or closed-form quantiles where available.

    TODO [T146-T149, AS3 - scheduled]:
    - Extend ``levels`` default to ``(0.95, 0.99, 0.995)`` so the audit always samples VaR_99.5.
    - Persist the extra quantile pairs to ``TailMetrics`` and expose them in ``tail_report`` / CLI formatter.
    - Swap the naive Monte Carlo approximation with deterministic quantile functions exposed by each fitter
      (Laplace/Student-T analytic quantiles, GARCH simulation) so reruns remain deterministic.
    - Emit normalized tail errors for every configured level and surface them inside ``selection_report``
      and CLI output (per US6a AS3 acceptance scenarios).
    """
    tail_results: List[TailMetrics] = []
    samples_by_model: Dict[str, np.ndarray] = {}
    fit_success = {fr.model_name: fr.fit_success for fr in fit_results}

    # Empirical quantiles
    emp_q = {lvl: np.quantile(returns, 1.0 - lvl) for lvl in levels}

    for spec in fitted_models:
        if not fit_success.get(spec.name, False):
            continue

        try:
            fitter = spec.cls if hasattr(spec.cls, "sample") else spec.cls()
            draw_seed = _derive_seed(seed, spec.name, "tail")
            simulated = fitter.sample(n_paths=mc_paths, n_steps=mc_steps, seed=draw_seed).reshape(-1)  # type: ignore[attr-defined]
            samples_by_model[spec.name] = simulated
            errors = tail_error(returns, simulated, levels=levels)

            def _metric(level: float, key: str) -> Tuple[float, float, float]:
                entry = errors.get(key)
                if not entry:
                    return emp_q[level], float(np.nan), float("inf")
                return entry["empirical"], entry["model"], entry["relative_error"]

            emp_95, model_95, err_95 = _metric(0.95, "var_95.0")
            emp_99, model_99, err_99 = _metric(0.99, "var_99.0")
            emp_995, model_995, err_995 = _metric(0.995, "var_99.5")

            tail_results.append(
                TailMetrics(
                    model_name=spec.name,
                    var_emp_95=emp_95,
                    var_emp_99=emp_99,
                    var_emp_995=emp_995,
                    var_model_95=model_95,
                    var_model_99=model_99,
                    var_model_995=model_995,
                    tail_error_95=err_95,
                    tail_error_99=err_99,
                    tail_error_995=err_995,
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("tail metric calculation failed", extra={"model": spec.name, "error": str(exc)})

    return tail_results, samples_by_model


def run_var_backtests(
    returns_test: np.ndarray,
    fitted_models: Sequence[ModelSpec],
    fit_results: Sequence[FitResult],
    samples_by_model: Dict[str, np.ndarray],
    levels: Sequence[float] = (0.95, 0.99),
) -> List[VarBacktestResult]:
    """
    Perform VaR backtests on an out-of-sample segment using each model.

    TODO [T150-T155, AS4 - scheduled]:
    - Call ``qse.var_predictor`` to generate rolling one-step VaR forecasts instead of
      sampling static returns from ``fitter.sample``.
    - Replace placeholder p-values with the actual Kupiec (LR_uc) and Christoffersen (LR_ind) tests
      defined under ``distributions.backtesting`` modules and persist their statistics in ``VarBacktestResult``.
    - Compute breach counts via ``breach_counter.py`` and aggregate everything within
      ``backtest_report.py`` so CLI output can summarize pass/fail decisions per AS4.
    - Enforce FR requirement: audit must mark catastrophic failure when **both** tests return ``p < 0.01``.
    """
    results: List[VarBacktestResult] = []
    fit_success = {fr.model_name: fr.fit_success for fr in fit_results}

    for spec in fitted_models:
        if not fit_success.get(spec.name, False):
            continue

        sample = samples_by_model.get(spec.name)
        if sample is None or sample.size == 0:
            continue

        try:
            for level in levels:
                var_level = predict_var_from_samples(sample, level)
                n_breaches, breach_sequence = count_breaches(returns_test, var_level)
                n_obs = len(returns_test)
                expected_breaches = (1.0 - level) * n_obs
                kupiec = kupiec_pvalue(n_obs, n_breaches, level)
                christ = christoffersen_pvalue(breach_sequence)
                passed = kupiec >= 0.05 and christ >= 0.05

                results.append(
                    VarBacktestResult(
                        model_name=spec.name,
                        level=level,
                        n_obs=n_obs,
                        n_breaches=n_breaches,
                        expected_breaches=expected_breaches,
                        kupiec_pvalue=kupiec,
                        christoffersen_pvalue=christ,
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
    fit_results: Sequence[FitResult],
    hist_metrics: dict,
    paths: int = 10_000,
    steps: int = 252,
    seed: int | None = None,
) -> Tuple[List[SimulationMetrics], Dict[str, dict]]:
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
    realism_reports: Dict[str, dict] = {}
    fit_success = {fr.model_name: fr.fit_success for fr in fit_results}

    hist_vol = hist_metrics.get("annualized_vol", 0.0)
    hist_acf = hist_metrics.get("acf_sq_lag1", 0.0)

    for spec in fitted_models:
        if not fit_success.get(spec.name, False):
            continue
        try:
            fitter = spec.cls if hasattr(spec.cls, "sample") else spec.cls()
            draw_seed = _derive_seed(seed, spec.name, "sim")
            returns = fitter.sample(n_paths=paths, n_steps=steps, seed=draw_seed)  # type: ignore[attr-defined]
            price_paths = np.exp(np.log(max(s0, 1e-9)) + np.cumsum(returns, axis=1))
            price_paths = np.clip(price_paths, 1e-8, None)

            returns_flat = returns.reshape(-1)
            sim_vol = annualized_volatility(returns_flat)
            sim_acf = autocorr_squared_returns(returns_flat, lag=1)
            drawdowns = [max_drawdown(path) for path in price_paths]
            mean_max_dd = float(np.mean(drawdowns))
            extremes = extreme_move_frequencies(returns_flat)

            sim_detail = {
                "annualized_vol": sim_vol,
                "acf_sq_lag1": sim_acf,
                "max_drawdown": mean_max_dd,
                "extremes": extremes,
            }
            realism_reports[spec.name] = build_realism_report(sim_detail, hist_metrics)

            sim_results.append(
                SimulationMetrics(
                    model_name=spec.name,
                    mean_annualized_vol=sim_vol,
                    acf_sq_returns_lag1=sim_acf,
                    mean_max_drawdown=mean_max_dd,
                    freq_gt_3pct_move=extremes.get("gt_3pct", 0.0),
                    freq_gt_5pct_move=extremes.get("gt_5pct", 0.0),
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("simulation metrics failed", extra={"model": spec.name, "error": str(exc)})

    return sim_results, realism_reports


def score_models(
    fit_results: Sequence[FitResult],
    tail_metrics: Sequence[TailMetrics],
    var_backtests: Sequence[VarBacktestResult],
    sim_metrics: Sequence[SimulationMetrics],
    realism_reports: Dict[str, dict],
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

    fit_by_name = {fr.model_name: fr for fr in fit_results}
    tail_by_name = {tm.model_name: tm for tm in tail_metrics}
    sim_by_name = {sm.model_name: sm for sm in sim_metrics}

    aic_values = [fr.aic for fr in fit_results if np.isfinite(fr.aic)]
    if aic_values:
        aic_min, aic_max = min(aic_values), max(aic_values)
    else:
        aic_min = aic_max = 0.0
    aic_denom = (aic_max - aic_min) or 1.0

    var_by_name: Dict[str, List[VarBacktestResult]] = {}
    for vb in var_backtests:
        var_by_name.setdefault(vb.model_name, []).append(vb)

    def _var_penalty(entries: List[VarBacktestResult]) -> float:
        if not entries:
            return 0.0
        penalties = []
        for vb in entries:
            if vb.kupiec_pvalue < 0.01 and vb.christoffersen_pvalue < 0.01:
                penalties.append(1.0)
                continue
            kupiec_shortfall = max(0.0, 0.05 - vb.kupiec_pvalue) / 0.05
            christ_shortfall = max(0.0, 0.05 - vb.christoffersen_pvalue) / 0.05
            penalties.append((kupiec_shortfall + christ_shortfall) / 2.0)
        return float(np.mean(penalties))

    scores: List[ModelScore] = []

    for name, fr in fit_by_name.items():
        aic_norm = float(((fr.aic - aic_min) / aic_denom) if np.isfinite(fr.aic) else 1.0)
        tm = tail_by_name.get(name)
        tail_err = tm.tail_error_99 if tm else 1.0
        var_penalty = _var_penalty(var_by_name.get(name, []))
        deltas = realism_reports.get(name, {}).get("deltas", {})
        cluster_error = abs(deltas.get("annualized_vol", 0.0)) + abs(deltas.get("acf_sq_lag1", 0.0))

        total = composite_score(aic_norm, tail_err, var_penalty, cluster_error, weights=(
            weights["aic"],
            weights["tail"],
            weights["var"],
            weights["cluster"],
        ))

        components = {
            "aic_norm": aic_norm,
            "tail_error_99": tail_err,
            "var_penalty": var_penalty,
            "vol_cluster_error": cluster_error,
        }

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
    constraints: Optional[Dict[str, Dict[str, bool]]] = None,
) -> Optional[ModelSpec]:
    """
    Select the best model by score, optionally requiring it to be heavy-tailed.
    Falls back to best available fit if heavy-tailed models are unavailable.
    """
    score_by_name = {s.model_name: s for s in scores}
    fit_by_name = {fr.model_name: fr for fr in fit_results if fr.fit_success}

    def _best(names: List[str]) -> Optional[str]:
        if not names:
            return None
        return max(names, key=lambda n: score_by_name[n].total_score)

    eligible_names: List[str] = []
    if require_heavy_tails:
        eligible_names = [
            name
            for name, fr in fit_by_name.items()
            if fr.heavy_tailed is True and name in score_by_name
        ]
    else:
        eligible_names = [name for name in score_by_name]

    if constraints:
        eligible_names = [
            name
            for name in eligible_names
            if constraints.get(name, {}).get("var_pass", True)
        ]

    best_name = _best(eligible_names)
    if best_name is None:
        # Fallback: any fit_success model
        log.warning(
            "no heavy-tailed models available; falling back to best available fit",
            extra={"candidates": list(fit_by_name.keys())},
        )
        fallback_names = [name for name in fit_by_name.keys() if name in score_by_name]
        if constraints:
            fallback_names = [
                name for name in fallback_names if constraints.get(name, {}).get("var_pass", True)
            ]
        best_name = _best(fallback_names)

    if best_name is None:
        return None

    for spec in candidate_models:
        if spec.name == best_name:
            return spec
    return None


def select_best_fit(
    fit_results: Sequence[FitResult],
    best_model: Optional[ModelSpec],
) -> Optional[FitResult]:
    """Return the FitResult for the chosen model, if available."""
    if best_model is None:
        return None
    for fr in fit_results:
        if fr.model_name == best_model.name:
            return fr
    return None


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------


def audit_distributions_for_symbol(
    symbol: str,
    price_series: pd.Series,
    train_fraction: float = 0.7,
    candidate_models: Optional[Sequence[ModelSpec]] = None,
    s0_override: Optional[float] = None,
    require_heavy_tails: bool = True,
    plot_fit: bool = False,
    plot_output_path: Optional[str] = None,
    cache_dir: str | None = None,
    lookback_days: int | None = None,
    end_date: str | None = None,
    data_source: str | None = None,
    force_refit: bool = False,
    seed: int | None = None,
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
    8. Optionally generate fit diagnostic plots.

    Parameters
    ----------
    symbol : str
        Asset symbol for labeling
    price_series : pd.Series
        Historical price series
    train_fraction : float, default=0.8
        Fraction of data for training (remainder for test)
    candidate_models : Optional[Sequence[ModelSpec]]
        Models to fit; defaults to Laplace, Student-t, GARCH-t
    s0_override : Optional[float]
        Override initial price for simulations
    require_heavy_tails : bool, default=True
        Require selected model to have heavy tails (excess kurtosis >= 1.0)
    plot_fit : bool, default=False
        Generate diagnostic plots showing fit quality
    plot_output_path : Optional[str]
        Custom path for plot output; defaults to output/distribution_fits/{symbol}_fit_diagnostics.png

    Caching/reproducibility (T169-T172, AS7-8) implemented: cache entries live at
    ``output/distribution_audits`` with a 30-day TTL, ``--force-refit`` bypasses
    cached results, and deterministic seeding keeps repeated runs identical.

    Integration hooks (T173-T176, AS9-10) implemented: the integration helpers
    auto-load cached models for US1/US6, emit warnings when entries are stale,
    and fall back to Laplace while marking metadata when no audit exists.

    NOTE [T177-T179]: CLI integration is implemented via
    ``qse.cli.commands.audit_distributions`` and the
    `audit_formatter`, so this orchestration entry point is already wired to the CLI command.

    Enhanced error handling (T180-T182, AS11-12) implemented: models with
    insufficient samples or convergence failures are logged and marked FAILED,
    and the audit only aborts when every candidate fails to converge.
    """
    prices = price_series.dropna().astype(float)
    if len(prices) < 100:
        raise ValueError(f"Not enough data to audit distributions for {symbol}")

    log_returns = np.log(prices / prices.shift(1)).dropna().values

    if seed is not None:
        import random
        np.random.seed(seed)
        random.seed(seed)

    cache_base = Path(cache_dir) if cache_dir else Path("output") / "distribution_audits"
    cache_path = get_cache_path(cache_base, symbol, lookback_days, end_date, data_source)
    cached = load_cache(cache_path)
    if cached and not force_refit:
        if is_fresh(cache_path, ttl_days=TTL_DAYS):
            log.info("Loaded audit from cache", extra={"path": str(cache_path)})
            data = deserialize_payload(json.dumps(cached))
            data["models"] = [m for m in (_rehydrate_model_spec(m) for m in data.get("models", [])) if m]
            data["fit_results"] = _rehydrate_fit_results(data.get("fit_results", []))
            if data.get("scores"):
                data["scores"] = _rehydrate_scores(data.get("scores", []))
            if data.get("best_model"):
                data["best_model"] = _rehydrate_model_spec(data["best_model"])  # type: ignore[assignment]
            if data.get("best_fit") and not isinstance(data["best_fit"], FitResult):
                data["best_fit"] = FitResult(**data["best_fit"])
            return DistributionAuditResult(**data)  # type: ignore[arg-type]
        warn_if_stale(cache_path, ttl_days=TTL_DAYS)

    r_train, r_test = train_test_split(log_returns, train_fraction=train_fraction, min_train=50)

    if candidate_models is None:
        candidate_models = [
            ModelSpec(name="laplace", cls=LaplaceFitter(), config={}),
            ModelSpec(name="student_t", cls=StudentTFitter(), config={}),
            ModelSpec(name="garch_t", cls=GarchTFitter(), config={}),
        ]

    fit_results = fit_candidate_models(r_train, candidate_models, symbol=symbol)

    if not any(fr.fit_success for fr in fit_results):
        status_summary = ", ".join(
            f"{fr.model_name}={fr.fit_message or fr.error or 'failed'}" for fr in fit_results
        )
        raise DistributionFitError(
            "Distribution audit failed: no models converged | " + status_summary
        )

    tail_metrics, samples_by_model = compute_tail_metrics(
        r_train,
        candidate_models,
        fit_results,
        seed=seed,
    )

    var_backtests = run_var_backtests(r_test, candidate_models, fit_results, samples_by_model)

    hist_metrics = compute_historical_metrics(r_train)

    s0 = float(s0_override if s0_override is not None else prices.iloc[-1])
    sim_metrics, realism_reports = simulate_paths_and_metrics(
        symbol=symbol,
        s0=s0,
        fitted_models=candidate_models,
        fit_results=fit_results,
        hist_metrics=hist_metrics,
        paths=10_000,
        steps=252,
        seed=seed,
    )

    scores = score_models(
        fit_results=fit_results,
        tail_metrics=tail_metrics,
        var_backtests=var_backtests,
        sim_metrics=sim_metrics,
        realism_reports=realism_reports,
    )

    var_entries: Dict[str, List[VarBacktestResult]] = {}
    for vb in var_backtests:
        var_entries.setdefault(vb.model_name, []).append(vb)

    constraints = {
        fr.model_name: {
            "heavy_tailed": bool(fr.heavy_tailed),
            "var_pass": not any(
                r.kupiec_pvalue < 0.01 and r.christoffersen_pvalue < 0.01
                for r in var_entries.get(fr.model_name, [])
            ),
        }
        for fr in fit_results
    }

    best_model = select_best_model(
        candidate_models=candidate_models,
        scores=scores,
        require_heavy_tails=require_heavy_tails,
        fit_results=fit_results,
        constraints=constraints,
    )
    best_fit = select_best_fit(fit_results, best_model)

    # Tail diagnostics (QQ/tail errors/kurtosis) per model
    tail_reports: Dict[str, dict] = {}
    for name, samples in samples_by_model.items():
        try:
            tail_reports[name] = build_tail_report(r_train, samples)
        except Exception as exc:
            log.warning("tail diagnostics failed", extra={"model": name, "error": str(exc)})

    # Selection report using simple constraints (heavy-tail & VaR pass)
    selection_report = build_selection_report(scores, best_model.name if best_model else None)

    # Generate fit diagnostic plots if requested
    if plot_fit:
        from qse.distributions.plotting import plot_distribution_fits

        output_path = None
        if plot_output_path:
            output_path = Path(plot_output_path)
        else:
            # Default to output/distribution_fits/{symbol}_fit_diagnostics.png
            output_path = Path("output") / "distribution_fits" / f"{symbol}_fit_diagnostics.png"

        try:
            plot_distribution_fits(
                returns=log_returns,
                fit_results=fit_results,
                candidate_models=candidate_models,
                symbol=symbol,
                output_path=output_path,
                show_plot=False,  # Don't block in automated workflows
            )
            log.info(f"Generated fit diagnostic plots for {symbol}")
        except Exception as exc:
            log.warning(f"Failed to generate fit plots for {symbol}: {exc}")

    result = DistributionAuditResult(
        symbol=symbol,
        models=list(candidate_models),
        fit_results=fit_results,
        tail_metrics=tail_metrics,
        var_backtests=var_backtests,
        simulation_metrics=sim_metrics,
        scores=scores,
        best_model=best_model,
        best_fit=best_fit,
        tail_reports=tail_reports,
        realism_reports=realism_reports,
        selection_report=selection_report,
    )

    # Save to cache if enabled
    try:
        payload = serialize_payload(result)
        save_cache(cache_path, json.loads(payload))
    except Exception as exc:
        log.warning("Failed to cache audit result", extra={"error": str(exc)})

    return result


def fit_best_distribution_for_returns(
    returns: np.ndarray,
    require_heavy_tails: bool = True,
) -> tuple[str, FitResult]:
    """
    Lightweight helper to select the best-fitting distribution on a log-return array.

    Returns (model_name, FitResult) and NEVER silently falls back: failures raise DistributionFitError.
    """
    candidate_models = [
        ModelSpec(name="laplace", cls=LaplaceFitter(), config={}),
        ModelSpec(name="student_t", cls=StudentTFitter(), config={}),
        ModelSpec(name="garch_t", cls=GarchTFitter(), config={}),
    ]
    fit_results = fit_candidate_models(returns, candidate_models)
    scores = score_models(
        fit_results=fit_results,
        tail_metrics=[],
        var_backtests=[],
        sim_metrics=[],
        realism_reports={},
    )
    best_model = select_best_model(candidate_models, scores, require_heavy_tails, fit_results)
    best_fit = select_best_fit(fit_results, best_model)

    if best_model is None or best_fit is None or not best_fit.fit_success:
        raise DistributionFitError("No suitable distribution fit available for returns")

    return best_model.name, best_fit


def instantiate_distribution(model_name: str) -> ReturnDistribution:
    """
    Map model name to concrete ReturnDistribution implementation for US1/US6 integration.
    """
    if model_name == "laplace":
        from qse.distributions.laplace import LaplaceDistribution
        return LaplaceDistribution()
    if model_name in {"student_t", "student-t"}:
        from qse.distributions.student_t import StudentTDistribution
        return StudentTDistribution()
    raise DistributionFitError(f"Unsupported distribution model: {model_name}")
