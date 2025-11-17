"""Conditional episode filtering and backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from quant_scenario_engine.schema.episode import CandidateEpisode
from quant_scenario_engine.schema.strategy import StrategyParams
from quant_scenario_engine.simulation.metrics import MetricsReport, compute_metrics
from quant_scenario_engine.simulation.simulator import MarketSimulator
from quant_scenario_engine.strategies.factory import get_strategy
from quant_scenario_engine.features.technical import compute_all_features
from quant_scenario_engine.schema.metrics import metrics_to_json
from quant_scenario_engine.utils.logging import get_logger

log = get_logger(__name__, component="conditional")


def extract_episode_windows(df: pd.DataFrame, episodes: Iterable[CandidateEpisode]) -> List[pd.DataFrame]:
    if "date" in df.columns:
        df = df.set_index(pd.to_datetime(df["date"]))
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must be indexed by datetime or contain a 'date' column")

    windows: List[pd.DataFrame] = []
    for ep in episodes:
        start = pd.to_datetime(ep.t0)
        window = df.loc[start : start + pd.Timedelta(days=ep.horizon - 1)]
        if len(window) < ep.horizon:
            log.warning("episode skipped due to insufficient data", extra={"t0": str(ep.t0)})
            continue
        windows.append(window)
    return windows


def aggregate_episode_metrics(reports: List[MetricsReport]) -> Optional[MetricsReport]:
    if not reports:
        return None
    # Average float fields, keep non-float from first
    float_fields = [
        "mean_pnl",
        "median_pnl",
        "max_drawdown",
        "sharpe",
        "sortino",
        "var",
        "cvar",
        "bankruptcy_rate",
    ]
    base = reports[0]
    agg = {f: 0.0 for f in float_fields}
    for r in reports:
        for f in float_fields:
            agg[f] += getattr(r, f)
    count = float(len(reports))
    for f in float_fields:
        agg[f] /= count
    return MetricsReport(
        mean_pnl=agg["mean_pnl"],
        median_pnl=agg["median_pnl"],
        max_drawdown=agg["max_drawdown"],
        sharpe=agg["sharpe"],
        sortino=agg["sortino"],
        var=agg["var"],
        cvar=agg["cvar"],
        var_method=base.var_method,
        lookback_window=base.lookback_window,
        covariance_estimator=base.covariance_estimator,
        bankruptcy_rate=agg["bankruptcy_rate"],
        early_exercise_events=base.early_exercise_events,
    )


@dataclass
class ConditionalResult:
    unconditional: MetricsReport
    conditional: Optional[MetricsReport]
    episode_count: int
    reports: List[MetricsReport]
    comparison: Optional[dict] = None

    def to_json(self) -> str:
        import json

        payload = {
            "unconditional": json.loads(metrics_to_json(self.unconditional)),
            "conditional": json.loads(metrics_to_json(self.conditional)) if self.conditional else None,
            "episode_count": self.episode_count,
            "comparison": self.comparison,
        }
        return json.dumps(payload, indent=2)


def _run_simulation_on_paths(price_paths: np.ndarray, stock_strategy: str, option_strategy: Optional[str], option_spec, features=None) -> MetricsReport:
    from quant_scenario_engine.simulation.compare import _build_signals

    if option_strategy:
        signals = _build_signals(
            price_paths,
            stock_strategy_name=stock_strategy,
            option_strategy_name=option_strategy,
            option_spec=option_spec,
            stock_params=None,
            option_params=None,
            features=features,
        )
    else:
        stock_impl = get_strategy(stock_strategy, kind="stock")
        stock_signals = stock_impl.generate_signals(
            price_paths,
            features=features or {},
            params=StrategyParams(name=stock_strategy, kind="stock", params={}),
        )
        signals = stock_signals
    sim = MarketSimulator()
    metrics = sim.run(price_paths, signals)
    return metrics


def run_conditional_backtest(
    *,
    df: pd.DataFrame,
    episodes: List[CandidateEpisode],
    stock_strategy: str,
    option_strategy: Optional[str] = None,
    option_spec=None,
) -> ConditionalResult:
    if df.empty:
        raise ValueError("DataFrame is empty")

    price_paths_full = df["close"].to_numpy().reshape(1, -1)
    features_full = compute_all_features(price_paths_full)
    unconditional = _run_simulation_on_paths(price_paths_full, stock_strategy, option_strategy, option_spec, features_full)

    if len(episodes) < 30:
        log.warning("conditional episodes below min threshold", extra={"count": len(episodes)})

    windows = extract_episode_windows(df, episodes)
    reports: List[MetricsReport] = []
    for window in windows:
        price_paths = window["close"].to_numpy().reshape(1, -1)
        feats = compute_all_features(price_paths)
        metrics = _run_simulation_on_paths(price_paths, stock_strategy, option_strategy, option_spec, feats)
        reports.append(metrics)

    aggregate = aggregate_episode_metrics(reports)
    comparison = None
    if aggregate:
        comparison = {
            "episode_count": len(reports),
            "delta_mean_pnl": round(aggregate.mean_pnl - unconditional.mean_pnl, 2),
            "delta_sharpe": round(aggregate.sharpe - unconditional.sharpe, 2),
            "delta_sortino": round(aggregate.sortino - unconditional.sortino, 2),
            "delta_max_drawdown": round(aggregate.max_drawdown - unconditional.max_drawdown, 2),
            "delta_cvar": round(aggregate.cvar - unconditional.cvar, 2),
        }
    return ConditionalResult(
        unconditional=unconditional,
        conditional=aggregate,
        episode_count=len(reports),
        reports=reports,
        comparison=comparison,
    )
