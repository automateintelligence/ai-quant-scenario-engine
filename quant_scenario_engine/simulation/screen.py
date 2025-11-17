"""Universe screening utilities for candidate selection."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Mapping, Optional, List

import pandas as pd

from quant_scenario_engine.interfaces.candidate_selector import CandidateSelector
from quant_scenario_engine.schema.episode import CandidateEpisode
from quant_scenario_engine.utils.logging import get_logger
from quant_scenario_engine.simulation.conditional import run_conditional_backtest
from quant_scenario_engine.strategies.factory import get_strategy
from quant_scenario_engine.schema.strategy import StrategyParams
from quant_scenario_engine.features.technical import compute_all_features
from quant_scenario_engine.simulation.simulator import MarketSimulator
from quant_scenario_engine.models.screen import SymbolScreenResult

log = get_logger(__name__, component="screen")


def _clamp_workers(max_workers: int) -> int:
    return max(1, min(int(max_workers), 6))


def _screen_symbol(symbol: str, df: pd.DataFrame, selector: CandidateSelector) -> list[CandidateEpisode]:
    if df.empty:
        log.warning("empty dataframe for symbol", extra={"symbol": symbol})
        return []
    df = df.copy()
    df["symbol"] = symbol
    episodes = selector.select(df)
    for ep in episodes:
        ep.symbol = symbol
    return episodes


def screen_universe(
    *,
    universe: Mapping[str, pd.DataFrame],
    selector: CandidateSelector,
    max_workers: int = 4,
    top_n: int | None = None,
) -> list[CandidateEpisode]:
    """Apply selector across a universe of symbols with optional parallelism."""

    worker_count = _clamp_workers(max_workers)
    results: list[CandidateEpisode] = []

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_screen_symbol, sym, df, selector): sym for sym, df in universe.items()}
        for fut in as_completed(futures):
            symbol = futures[fut]
            try:
                episodes = fut.result()
                results.extend(episodes)
            except Exception as exc:  # pragma: no cover - defensive logging
                log.error("screening failed for symbol", extra={"symbol": symbol, "error": str(exc)})

    if top_n is not None and results:
        results.sort(key=lambda ep: ep.score or 0.0, reverse=True)
        results = results[: int(top_n)]

    log.info("screening complete", extra={"candidates": len(results)})
    return results


def run_strategy_screen(
    *,
    universe: Mapping[str, pd.DataFrame],
    strategy: str,
    rank_by: str = "sharpe",
    selector: Optional[CandidateSelector] = None,
    min_episodes: int = 10,
    top_n: Optional[int] = None,
) -> List[SymbolScreenResult]:
    results: List[SymbolScreenResult] = []
    rank_by = rank_by.lower()

    for symbol, df in universe.items():
        price_paths = df["close"].to_numpy().reshape(1, -1)
        features = compute_all_features(price_paths)
        stock_impl = get_strategy(strategy, kind="stock")
        signals = stock_impl.generate_signals(
            price_paths,
            features=features,
            params=StrategyParams(name=strategy, kind="stock", params={}),
        )
        metrics_uncond = MarketSimulator().run(price_paths, signals)

        cond_metrics = None
        episode_count = None
        low_confidence = False
    if selector:
        episodes = selector.select(df.reset_index())
        episode_count = len(episodes)
        if episode_count < min_episodes:
            low_confidence = True
        cond_result = run_conditional_backtest(
            df=df.reset_index(),
            episodes=episodes,
            stock_strategy=strategy,
        )
        cond_metrics = cond_result.conditional
        comparison = cond_result.comparison
    
    rank_metric = getattr(cond_metrics or metrics_uncond, rank_by, None)
    results.append(
        SymbolScreenResult(
            symbol=symbol,
            metrics_unconditional=metrics_uncond,
            metrics_conditional=cond_metrics,
            episode_count=episode_count,
            rank_metric=rank_metric,
            low_confidence=low_confidence,
            comparison=comparison if selector else None,
        )
    )

    results.sort(key=lambda r: r.rank_metric or float("-inf"), reverse=True)
    if top_n is not None:
        results = results[: int(top_n)]
    return results
