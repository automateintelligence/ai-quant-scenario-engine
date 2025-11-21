"""Strategy Optimizer - Core optimization engine for discovering optimal option structures (US1/T013, US5/T028)."""

from __future__ import annotations

import logging
import time
from typing import Any

from qse.scorers.intraday_spreads import IntradaySpreadsScorer


class StrategyOptimizer:
    """
    Main optimizer coordinating Stage 0-4 filtering and Top-10/Top-100 ranking.

    Spec References: FR-048 to FR-055, FR-061
    Tasks: T013 (Phase 3), T014-T018 (Phase 4)
    """

    def __init__(self, config: dict[str, Any], data_provider: Any, logger: logging.Logger):
        """
        Initialize optimizer with config and data provider.

        Args:
            config: Merged configuration with regimes, mc, filters, scoring sections
            data_provider: Data source for fetching option chains (Schwab/yfinance with fallback)
            logger: Structured logger for diagnostics
        """
        self.config = config
        self.data_provider = data_provider
        self.log = logger

        # Extract config sections
        self.regimes = config.get("regimes", {})
        self.mc_config = config.get("mc", {})
        self.filter_config = config.get("filters", {})
        self.scoring_config = config.get("scoring", {})

        # Initialize scorer (US5/T028, FR-034, FR-035)
        self.scorer = IntradaySpreadsScorer()

        self.log.info(
            f"StrategyOptimizer initialized: num_paths={self.mc_config.get('num_paths', 5000)} "
            f"max_capital={self.filter_config.get('max_capital', 15000)} "
            f"scorer={self.scorer.__class__.__name__}"
        )

    def optimize(self, ticker: str, regime: str, trade_horizon: int) -> dict[str, Any]:
        """
        Full optimization sweep (Stage 0-4) returning Top-10 ranked strategies.

        Runtime: Up to 1 hour for broad candidate search (FR-061 mode a).

        Args:
            ticker: Stock ticker symbol (e.g., NVDA)
            regime: Regime label from config (e.g., strong-bullish)
            trade_horizon: Trade horizon in days (e.g., 1)

        Returns:
            Dictionary with:
                - top10: List of top 10 ranked CandidateStructure dicts (FR-048)
                - top100: List of top 100 cached candidates (FR-049)
                - diagnostics: Stage counts, rejection breakdowns, runtime (FR-054, FR-055, FR-075)

        Spec: US1 acceptance scenarios 1-4, FR-048 to FR-055
        """
        start_time = time.time()
        self.log.info(f"Starting full optimization: {ticker} regime={regime} horizon={trade_horizon}d")

        try:
            # Phase 4 implementation (T014-T018) will replace this stub with:
            # 1. Fetch option chain via data_provider (FR-004)
            # 2. Stage 0: Expiry selection (T014, FR-006)
            # 3. Stage 1: Strike filtering by moneyness/liquidity (T015, FR-007)
            # 4. Stage 2: Structure generation with width limits (T016, FR-008)
            # 5. Stage 3: Analytic prefilter + hard constraints (T017, FR-009-FR-011)
            # 6. Stage 4: MC scoring on survivors (T018, FR-012)
            # 7. Rank by composite score and return Top-10/Top-100

            # Stub: Return empty result with diagnostic hint
            result = {
                "top10": [],
                "top100": [],
                "diagnostics": {
                    "stage_counts": {
                        "Stage 0 (expiries)": 0,
                        "Stage 1 (strikes)": 0,
                        "Stage 2 (structures)": 0,
                        "Stage 3 (survivors)": 0,
                        "Stage 4 (MC scored)": 0,
                    },
                    "rejections": {
                        "capital_filter": 0,
                        "maxloss_filter": 0,
                        "epnl_filter": 0,
                        "pop_filter": 0,
                    },
                    "hints": (
                        "STUB: Phase 4 (T014-T018) will implement Stage 0-4 filtering. "
                        "Currently returning empty results."
                    ),
                    "runtime_seconds": time.time() - start_time,
                },
            }

            self.log.info(
                f"Optimization complete (STUB): runtime={result['diagnostics']['runtime_seconds']:.1f}s"
            )
            return result

        except Exception as exc:
            self.log.exception(f"Optimization failed for {ticker}: {exc}")
            raise

    def retest_top10(
        self, ticker: str, regime: str, trade_horizon: int, cached_top10: list[dict]
    ) -> dict[str, Any]:
        """
        Retest existing Top-10 list with fresh market data (<30s target, FR-061 mode b).

        Reuses cached candidate structures from previous optimization run.
        Only fetches fresh option chain and reprices + re-scores survivors.

        Args:
            ticker: Stock ticker symbol
            regime: Regime label
            trade_horizon: Trade horizon in days
            cached_top10: Previously cached Top-10 list from optimize() output

        Returns:
            Same structure as optimize() with refreshed metrics

        Spec: FR-061 mode b, Independent Test for US1
        """
        start_time = time.time()
        self.log.info(f"Retesting Top-10: {ticker} regime={regime} horizon={trade_horizon}d")

        try:
            # Phase 4 implementation will:
            # 1. Fetch fresh option chain
            # 2. Reprice cached structures with updated market data
            # 3. Re-run Stage 4 MC scoring only (skip Stage 0-3)
            # 4. Re-rank and return Top-10

            # Stub: Return cached input as-is with updated timestamp
            result = {
                "top10": cached_top10,
                "top100": [],  # Not cached in retest mode
                "diagnostics": {
                    "stage_counts": {
                        "Retest (cached structures)": len(cached_top10),
                    },
                    "hints": "STUB: Retest mode will reprice cached structures with fresh market data",
                    "runtime_seconds": time.time() - start_time,
                },
            }

            self.log.info(
                f"Retest complete (STUB): runtime={result['diagnostics']['runtime_seconds']:.1f}s"
            )
            return result

        except Exception as exc:
            self.log.exception(f"Retest failed for {ticker}: {exc}")
            raise

    def _add_score_decomposition(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Add composite score and decomposition to each candidate (US5/T028, FR-041).

        Mutates candidate dicts to include:
            - composite_score: Overall score in [0, 1] range
            - score_decomposition: Dict with individual component contributions

        Args:
            candidates: List of CandidateStructure dicts with metrics populated

        Returns:
            Same candidates list with scoring added (for chaining)

        Spec: FR-041 (score decomposition in output)
        Tasks: T028
        """
        for candidate in candidates:
            metrics = candidate.get("metrics", {})

            # Compute composite score
            candidate["composite_score"] = self.scorer.score(candidate, metrics, self.config)

            # Add decomposition for diagnostics
            candidate["score_decomposition"] = self.scorer.decompose(
                candidate, metrics, self.config
            )

        return candidates
