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
        from datetime import datetime

        import numpy as np
        import pandas as pd

        from qse.distributions.factory import get_distribution
        from qse.distributions.regime_loader import load_regime_params
        from qse.optimizers.candidate_filter import (
            Stage0Config,
            Stage1Config,
            filter_strikes,
            select_expiries,
        )
        from qse.optimizers.candidate_generator import CandidateGenerator, GeneratorConfig
        from qse.optimizers.mc_engine import MCConfig, MCEngine
        from qse.optimizers.prefilter import Prefilter, Stage3Config
        from qse.pricing.black_scholes import BlackScholesPricer

        start_time = time.time()
        self.log.info(f"Starting full optimization: {ticker} regime={regime} horizon={trade_horizon}d")

        try:
            # =================================================================
            # Stage 0: Fetch option chain and select expiries (T014, FR-006)
            # =================================================================
            self.log.info("Stage 0: Fetching option chain and selecting expiries")

            # Fetch option chain from data provider
            # For now, create a synthetic chain for testing
            # TODO: Replace with actual data_provider.fetch_option_chain(ticker)
            spot = 500.0  # Synthetic spot price
            chain_df = self._create_synthetic_chain(spot)

            as_of = datetime.now()
            stage0_config = Stage0Config(
                min_dte=self.filter_config.get("min_dte", 7),
                max_dte=self.filter_config.get("max_dte", 45),
                min_expiries=3,
                max_expiries=5,
            )

            selected_expiries = select_expiries(chain_df, as_of, stage0_config)
            self.log.info(f"Stage 0: Selected {len(selected_expiries)} expiries")

            # =================================================================
            # Stage 1: Strike filtering by moneyness + liquidity (T015, FR-007)
            # =================================================================
            self.log.info("Stage 1: Filtering strikes by moneyness and liquidity")

            stage1_config = Stage1Config(
                moneyness_low=self.filter_config.get("moneyness_low", 0.85),
                moneyness_high=self.filter_config.get("moneyness_high", 1.15),
                min_volume=self.filter_config.get("min_volume", 100),
                min_open_interest=self.filter_config.get("min_open_interest", 100),
                max_bid_ask_pct=self.filter_config.get("max_bid_ask_pct", 0.15),
            )

            filtered_chain = filter_strikes(chain_df, spot, selected_expiries, stage1_config)
            num_strikes = len(filtered_chain["strike"].unique())
            self.log.info(f"Stage 1: Retained {num_strikes} strikes after filtering")

            # =================================================================
            # Stage 2: Structure generation (T016, FR-008)
            # =================================================================
            self.log.info("Stage 2: Generating candidate structures")

            generator_config = GeneratorConfig(
                min_width=1,
                max_width=self.filter_config.get("max_width", 3),
            )
            generator = CandidateGenerator(generator_config)
            candidates = generator.generate(filtered_chain, spot)

            self.log.info(f"Stage 2: Generated {len(candidates)} candidate structures")

            # =================================================================
            # Stage 3: Analytic prefilter + hard constraints (T017, FR-009-FR-011)
            # =================================================================
            self.log.info("Stage 3: Applying analytic prefilter with hard constraints")

            stage3_config = Stage3Config(
                max_capital=self.filter_config.get("max_capital", 15000),
                max_loss_pct=self.filter_config.get("max_loss_pct", 0.05),
                min_expected_pnl=self.filter_config.get("min_expected_pnl", 500),
                min_pop_breakeven=self.filter_config.get("min_pop_breakeven", 0.60),
                min_pop_target=self.filter_config.get("min_pop_target", 0.30),
                top_k_per_type=self.filter_config.get("top_k_per_type", 20),
            )

            prefilter = Prefilter(stage3_config)
            survivors = prefilter.evaluate(candidates, spot)

            self.log.info(f"Stage 3: {len(survivors)} survivors advanced to Stage 4")

            # =================================================================
            # Stage 4: Full MC scoring (T018, FR-012)
            # =================================================================
            self.log.info("Stage 4: Running full Monte Carlo scoring")

            # Load regime parameters
            regime_params = load_regime_params(
                regime=regime,
                regimes_cfg=self.regimes,
                mode=self.config.get("regime_mode", "table"),
            )

            # Get distribution and fit with synthetic returns for MVP
            # TODO: Replace with real historical returns from data_provider
            distribution = get_distribution(self.config.get("distribution", "garch_t"))

            # For MVP testing: Generate synthetic returns for fitting
            # In production, use: returns = self.data_provider.fetch_returns(ticker, days=252)
            synthetic_returns = np.random.normal(
                loc=regime_params.mean_daily_return if regime_params else 0.001,
                scale=regime_params.daily_vol if regime_params else 0.02,
                size=252  # 1 year of daily returns
            )
            distribution.fit(synthetic_returns, min_samples=252)

            # Create MC engine
            mc_config = MCConfig(
                num_paths=self.mc_config.get("num_paths", 5000),
                bars_per_day=1,
                seed=self.mc_config.get("seed", 42),
            )

            pricer = BlackScholesPricer()
            mc_engine = MCEngine(distribution, pricer, mc_config)

            # Score survivors
            scored_candidates = mc_engine.score_candidates(
                survivors, spot, trade_horizon, regime_params
            )

            self.log.info(f"Stage 4: Scored {len(scored_candidates)} candidates with MC")

            # =================================================================
            # Ranking and scoring (T028, FR-041)
            # =================================================================
            self.log.info("Ranking candidates by composite score")

            # Add score decomposition
            scored_candidates = self._add_score_decomposition(scored_candidates)

            # Sort by composite score
            ranked = sorted(
                scored_candidates,
                key=lambda c: c.metrics.score if c.metrics else 0.0,
                reverse=True,
            )

            top10 = ranked[:10]
            top100 = ranked[:100]

            # Convert to dictionaries
            top10_dicts = [self._candidate_to_dict(c) for c in top10]
            top100_dicts = [self._candidate_to_dict(c) for c in top100]

            # =================================================================
            # Diagnostics (FR-054, FR-055, FR-075)
            # =================================================================
            runtime = time.time() - start_time

            result = {
                "top10": top10_dicts,
                "top100": top100_dicts,
                "diagnostics": {
                    "stage_counts": {
                        "Stage 0 (expiries)": len(selected_expiries),
                        "Stage 1 (strikes)": num_strikes,
                        "Stage 2 (structures)": len(candidates),
                        "Stage 3 (survivors)": len(survivors),
                        "Stage 4 (MC scored)": len(scored_candidates),
                    },
                    "rejections": {
                        "capital_filter": len(candidates) - len(survivors),
                        "maxloss_filter": 0,  # TODO: Track individual filter rejections
                        "epnl_filter": 0,
                        "pop_filter": 0,
                    },
                    "hints": f"Optimization complete. Top-10 ranked by composite score.",
                    "runtime_seconds": runtime,
                    "regime": regime,
                    "trade_horizon_days": trade_horizon,
                },
            }

            self.log.info(f"Optimization complete: runtime={runtime:.1f}s, top10 count={len(top10)}")
            return result

        except Exception as exc:
            self.log.exception(f"Optimization failed for {ticker}: {exc}")
            raise

    def _create_synthetic_chain(self, spot: float) -> Any:
        """Create synthetic option chain for testing.

        TODO: Replace with actual data provider integration.
        """
        import pandas as pd
        from datetime import datetime, timedelta

        # Generate 5 expiries (7, 14, 21, 28, 35 DTE) to ensure at least 3 in [7,45] window
        expiries = [
            datetime.now() + timedelta(days=7),
            datetime.now() + timedelta(days=14),
            datetime.now() + timedelta(days=21),
            datetime.now() + timedelta(days=28),
            datetime.now() + timedelta(days=35),
        ]

        # Generate strikes around spot (±20% with finer granularity)
        strikes = [spot * (0.80 + 0.03 * i) for i in range(14)]

        rows = []
        for expiry in expiries:
            dte = (expiry - datetime.now()).days
            time_factor = max(1.0, dte / 7.0)  # More time = higher premium

            for strike in strikes:
                for option_type in ["call", "put"]:
                    # Realistic option pricing: intrinsic + time value
                    moneyness = strike / spot

                    if option_type == "call":
                        intrinsic = max(0, spot - strike)
                        time_value = max(2.0, spot * 0.02 * time_factor)  # 2% of spot per week
                        mid = intrinsic + time_value
                    else:
                        intrinsic = max(0, strike - spot)
                        time_value = max(2.0, spot * 0.02 * time_factor)
                        mid = intrinsic + time_value

                    rows.append({
                        "expiry": expiry,
                        "strike": strike,
                        "option_type": option_type,
                        "bid": mid * 0.95,
                        "ask": mid * 1.05,
                        "mid": mid,
                        "volume": 500,
                        "open_interest": 1000,
                    })

        return pd.DataFrame(rows)

    def _candidate_to_dict(self, candidate: Any) -> dict[str, Any]:
        """Convert CandidateStructure to dictionary for JSON serialization."""
        return {
            "structure_type": candidate.structure_type,
            "expiry": str(candidate.expiry),
            "width": candidate.width,
            "net_premium": candidate.net_premium,
            "legs": [
                {
                    "option_type": leg.option_type,
                    "strike": leg.strike,
                    "side": leg.side,
                    "premium": leg.premium,
                    "expiry": str(leg.expiry),
                }
                for leg in candidate.legs
            ],
            "metrics": {
                "expected_pnl": candidate.metrics.expected_pnl,
                "pop_breakeven": candidate.metrics.pop_breakeven,
                "pop_target": candidate.metrics.pop_target,
                "capital": candidate.metrics.capital,
                "max_loss": candidate.metrics.max_loss,
                "score": candidate.metrics.score,
                "mc_paths": candidate.metrics.mc_paths,
            } if candidate.metrics else None,
            "composite_score": candidate.composite_score if hasattr(candidate, "composite_score") else None,
            "score_decomposition": candidate.score_decomposition if hasattr(candidate, "score_decomposition") else None,
        }

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

    def _add_score_decomposition(self, candidates: list[Any]) -> list[Any]:
        """
        Add composite score and decomposition to each candidate (US5/T028, FR-041).

        Adds attributes to CandidateStructure objects:
            - composite_score: Overall score in [0, 1] range
            - score_decomposition: Dict with individual component contributions

        Args:
            candidates: List of CandidateStructure objects with metrics populated

        Returns:
            Same candidates list with scoring added (for chaining)

        Spec: FR-041 (score decomposition in output)
        Tasks: T028
        """
        for candidate in candidates:
            # CandidateStructure has metrics as an attribute, not dict key
            metrics_obj = candidate.metrics if hasattr(candidate, "metrics") and candidate.metrics else None

            if metrics_obj:
                # Convert CandidateMetrics dataclass to dict format expected by IntradaySpreadsScorer
                # MVP: Greeks (Delta, Gamma, Vega, Theta) set to 0 - will compute in future phases
                metrics_dict = {
                    "POP_0": metrics_obj.pop_breakeven,
                    "ROC": metrics_obj.expected_pnl / metrics_obj.capital if metrics_obj.capital > 0 else 0.0,
                    "Theta": 0.0,  # TODO: Compute from option pricing model in Phase 4
                    "Delta": 0.0,  # TODO: Compute from option pricing model in Phase 4
                    "Gamma": 0.0,  # TODO: Compute from option pricing model in Phase 4
                    "Vega": 0.0,   # TODO: Compute from option pricing model in Phase 4
                    "MaxLoss": metrics_obj.max_loss,
                }

                # Convert CandidateStructure to dict format (scorer expects dicts)
                candidate_dict = {
                    "structure_type": candidate.structure_type,
                    "legs": candidate.legs,
                    "expiry": candidate.expiry,
                    "width": candidate.width,
                }

                # Compute composite score
                candidate.composite_score = self.scorer.score(candidate_dict, metrics_dict, self.config)

                # Add decomposition for diagnostics
                candidate.score_decomposition = self.scorer.decompose(
                    candidate_dict, metrics_dict, self.config
                )
            else:
                # No metrics available
                candidate.composite_score = 0.0
                candidate.score_decomposition = {}

        return candidates
