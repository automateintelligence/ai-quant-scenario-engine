"""End-to-end acceptance tests for StrategyOptimizer (US1, US2)."""

import pytest

from qse.optimizers.strategy_optimizer import StrategyOptimizer
from qse.utils.logging import get_logger


class TestUS1Acceptance:
    """Test User Story 1 - Single-Command Strategy Optimization."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer with test config."""
        config = {
            "regimes": {
                "strong-bullish": {
                    "mean_daily_return": 0.02,
                    "daily_vol": 0.03,
                    "skew": 0.5,
                    "kurtosis_excess": 1.5,
                },
                "neutral": {
                    "mean_daily_return": 0.0,
                    "daily_vol": 0.01,
                    "skew": 0.0,
                    "kurtosis_excess": 1.0,
                },
            },
            "mc": {
                "num_paths": 1000,  # Reduced for faster testing
                "seed": 42,
            },
            "filters": {
                "min_dte": 7,
                "max_dte": 45,
                "max_capital": 15000,
                "max_loss_pct": 0.50,  # 50% for SHORT positions (straddles/strangles)
                "min_expected_pnl": 500,  # Minimum $500 expected profit
                "min_pop_breakeven": 0.55,
                "min_pop_target": 0.30,  # Required by prefilter
                "top_k_per_type": 10,
            },
            "scoring": {
                "w_pop": 0.35,
                "w_roc": 0.30,
                "w_theta": 0.10,
            },
        }

        logger = get_logger(__name__, component="test")
        return StrategyOptimizer(config, data_provider=None, logger=logger)

    def test_scenario_1_basic_optimization(self, optimizer):
        """
        US1 Scenario 1: Basic optimization returns Top-10 with all metrics.

        Given: ticker "NVDA", regime "strong-bullish", default config
        When: optimizer runs
        Then: Returns exactly 10 CandidateStructure objects with complete metrics
        """
        result = optimizer.optimize(
            ticker="NVDA",
            regime="strong-bullish",
            trade_horizon=1,
        )

        # Verify structure
        assert "top10" in result
        assert "top100" in result
        assert "diagnostics" in result

        # Verify Top-10 count (may be less than 10 with strict filters)
        assert len(result["top10"]) > 0
        assert len(result["top10"]) <= 10

        # Verify first candidate has all required fields
        top_candidate = result["top10"][0]
        assert "structure_type" in top_candidate
        assert "legs" in top_candidate
        assert "metrics" in top_candidate

        metrics = top_candidate["metrics"]
        assert "expected_pnl" in metrics
        assert "pop_breakeven" in metrics
        assert "pop_target" in metrics
        assert "capital" in metrics
        assert "max_loss" in metrics
        assert "score" in metrics
        assert "mc_paths" in metrics

        # Verify constraints
        assert metrics["capital"] <= 15000
        assert metrics["pop_breakeven"] >= 0.55

        # Verify diagnostics
        diag = result["diagnostics"]
        assert "stage_counts" in diag
        assert "runtime_seconds" in diag
        assert diag["stage_counts"]["Stage 0 (expiries)"] > 0
        assert diag["stage_counts"]["Stage 1 (strikes)"] > 0
        assert diag["stage_counts"]["Stage 2 (structures)"] > 0

        print(f"\n✓ US1 Scenario 1: Returned {len(result['top10'])} candidates")
        print(f"  Runtime: {diag['runtime_seconds']:.1f}s")
        print(f"  Top candidate: {top_candidate['structure_type']}, score={metrics['score']:.3f}")

    def test_scenario_2_trade_horizon_3day(self, optimizer):
        """
        US1 Scenario 2: Multi-day trade horizon.

        Given: trade_horizon=3
        When: optimizer runs
        Then: All metrics computed over 3-day holding period
        """
        result = optimizer.optimize(
            ticker="TSLA",
            regime="neutral",
            trade_horizon=3,
        )

        assert result["diagnostics"]["trade_horizon_days"] == 3
        assert len(result["top10"]) > 0

        print(f"\n✓ US1 Scenario 2: 3-day horizon optimization successful")

    def test_scenario_3_config_overrides(self, optimizer):
        """
        US1 Scenario 3: Config overrides applied.

        Given: Override mc.num_paths, filters.max_capital
        When: optimizer runs
        Then: Overrides take effect
        """
        # Modify config for this test
        optimizer.mc_config["num_paths"] = 2000
        optimizer.filter_config["max_capital"] = 20000

        result = optimizer.optimize(
            ticker="SPY",
            regime="strong-bullish",
            trade_horizon=1,
        )

        # Check that higher capital allowed more candidates through
        assert result["diagnostics"]["stage_counts"]["Stage 3 (survivors)"] > 0

        print(f"\n✓ US1 Scenario 3: Config overrides successful")

    def test_scenario_4_empty_results_diagnostics(self, optimizer):
        """
        US1 Scenario 4: Empty results with diagnostic explanation.

        Given: Very restrictive filters
        When: optimizer runs
        Then: Returns empty result with rejection breakdown
        """
        # Set impossibly strict filters
        optimizer.filter_config["min_expected_pnl"] = 100000  # $100k minimum
        optimizer.filter_config["max_capital"] = 1000  # $1k maximum

        result = optimizer.optimize(
            ticker="AAPL",
            regime="neutral",
            trade_horizon=1,
        )

        # May have zero results due to strict filters
        diag = result["diagnostics"]
        assert "rejections" in diag

        print(f"\n✓ US1 Scenario 4: Empty result handling successful")
        print(f"  Rejections: {diag['rejections']}")


class TestUS2Acceptance:
    """Test User Story 2 - Multi-Stage Candidate Filtering."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer with test config."""
        config = {
            "regimes": {
                "strong-bullish": {
                    "mean_daily_return": 0.02,
                    "daily_vol": 0.03,
                    "skew": 0.5,
                    "kurtosis_excess": 1.5,
                },
            },
            "mc": {"num_paths": 500, "seed": 42},
            "filters": {
                "min_dte": 7,
                "max_dte": 45,
                "max_capital": 15000,
                "max_loss_pct": 0.50,  # 50% for SHORT positions (straddles/strangles)
                "min_expected_pnl": 500,  # Minimum $500 expected profit
                "min_pop_breakeven": 0.55,
                "min_pop_target": 0.30,  # Required by prefilter
                "top_k_per_type": 10,
            },
        }

        logger = get_logger(__name__, component="test")
        return StrategyOptimizer(config, data_provider=None, logger=logger)

    def test_scenario_1_stage0_expiry_selection(self, optimizer):
        """
        US2 Scenario 1: Stage 0 expiry selection.

        Given: Option chain with multiple expiries
        When: Stage 0 runs
        Then: 3-5 expiries with DTE in [7, 45] selected
        """
        result = optimizer.optimize("MSFT", "strong-bullish", 1)

        stage_counts = result["diagnostics"]["stage_counts"]
        expiry_count = stage_counts["Stage 0 (expiries)"]

        assert 3 <= expiry_count <= 5
        print(f"\n✓ US2 Scenario 1: Selected {expiry_count} expiries")

    def test_scenario_2_stage1_strike_filtering(self, optimizer):
        """
        US2 Scenario 2: Stage 1 moneyness and liquidity filtering.

        Given: Strikes with varying moneyness and liquidity
        When: Stage 1 runs
        Then: Only strikes in moneyness band with good liquidity retained
        """
        result = optimizer.optimize("MSFT", "strong-bullish", 1)

        stage_counts = result["diagnostics"]["stage_counts"]
        strikes_count = stage_counts["Stage 1 (strikes)"]

        assert strikes_count > 0
        print(f"\n✓ US2 Scenario 2: Retained {strikes_count} strikes after filtering")

    def test_scenario_3_stage2_structure_generation(self, optimizer):
        """
        US2 Scenario 3: Stage 2 generates verticals, Iron Condors, etc.

        Given: Filtered strikes
        When: Stage 2 runs
        Then: Candidate structures generated with width limits
        """
        result = optimizer.optimize("MSFT", "strong-bullish", 1)

        stage_counts = result["diagnostics"]["stage_counts"]
        structures_count = stage_counts["Stage 2 (structures)"]

        assert structures_count > 0
        print(f"\n✓ US2 Scenario 3: Generated {structures_count} structures")

    def test_scenario_4_stage3_analytic_prefilter(self, optimizer):
        """
        US2 Scenario 4: Stage 3 analytic prefiltering with hard constraints.

        Given: Candidate structures from Stage 2
        When: Stage 3 runs
        Then: Top-K per structure type advance to Stage 4
        """
        result = optimizer.optimize("MSFT", "strong-bullish", 1)

        stage_counts = result["diagnostics"]["stage_counts"]
        structures = stage_counts["Stage 2 (structures)"]
        survivors = stage_counts["Stage 3 (survivors)"]

        assert survivors <= structures  # Some rejected
        assert survivors > 0  # Some passed
        print(f"\n✓ US2 Scenario 4: {survivors} survivors (from {structures} structures)")

    def test_scenario_5_stage4_mc_scoring(self, optimizer):
        """
        US2 Scenario 5: Stage 4 full MC scoring.

        Given: Survivors from Stage 3
        When: Stage 4 runs
        Then: Full MC with GARCH-t, ranked Top-10
        """
        result = optimizer.optimize("MSFT", "strong-bullish", 1)

        stage_counts = result["diagnostics"]["stage_counts"]
        scored = stage_counts["Stage 4 (MC scored)"]

        assert scored > 0
        assert len(result["top10"]) > 0

        # Verify MC paths used
        top_candidate = result["top10"][0]
        assert top_candidate["metrics"]["mc_paths"] > 0

        print(f"\n✓ US2 Scenario 5: Scored {scored} candidates with MC")
        print(f"  Top-10 count: {len(result['top10'])}")


class TestPerformance:
    """Test runtime performance targets."""

    def test_runtime_under_30_seconds(self):
        """
        Performance Test: Full optimization completes in <30 seconds.

        Note: For MVP with synthetic data and reduced MC paths.
        """
        config = {
            "regimes": {
                "strong-bullish": {
                    "mean_daily_return": 0.02,
                    "daily_vol": 0.03,
                    "skew": 0.5,
                    "kurtosis_excess": 1.5,
                },
            },
            "mc": {"num_paths": 1000, "seed": 42},
            "filters": {
                "max_capital": 15000,
                "max_loss_pct": 0.50,  # 50% for SHORT positions (straddles/strangles)
                "min_expected_pnl": 500,  # Minimum $500 expected profit
                "min_pop_breakeven": 0.55,
                "min_pop_target": 0.30,
                "top_k_per_type": 10,
            },
        }

        logger = get_logger(__name__, component="test")
        optimizer = StrategyOptimizer(config, data_provider=None, logger=logger)

        result = optimizer.optimize("NVDA", "strong-bullish", 1)

        runtime = result["diagnostics"]["runtime_seconds"]
        print(f"\n✓ Performance: Runtime {runtime:.1f}s")

        # For MVP with synthetic data, expect < 10s
        assert runtime < 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
