"""Integration tests for strategies with feature computation."""

from __future__ import annotations

import numpy as np
import pytest

from quant_scenario_engine.distributions.factory import get_distribution
from quant_scenario_engine.features.technical import compute_all_features, compute_rsi
from quant_scenario_engine.models.options import OptionSpec
from quant_scenario_engine.simulation.compare import run_compare


class TestStrategiesWithFeatures:
    """Test all strategies work with automatic feature computation."""

    @pytest.fixture
    def base_config(self):
        """Base configuration for compare runs."""
        return {
            "s0": 100.0,
            "n_paths": 50,
            "n_steps": 60,
            "seed": 42,
        }

    @pytest.fixture
    def option_spec(self):
        """Standard option specification."""
        return OptionSpec(
            option_type="call",
            strike=100.0,
            maturity_days=30,
            implied_vol=0.2,
            risk_free_rate=0.01,
            contracts=1,
        )

    @pytest.fixture
    def distribution(self):
        """Fitted Laplace distribution."""
        dist = get_distribution("laplace")
        dist.fit(np.random.laplace(0, 0.01, size=500))
        return dist

    def test_stock_basic_no_features_required(self, base_config, option_spec, distribution):
        """Test stock_basic strategy works without features."""
        result = run_compare(
            **base_config,
            distribution=distribution,
            stock_strategy="stock_basic",
            option_strategy="call_basic",
            option_spec=option_spec,
            compute_features=False,  # Disable feature computation
        )
        assert result.metrics is not None
        assert hasattr(result.metrics, "mean_pnl")

    def test_stock_sma_trend_no_features_required(self, base_config, option_spec, distribution):
        """Test stock_sma_trend strategy works without features."""
        result = run_compare(
            **base_config,
            distribution=distribution,
            stock_strategy="stock_sma_trend",
            option_strategy="option_atm_call_momentum",
            option_spec=option_spec,
            compute_features=False,
        )
        assert result.metrics is not None
        assert hasattr(result.metrics, "mean_pnl")

    def test_stock_rsi_reversion_with_auto_features(self, base_config, option_spec, distribution):
        """Test stock_rsi_reversion with automatic feature computation."""
        result = run_compare(
            **base_config,
            distribution=distribution,
            stock_strategy="stock_rsi_reversion",
            option_strategy="call_basic",
            option_spec=option_spec,
            compute_features=True,  # Enable auto feature computation
        )
        assert result.metrics is not None
        assert hasattr(result.metrics, "mean_pnl")

    def test_option_atm_put_rsi_with_auto_features(self, base_config, option_spec, distribution):
        """Test option_atm_put_rsi with automatic feature computation."""
        put_spec = OptionSpec(
            option_type="put",
            strike=100.0,
            maturity_days=21,
            implied_vol=0.2,
            risk_free_rate=0.01,
            contracts=1,
        )
        result = run_compare(
            **base_config,
            distribution=distribution,
            stock_strategy="stock_basic",
            option_strategy="option_atm_put_rsi",
            option_spec=put_spec,
            compute_features=True,
        )
        assert result.metrics is not None
        assert hasattr(result.metrics, "mean_pnl")

    def test_all_strategies_with_precomputed_features(self, base_config, option_spec, distribution):
        """Test strategies with pre-computed features."""
        from quant_scenario_engine.mc.generator import generate_price_paths

        # Generate price paths
        price_paths = generate_price_paths(
            s0=base_config["s0"],
            distribution=distribution,
            n_paths=base_config["n_paths"],
            n_steps=base_config["n_steps"],
            seed=base_config["seed"],
        )

        # Pre-compute features
        features = compute_all_features(price_paths, fillna=True)

        # Verify RSI was computed
        assert "rsi" in features
        assert features["rsi"].shape == price_paths.shape

        # Test with pre-computed features
        result = run_compare(
            **base_config,
            distribution=distribution,
            stock_strategy="stock_rsi_reversion",
            option_strategy="option_atm_put_rsi",
            option_spec=option_spec,
            features=features,  # Pass pre-computed features
            compute_features=False,
        )
        assert result.metrics is not None


class TestTechnicalIndicators:
    """Test technical indicator calculations."""

    @pytest.fixture
    def sample_price_paths(self):
        """Generate sample price paths for testing."""
        np.random.seed(42)
        n_paths, n_steps = 10, 100
        returns = np.random.normal(0.001, 0.02, size=(n_paths, n_steps))
        prices = 100.0 * np.exp(np.cumsum(returns, axis=1))
        return prices

    def test_rsi_calculation(self, sample_price_paths):
        """Test RSI calculation returns valid values."""
        rsi = compute_rsi(sample_price_paths, period=14, fillna=True)

        assert rsi.shape == sample_price_paths.shape
        assert np.all(rsi >= 0)
        assert np.all(rsi <= 100)
        assert not np.any(np.isnan(rsi))  # With fillna=True

    def test_rsi_oversold_overbought(self, sample_price_paths):
        """Test RSI correctly identifies oversold/overbought conditions."""
        rsi = compute_rsi(sample_price_paths, period=14, fillna=True)

        # Create synthetic oversold condition (strong downtrend)
        down_prices = np.copy(sample_price_paths)
        down_prices[:, 20:40] = down_prices[:, 19:20] * np.linspace(1.0, 0.85, 20)
        rsi_down = compute_rsi(down_prices, period=14, fillna=True)

        # Should have some oversold readings (RSI < 30)
        assert np.any(rsi_down[:, 30:40] < 30)

    def test_compute_all_features(self, sample_price_paths):
        """Test compute_all_features returns expected indicators."""
        features = compute_all_features(sample_price_paths, fillna=True)

        expected_features = [
            "rsi",
            "sma_20",
            "sma_50",
            "sma_200",
            "ema_12",
            "ema_26",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "atr",
            "stochastic_k",
            "stochastic_d",
        ]

        for feature in expected_features:
            assert feature in features, f"Missing feature: {feature}"
            assert features[feature].shape == sample_price_paths.shape
