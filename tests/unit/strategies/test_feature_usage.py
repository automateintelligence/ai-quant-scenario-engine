import numpy as np

from qse.models.options import OptionSpec
from qse.schema.strategy import StrategyParams
from qse.strategies.option_call import OptionCallStrategy
from qse.strategies.stock_basic import StockBasicStrategy


def test_stock_basic_uses_feature_sma_and_tracks_usage():
    price_paths = np.array([[1.0, 2.0, 3.0, 4.0]])
    sma_short = np.array([[1.0, 1.5, 2.0, 2.5]])
    sma_long = np.array([[2.0, 2.0, 2.0, 2.0]])
    features = {"sma_5": sma_short, "sma_20": sma_long}

    strategy = StockBasicStrategy(short_window=5, long_window=20)
    signals = strategy.generate_signals(price_paths, features, StrategyParams(name="s", kind="stock", params={}))

    assert signals.features_used == ["sma_5", "sma_20"]
    # short_ma > long_ma on last bar -> long
    assert signals.signals_stock[0, -1] == 1


def test_option_call_applies_rsi_filter_when_present():
    price_paths = np.array([[1.0, 2.0, 3.0]])
    features = {"rsi": np.array([[40.0, 40.0, 40.0]])}
    option_spec = OptionSpec(option_type="call", strike=100, maturity_days=10, implied_vol=0.2, risk_free_rate=0.01, contracts=1)
    strategy = OptionCallStrategy(option_spec)
    params = StrategyParams(name="opt", kind="option", params={"min_rsi": 50})

    signals = strategy.generate_signals(price_paths, features, params)

    assert signals.features_used == ["rsi"]
    # RSI filter blocks entries
    assert signals.signals_option.sum() == 0
