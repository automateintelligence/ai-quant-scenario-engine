import pandas as pd

from quant_scenario_engine.simulation.screen import run_strategy_screen
from quant_scenario_engine.selectors.gap_volume import GapVolumeSelector


def _build_universe():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df1 = pd.DataFrame({
        "date": dates,
        "open": [10 + i for i in range(30)],
        "high": [10 + i for i in range(30)],
        "low": [10 + i for i in range(30)],
        "close": [10 + i for i in range(30)],
        "volume": [1_000] * 30,
    }).set_index("date")
    df2 = pd.DataFrame({
        "date": dates,
        "open": [20 + i for i in range(30)],
        "high": [20 + i for i in range(30)],
        "low": [20 + i for i in range(30)],
        "close": [20 + i for i in range(30)],
        "volume": [2_000] * 30,
    }).set_index("date")
    return {"AAA": df1, "BBB": df2}


def test_run_strategy_screen_unconditional():
    universe = _build_universe()
    results = run_strategy_screen(universe=universe, strategy="stock_basic", rank_by="sharpe", selector=None, top_n=1)
    assert len(results) == 1
    assert results[0].metrics_unconditional is not None


def test_run_strategy_screen_conditional_low_confidence():
    universe = _build_universe()
    selector = GapVolumeSelector(gap_min=0.5, volume_z_min=5, horizon=5)  # ensures few/no episodes
    results = run_strategy_screen(universe=universe, strategy="stock_basic", rank_by="sharpe", selector=selector, min_episodes=10)
    assert len(results) >= 1
    assert any(r.low_confidence for r in results)
