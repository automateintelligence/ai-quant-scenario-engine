from datetime import datetime, timedelta
import pandas as pd

from quant_scenario_engine.simulation.screen import run_strategy_screen


def _fake_df(symbol: str) -> pd.DataFrame:
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": [i + 1 for i in range(30)],
            "high": [i + 1 for i in range(30)],
            "low": [i + 1 for i in range(30)],
            "close": [i + 1 for i in range(30)],
            "volume": [1_000] * 30,
        }
    ).set_index("date")


def test_screen_mode_b_creates_results():
    universe = {"AAA": _fake_df("AAA"), "BBB": _fake_df("BBB")}
    results = run_strategy_screen(
        universe=universe,
        strategy="stock_sma_trend",
        rank_by="sharpe",
        selector=None,
        min_episodes=10,
        top_n=20,
    )

    assert len(results) >= 1
    assert all(r.metrics_unconditional for r in results)
