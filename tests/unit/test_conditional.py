import pandas as pd

from quant_scenario_engine.schema.episode import CandidateEpisode
from quant_scenario_engine.simulation.conditional import run_conditional_backtest


def test_run_conditional_backtest_with_episode():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "open": range(20),
        "high": range(20),
        "low": range(20),
        "close": [100 + i for i in range(20)],
        "volume": [1_000] * 20,
    })

    episode = CandidateEpisode(symbol="TEST", t0=dates[0], horizon=20, state_features={})
    result = run_conditional_backtest(
        df=df,
        episodes=[episode],
        stock_strategy="stock_basic",
    )

    assert result.conditional is not None
    assert result.episode_count == 1
    formatted = result.conditional.to_formatted_dict()
    assert "mean_pnl$" in formatted
    assert isinstance(formatted["mean_pnl$"], float)
    assert result.comparison is not None
    assert "delta_mean_pnl" in result.comparison
