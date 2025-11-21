import json
from pathlib import Path

from qse.simulation.replay import replay_run
from qse.schema.run_meta import RunMeta


def test_replay_with_metrics_and_no_drift(tmp_path: Path):
    run_meta = RunMeta(
        run_id="run_test",
        symbol="TEST",
        config={},
        storage_policy="memory",
        metrics={"mean_pnl": 1.0, "median_pnl": 1.0, "max_drawdown": -0.1, "sharpe": 0.5, "sortino": 0.4, "var": -0.1, "cvar": -0.1, "var_method": "historical", "lookback_window": None, "covariance_estimator": "sample", "bankruptcy_rate": 0.0, "early_exercise_events": 0},
        reproducibility=None,
    )
    path = tmp_path / "run_meta.json"
    path.write_text(run_meta.to_json())

    meta, metrics, paths = replay_run(path)
    assert metrics.mean_pnl == 1.0
    assert meta.is_replay is True
    assert paths is None
