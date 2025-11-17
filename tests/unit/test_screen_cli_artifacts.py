import json
from pathlib import Path
import types
import sys

import pandas as pd

# Typer stub to avoid dependency in this unit test
typer_stub = types.SimpleNamespace(Option=lambda *args, **kwargs: None)
class _Exit(Exception):
    def __init__(self, code=1):
        super().__init__(f"exit {code}")
        self.code = code
typer_stub.Exit = _Exit
sys.modules.setdefault("typer", typer_stub)

# yfinance stub for cache imports
fake_yf = types.SimpleNamespace(Ticker=lambda *args, **kwargs: None)
sys.modules.setdefault("yfinance", fake_yf)

from quant_scenario_engine.cli.commands.screen import screen


def test_screen_writes_artifact(tmp_path, monkeypatch):
    # Build small universe CSV
    path = tmp_path / "uni.csv"
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "symbol": ["AAA"] * 5,
        "date": dates,
        "open": [1, 2, 3, 4, 5],
        "high": [1, 2, 3, 4, 5],
        "low": [1, 2, 3, 4, 5],
        "close": [1, 2, 3, 4, 5],
        "volume": [10] * 5,
    })
    df.to_csv(path, index=False)

    out_dir = tmp_path / "runs"

    screen(
        universe=str(path),
        symbols="",
        start=None,
        end=None,
        interval="1d",
        target=tmp_path,
        gap_min=0.01,
        volume_z_min=0.0,
        horizon=2,
        strategy=None,
        rank_by="sharpe",
        conditional_file=None,
        top=None,
        max_workers=1,
        output=out_dir,
    )

    artifact = out_dir / "screen_results_unconditional.json"
    assert artifact.exists()
    data = json.loads(artifact.read_text())
    assert "candidates" in data
