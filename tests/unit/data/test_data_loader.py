import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from qse.data.data_loader import DataLoader
from qse.exceptions import DataSourceError


def _make_df(start: str, periods: int) -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=periods, freq="D")
    return pd.DataFrame(
        {
            "open": range(1, periods + 1),
            "high": range(2, periods + 2),
            "low": range(0, periods),
            "close": range(1, periods + 1),
            "volume": [100] * periods,
        },
        index=dates,
    )


class StubLoader(DataLoader):
    def __init__(self, base_dir: Path, responses: dict, **kwargs):
        super().__init__(base_dir, storage_format="pickle", **kwargs)
        self.responses = responses
        self.calls = []

    def _fetch_from_source(self, symbol: str, start: str, end: str, interval: str):
        self.calls.append((start, end))
        key = (start, end)
        if key not in self.responses:
            raise DataSourceError("missing response")
        return self.responses[key]


def test_rejects_wrong_category_path(tmp_path: Path):
    with pytest.raises(DataSourceError):
        DataLoader(tmp_path / "data", category="historical")


def test_version_selection_prefers_latest(tmp_path: Path):
    base_dir = tmp_path / "data" / "historical"
    loader = StubLoader(base_dir, responses={})
    part_v1 = loader._resolve_partition("AAPL", "interval=1d", "_v1")
    part_v1.mkdir(parents=True, exist_ok=True)
    df_v1 = _make_df("2023-01-01", 2)
    loader._write_cache(part_v1 / "data.pkl", part_v1 / "data.meta.json", df_v1, "AAPL", "2023-01-01", "2023-01-02")

    part_v2 = loader._resolve_partition("AAPL", "interval=1d", "_v2")
    part_v2.mkdir(parents=True, exist_ok=True)
    df_v2 = _make_df("2023-01-01", 3)
    loader._write_cache(part_v2 / "data.pkl", part_v2 / "data.meta.json", df_v2, "AAPL", "2023-01-01", "2023-01-03")

    # No responses needed because cache is hit
    out = loader.load_ohlcv("AAPL", "2023-01-01", "2023-01-03", interval="1d")
    assert len(out) == 3
    assert out.iloc[-1]["close"] == 3  # from v2


def test_corporate_action_triggers_full_refresh(tmp_path: Path):
    base_dir = tmp_path / "data" / "historical"
    responses = {}
    base_df = _make_df("2023-01-01", 3)
    responses[("2023-01-01", "2023-01-03")] = base_df
    overlap_df = _make_df("2023-01-03", 1)
    overlap_df.iloc[0, overlap_df.columns.get_loc("close")] = 200  #  >1% divergence
    responses[("2023-01-03", "2023-01-04")] = overlap_df
    refreshed = _make_df("2023-01-01", 5)
    responses[("2023-01-01", "2023-01-05")] = refreshed

    loader = StubLoader(base_dir, responses=responses)
    loader.load_ohlcv("AAPL", "2023-01-01", "2023-01-03", interval="1d")
    out = loader.load_ohlcv("AAPL", "2023-01-01", "2023-01-05", interval="1d")

    # Calls: initial fetch, overlap check, full refresh
    assert loader.calls == [
        ("2023-01-01", "2023-01-03"),
        ("2023-01-03", "2023-01-04"),
        ("2023-01-01", "2023-01-05"),
    ]
    assert len(out) == 5


def test_incremental_append_without_corporate_action(tmp_path: Path):
    base_dir = tmp_path / "data" / "historical"
    responses = {}
    base_df = _make_df("2023-01-01", 3)
    responses[("2023-01-01", "2023-01-03")] = base_df
    overlap_df = _make_df("2023-01-03", 1)
    overlap_df.iloc[0, overlap_df.columns.get_loc("close")] = base_df["close"].iloc[-1]
    responses[("2023-01-03", "2023-01-04")] = overlap_df
    incremental = _make_df("2023-01-03", 2)
    responses[("2023-01-03", "2023-01-05")] = incremental

    loader = StubLoader(base_dir, responses=responses)
    loader.load_ohlcv("AAPL", "2023-01-01", "2023-01-03", interval="1d")
    out = loader.load_ohlcv("AAPL", "2023-01-01", "2023-01-05", interval="1d")

    # incremental fetch should have appended two new rows (total 4 unique dates because one overlaps)
    assert len(out) == 4
    assert loader.calls[-1] == ("2023-01-03", "2023-01-05")


def test_allow_stale_cache_on_failure(tmp_path: Path):
    base_dir = tmp_path / "data" / "historical"
    base_dir.mkdir(parents=True, exist_ok=True)
    loader = StubLoader(base_dir, responses={("2023-01-01", "2023-01-02"): _make_df("2023-01-01", 2)})
    # seed cache
    loader.load_ohlcv("AAPL", "2023-01-01", "2023-01-02", interval="1d")

    cache_meta = loader._resolve_partition("AAPL", "interval=1d", None) / "data.meta.json"
    meta = json.loads(cache_meta.read_text())
    meta["fetched_at"] = (datetime.utcnow() - timedelta(days=10)).isoformat()
    cache_meta.write_text(json.dumps(meta))

    loader.responses = {}  # force fetch failure
    out = loader.load_ohlcv(
        "AAPL", "2023-01-01", "2023-01-02", interval="1d", allow_stale_cache=True
    )
    assert len(out) == 2


def test_option_chain_caching(tmp_path: Path):
    class ChainSource:
        name = "chain"

        def fetch_option_chain(self, symbol: str, expiry: str | None = None):
            return pd.DataFrame(
                {
                    "expiry": ["2024-01-19"],
                    "strike": [100.0],
                    "option_type": ["call"],
                    "bid": [1.0],
                    "ask": [1.2],
                    "implied_volatility": [0.2],
                    "open_interest": [10],
                    "volume": [5],
                }
            )

    base_dir = tmp_path / "data" / "option_chains"
    loader = DataLoader(base_dir, category="option_chains", data_source=ChainSource())
    df = loader.load_option_chain("AAPL", as_of="2024-01-02", expiry="2024-01-19")
    assert not df.empty
    # cached read should avoid source call
    df_cached = loader.load_option_chain("AAPL", as_of="2024-01-02", expiry="2024-01-19")
    assert len(df_cached) == len(df)
