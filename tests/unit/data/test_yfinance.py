import pandas as pd
import pytest

from qse.data.data_loader import DataLoader
from qse.data.yfinance import YFinanceDataSource
from qse.exceptions import DataSourceError


def test_yfinance_fetch_normalizes_columns(monkeypatch):
    calls = {"count": 0}

    def fake_download(symbol, start, end, interval, progress):
        calls["count"] += 1
        return pd.DataFrame({
            "Open": [1, 2],
            "High": [2, 3],
            "Low": [0, 1],
            "Close": [1, 2],
            "Adj Close": [1, 2],
            "Volume": [10, 10],
        }, index=pd.date_range(start, periods=2, freq="D"))

    monkeypatch.setattr(YFinanceDataSource, "_download", staticmethod(fake_download))
    yf = YFinanceDataSource(max_retries=1)
    df = yf.fetch("AAPL", "2023-01-01", "2023-01-03", interval="1d")
    assert set(df.columns) == {"open", "high", "low", "close", "volume"}
    assert calls["count"] == 1


def test_yfinance_retries_then_fails(monkeypatch):
    def failing_download(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(YFinanceDataSource, "_download", staticmethod(failing_download))
    yf = YFinanceDataSource(max_retries=2, backoff_seconds=[0, 0])
    with pytest.raises(DataSourceError):
        yf.fetch("AAPL", "2023-01-01", "2023-01-03")


def test_dataloader_integration_with_yfinance(monkeypatch, tmp_path):
    def fake_download(symbol, start, end, interval, progress):
        return pd.DataFrame(
            {
                "Open": [1, 2, 3],
                "High": [2, 3, 4],
                "Low": [0, 1, 2],
                "Close": [1, 2, 3],
                "Volume": [10, 10, 10],
            },
            index=pd.date_range(start, periods=3, freq="D"),
        )

    monkeypatch.setattr(YFinanceDataSource, "_download", staticmethod(fake_download))
    yf = YFinanceDataSource(max_retries=1)
    loader = DataLoader(
        base_dir=tmp_path / "data" / "historical", data_source=yf, storage_format="pickle"
    )
    df = loader.load_ohlcv("AAPL", "2023-01-01", "2023-01-03", interval="1d")
    assert len(df) == 3
    assert "close" in df.columns


def test_option_chain_normalization(monkeypatch):
    class FakeChain:
        def __init__(self):
            self.calls = pd.DataFrame(
                {
                    "strike": [100.0],
                    "bid": [1.0],
                    "ask": [1.1],
                    "impliedVolatility": [0.2],
                    "openInterest": [10],
                    "volume": [5],
                }
            )
            self.puts = pd.DataFrame()

    def fake_option_chain(symbol, expiry):
        return FakeChain()

    monkeypatch.setattr(YFinanceDataSource, "_option_chain", staticmethod(fake_option_chain))
    yf = YFinanceDataSource(max_retries=1)
    df = yf.fetch_option_chain("AAPL", expiry="2024-01-19")
    assert set(["bid", "ask", "implied_volatility", "open_interest", "volume", "strike", "expiry"]).issubset(df.columns)
