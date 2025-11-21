import pandas as pd
import pytest

from qse.data.factory import FallbackDataSource
from qse.data.schwab import SchwabDataSource
from qse.data.yfinance import YFinanceDataSource
from qse.exceptions import DataSourceError


class FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_option_chain_normalization():
    payload = {
        "callExpDateMap": {
            "2025-01-17:0": {
                "100.0": [
                    {
                        "strikePrice": 100.0,
                        "bid": 1.2,
                        "ask": 1.4,
                        "volatility": 0.2,
                        "openInterest": 10,
                        "totalVolume": 5,
                    }
                ]
            }
        },
        "putExpDateMap": {},
    }

    def fake_get(url, headers=None, params=None, timeout=None):
        return FakeResponse(200, payload)

    ds = SchwabDataSource(access_token="token", http_get=fake_get)
    df = ds.fetch_option_chain("AAPL")
    assert set(["expiry", "strike", "option_type", "bid", "ask", "implied_volatility", "open_interest", "volume"]).issubset(df.columns)
    assert df.iloc[0]["strike"] == 100.0
    assert df.iloc[0]["option_type"] == "call"


def test_http_error_normalization():
    error_payload = {"errors": [{"detail": "Missing header"}]}

    def fake_get(url, headers=None, params=None, timeout=None):
        return FakeResponse(401, error_payload)

    ds = SchwabDataSource(access_token="token", http_get=fake_get)
    with pytest.raises(DataSourceError) as exc:
        ds.fetch_quotes(["AAPL"])
    assert "401" in str(exc.value)


def test_fallback_datasource_warns_and_recovers():
    class Primary:
        name = "primary"

        def fetch_ohlcv(self, *args, **kwargs):
            raise DataSourceError("boom")

    class Secondary(YFinanceDataSource):
        def fetch_ohlcv(self, *args, **kwargs):  # type: ignore[override]
            return pd.DataFrame({"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]}, index=pd.date_range("2024-01-01", periods=1))

    wrapper = FallbackDataSource(Primary(), Secondary())
    df = wrapper.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-02")
    assert not df.empty
