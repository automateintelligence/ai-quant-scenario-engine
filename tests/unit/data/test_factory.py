import pytest

from qse.data.factory import FallbackDataSource, data_source_factory, get_data_source
from qse.data.schwab import SchwabDataSource
from qse.data.schwab_stub import SchwabDataSourceStub
from qse.data.yfinance import YFinanceDataSource
from qse.exceptions import DataSourceError, DependencyError


def test_get_data_source_returns_providers():
    assert isinstance(get_data_source("yfinance"), YFinanceDataSource)
    assert isinstance(get_data_source("schwab_stub"), SchwabDataSourceStub)
    assert isinstance(get_data_source("schwab", access_token="token", http_get=lambda *_, **__: None), SchwabDataSource)
    with pytest.raises(DependencyError):
        get_data_source("unknown")


def test_schwab_stub_fetch_raises():
    stub = SchwabDataSourceStub()
    with pytest.raises(DataSourceError):
        stub.fetch("AAPL", "2023-01-01", "2023-01-02")


def test_factory_creates_instances():
    factory = data_source_factory("yfinance")
    inst = factory.create()
    assert isinstance(inst, YFinanceDataSource)


def test_factory_wraps_fallback():
    factory = data_source_factory("schwab", fallback="yfinance")
    inst = factory.create()
    assert isinstance(inst, FallbackDataSource)
