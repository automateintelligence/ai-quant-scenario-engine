"""Factory for data sources with optional fallback chaining."""

from __future__ import annotations

import logging

from qse.config.factories import FactoryBase
from qse.data.schwab import SchwabDataSource
from qse.data.schwab_stub import SchwabDataSourceStub
from qse.data.yfinance import YFinanceDataSource
from qse.exceptions import DataSourceError, DependencyError


log = logging.getLogger(__name__)


class FallbackDataSource:
    """Wrap a primary provider and fallback when DataSourceError is raised."""

    def __init__(self, primary, fallback, *, logger: logging.Logger | None = None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.logger = logger or log
        self.name = f"{getattr(primary, 'name', 'primary')}+{getattr(fallback, 'name', 'fallback')}"

    def fetch_ohlcv(self, *args, **kwargs):
        try:
            return self.primary.fetch_ohlcv(*args, **kwargs)
        except DataSourceError as exc:
            self.logger.warning("Primary data source failed; falling back", extra={"error": str(exc)})
            return self.fallback.fetch_ohlcv(*args, **kwargs)

    def fetch_option_chain(self, *args, **kwargs):
        try:
            return self.primary.fetch_option_chain(*args, **kwargs)
        except DataSourceError as exc:
            self.logger.warning("Primary option chain failed; falling back", extra={"error": str(exc)})
            return self.fallback.fetch_option_chain(*args, **kwargs)

    def fetch_quotes(self, *args, **kwargs):
        try:
            return self.primary.fetch_quotes(*args, **kwargs)
        except DataSourceError as exc:
            self.logger.warning("Primary quotes failed; falling back", extra={"error": str(exc)})
            return self.fallback.fetch_quotes(*args, **kwargs)

    def fetch_fundamentals(self, *args, **kwargs):
        try:
            return self.primary.fetch_fundamentals(*args, **kwargs)
        except DataSourceError as exc:
            self.logger.warning("Primary fundamentals failed; falling back", extra={"error": str(exc)})
            return self.fallback.fetch_fundamentals(*args, **kwargs)

    def fetch_analyst_ratings(self, *args, **kwargs):
        try:
            return self.primary.fetch_analyst_ratings(*args, **kwargs)
        except DataSourceError as exc:
            self.logger.warning("Primary analyst ratings failed; falling back", extra={"error": str(exc)})
            return self.fallback.fetch_analyst_ratings(*args, **kwargs)


def _build_provider(name: str, **kwargs):
    name = name.lower()
    if name == "yfinance":
        return YFinanceDataSource(**{k: v for k, v in kwargs.items() if k in {"max_retries"}})
    if name == "schwab":
        return SchwabDataSource(**{k: v for k, v in kwargs.items() if k in {"access_token", "timeout", "http_get"}})
    if name == "schwab_stub":
        return SchwabDataSourceStub()
    raise DependencyError(f"Unknown data source: {name}")


def get_data_source(name: str, *, fallback: str | None = None, **kwargs):
    primary = _build_provider(name, **kwargs)
    if fallback:
        secondary = _build_provider(fallback, **kwargs)
        return FallbackDataSource(primary, secondary)
    return primary


def data_source_factory(name: str, *, fallback: str | None = None, **kwargs) -> FactoryBase:
    return FactoryBase(name=name, builder=lambda: get_data_source(name, fallback=fallback, **kwargs))


__all__ = ["data_source_factory", "get_data_source", "FallbackDataSource"]
