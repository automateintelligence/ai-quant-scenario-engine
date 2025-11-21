"""Shared data-provider interfaces and configuration helpers.

These interfaces document the expectations for providers used by
`qse.data.data_loader.DataLoader` and the CLI fetch workflow. They also
capture configuration defaults for the Schwab→yfinance fallback chain
described in the 009-option-optimizer spec (FR-056/FR-058).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import pandas as pd


@dataclass(slots=True)
class DataSourceConfig:
    """Configuration block for selecting and tuning market data providers."""

    primary: str = "schwab"
    fallback: str | None = "yfinance"
    timeout_seconds: float = 10.0
    max_retries: int = 3


@runtime_checkable
class MarketDataSource(Protocol):
    """Minimal interface implemented by all market data providers."""

    name: str

    def fetch_ohlcv(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        """Return OHLCV bars for a symbol over a date range."""

    def fetch_option_chain(self, symbol: str, expiry: str | None = None) -> pd.DataFrame:
        """Return option chain snapshot with bid/ask/IV/OI/volume columns."""

    def fetch_quotes(self, symbols: Sequence[str], fields: Sequence[str] | None = None) -> dict[str, Any]:
        """Return quote payload for one or more symbols."""

    def fetch_fundamentals(self, symbol: str) -> dict[str, Any]:
        """Return fundamentals block for a single symbol."""

    def fetch_analyst_ratings(self, symbol: str) -> dict[str, Any]:
        """Return analyst ratings/targets for a single symbol."""


__all__ = ["DataSourceConfig", "MarketDataSource"]
