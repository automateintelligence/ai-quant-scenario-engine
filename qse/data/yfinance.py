"""YFinance data adapter with retry and column normalization."""

from __future__ import annotations

import time
from typing import Any, Iterable

import pandas as pd

from qse.exceptions import DataSourceError


class YFinanceDataSource:
    """yfinance-backed provider with retry and option-chain helpers."""

    name = "yfinance"

    def __init__(self, max_retries: int = 3, backoff_seconds: list[int] | None = None) -> None:
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds or [1, 2, 4]
        if len(self.backoff_seconds) < self.max_retries:
            self.backoff_seconds.extend([self.backoff_seconds[-1]] * (self.max_retries - len(self.backoff_seconds)))

    def fetch_ohlcv(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        return self.fetch(symbol=symbol, start=start, end=end, interval=interval)

    def fetch(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                df = self._download(symbol, start, end, interval, progress=False)
                if df is None or df.empty:
                    raise DataSourceError("Empty data returned from yfinance")
                df = self._normalize_columns(df)
                return df
            except Exception as exc:  # pragma: no cover - exercised via tests with retries
                last_exc = exc
                if attempt == self.max_retries - 1:
                    raise DataSourceError(f"Failed to fetch {symbol} after retries") from exc
                time.sleep(self.backoff_seconds[attempt])
        raise DataSourceError("Unknown fetch failure") from last_exc

    def fetch_quotes(self, symbols: Iterable[str], fields: Iterable[str] | None = None) -> dict[str, Any]:
        try:
            import yfinance as yf  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise DataSourceError("yfinance not installed") from exc

        output: dict[str, Any] = {}
        for symbol in symbols:
            info = yf.Ticker(symbol).info  # type: ignore[attr-defined]
            if not info:
                raise DataSourceError(f"Quote lookup failed for {symbol}")
            output[symbol] = info if fields is None else {k: info.get(k) for k in fields}
        return output

    def fetch_fundamentals(self, symbol: str) -> dict[str, Any]:
        quotes = self.fetch_quotes([symbol], fields=None)
        return quotes[symbol]

    def fetch_analyst_ratings(self, symbol: str) -> dict[str, Any]:
        fundamentals = self.fetch_fundamentals(symbol)
        return {k: v for k, v in fundamentals.items() if "rating" in k or "target" in k}

    def fetch_option_chain(self, symbol: str, expiry: str | None = None) -> pd.DataFrame:
        chain = self._option_chain(symbol, expiry)
        if chain is None:
            raise DataSourceError("Empty option chain returned from yfinance")

        frames: list[pd.DataFrame] = []
        for option_type, df in (("call", chain.calls), ("put", chain.puts)):
            if df is None or df.empty:
                continue
            frame = df.copy()
            frame.columns = [col.lower() for col in frame.columns]
            frame["option_type"] = option_type
            frame["expiry"] = expiry or getattr(chain, "expiry", "")
            frames.append(frame)

        if not frames:
            raise DataSourceError("Option chain returned no contracts")
        merged = pd.concat(frames, ignore_index=True)
        rename_map = {"impliedvolatility": "implied_volatility", "openinterest": "open_interest"}
        merged = merged.rename(columns=rename_map)
        return merged

    def _download(self, symbol: str, start: str, end: str, interval: str, progress: bool = False) -> pd.DataFrame:
        try:
            import yfinance as yf  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise DataSourceError("yfinance not installed") from exc

        return yf.download(symbol, start=start, end=end, interval=interval, progress=progress)

    def _option_chain(self, symbol: str, expiry: str | None):
        try:
            import yfinance as yf  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise DataSourceError("yfinance not installed") from exc

        ticker = yf.Ticker(symbol)
        if expiry:
            return ticker.option_chain(date=expiry)
        return ticker.option_chain()

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {col: col.lower() for col in df.columns}
        df = df.rename(columns=rename_map)
        # yfinance provides "adj close"; drop if present
        df = df.drop(columns=[c for c in df.columns if c.startswith("adj")], errors="ignore")
        df.index = pd.to_datetime(df.index)
        return df
