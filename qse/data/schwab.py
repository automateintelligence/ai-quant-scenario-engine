"""Schwab Trader API client with normalized responses and fallbacks."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from qse.exceptions import DataSourceError

log = logging.getLogger(__name__)


_HttpGetter = Callable[[str, Any], Any]


@dataclass(slots=True)
class _SchwabResponse:
    status_code: int
    payload: Mapping[str, Any]

    def json(self) -> Mapping[str, Any]:
        return self.payload


class SchwabDataSource:
    """REST adapter for Schwab market data endpoints.

    The client delegates all network calls to an injectable HTTP getter to
    simplify testing. Responses are normalized into pandas DataFrames for
    OHLCV and option chain workflows, with DataSourceError used for all
    recoverable failures (FR-004/FR-005).
    """

    name = "schwab"

    def __init__(
        self,
        *,
        access_token: str | None = None,
        base_url: str = "https://api.schwabapi.com/marketdata/v1",
        timeout: float = 10.0,
        http_get: _HttpGetter | None = None,
    ) -> None:
        self.access_token = access_token or os.getenv("SCHWAB_ACCESS_TOKEN")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http_get = http_get

    def fetch_ohlcv(
        self, symbol: str, start: str, end: str, interval: str = "1d"
    ) -> pd.DataFrame:
        params = {"symbol": symbol, "periodType": "custom", "startDate": start, "endDate": end}
        if interval in {"1m", "5m", "10m", "15m", "30m"}:
            params["frequencyType"] = "minute"
            params["frequency"] = int(interval.rstrip("m"))
        else:
            params["frequencyType"] = "daily"
            params["frequency"] = 1

        payload = self._request("pricehistory", params=params)
        candles = payload.get("candles")
        if not candles:
            raise DataSourceError("Schwab returned no candles for requested range")
        frame = pd.DataFrame(candles)
        if "datetime" in frame.columns:
            frame["datetime"] = pd.to_datetime(frame["datetime"], unit="ms")
            frame = frame.set_index("datetime").sort_index()
        rename_map = {col: col.lower() for col in frame.columns}
        frame = frame.rename(columns=rename_map)
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise DataSourceError(f"Missing OHLCV fields in Schwab response: {missing}")
        return frame[required]

    def fetch_quotes(self, symbols: Iterable[str], fields: Iterable[str] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"symbols": ",".join(symbols)}
        if fields:
            params["fields"] = ",".join(fields)
        payload = self._request("quotes", params=params)
        if not isinstance(payload, Mapping):
            raise DataSourceError("Unexpected Schwab quote payload")
        return dict(payload)

    def fetch_fundamentals(self, symbol: str) -> dict[str, Any]:
        quote_payload = self.fetch_quotes([symbol], fields=["fundamental", "reference"])
        details = quote_payload.get(symbol, {}) if isinstance(quote_payload, Mapping) else {}
        fundamentals = details.get("fundamental") if isinstance(details, Mapping) else None
        if not fundamentals:
            raise DataSourceError("Fundamental block missing from Schwab response")
        return dict(fundamentals)

    def fetch_analyst_ratings(self, symbol: str) -> dict[str, Any]:
        fundamentals = self.fetch_fundamentals(symbol)
        analyst_keys = {k: fundamentals[k] for k in fundamentals.keys() if "rating" in k.lower() or "target" in k.lower()}
        if not analyst_keys:
            raise DataSourceError("Analyst ratings unavailable in Schwab payload")
        return analyst_keys

    def fetch_option_chain(self, symbol: str, expiry: str | None = None) -> pd.DataFrame:
        params: dict[str, Any] = {"symbol": symbol}
        if expiry:
            params["fromDate"] = expiry
            params["toDate"] = expiry
        payload = self._request("chains", params=params)
        return self._normalize_option_chain(payload)

    def _normalize_option_chain(self, payload: Mapping[str, Any]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for key, option_type in (("callExpDateMap", "call"), ("putExpDateMap", "put")):
            exp_map = payload.get(key, {}) if isinstance(payload, Mapping) else {}
            for expiry, strikes in exp_map.items():
                expiry_label = expiry.split(":")[0]
                for strike, contracts in strikes.items():
                    for contract in contracts:
                        rows.append(
                            {
                                "expiry": expiry_label,
                                "strike": float(contract.get("strikePrice", strike)),
                                "option_type": option_type,
                                "bid": contract.get("bid"),
                                "ask": contract.get("ask"),
                                "implied_volatility": contract.get("volatility")
                                or contract.get("impliedVolatility"),
                                "open_interest": contract.get("openInterest"),
                                "volume": contract.get("totalVolume"),
                                "delta": contract.get("delta"),
                                "gamma": contract.get("gamma"),
                                "theta": contract.get("theta"),
                                "vega": contract.get("vega"),
                            }
                        )

        if not rows:
            raise DataSourceError("Schwab returned empty option chain")
        frame = pd.DataFrame(rows)
        return frame

    def _request(self, path: str, params: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        if not self.access_token:
            raise DataSourceError("Schwab access token required for API calls")
        response = self._perform_request(path, params or {})
        try:
            payload = response.json()
        except Exception as exc:  # pragma: no cover - defensive
            raise DataSourceError("Unable to parse Schwab response as JSON") from exc

        status_code = getattr(response, "status_code", 500)
        if status_code >= 400:
            message = self._extract_error(payload, status_code)
            raise DataSourceError(message)
        if not isinstance(payload, Mapping):
            raise DataSourceError("Schwab response payload must be a mapping")
        return payload

    def _perform_request(self, path: str, params: Mapping[str, Any]) -> _SchwabResponse:
        client = self._http_get
        if client is None:
            try:
                import requests
            except Exception as exc:  # pragma: no cover - optional dependency guard
                raise DataSourceError("requests is required for Schwab HTTP calls") from exc
            client = requests.get  # type: ignore[assignment]

        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "accept": "application/json",
        }
        response = client(url, headers=headers, params=params, timeout=self.timeout)
        if not hasattr(response, "status_code"):
            # Build a minimal response wrapper for testing hooks
            return _SchwabResponse(status_code=200, payload=response)
        return response  # type: ignore[return-value]

    @staticmethod
    def _extract_error(payload: Mapping[str, Any], status_code: int) -> str:
        errors = payload.get("errors") if isinstance(payload, Mapping) else None
        if isinstance(errors, list) and errors:
            detail = errors[0].get("detail") or errors[0].get("title")
            return f"Schwab API {status_code}: {detail}"
        return f"Schwab API {status_code}: unexpected error"


__all__ = ["SchwabDataSource"]
