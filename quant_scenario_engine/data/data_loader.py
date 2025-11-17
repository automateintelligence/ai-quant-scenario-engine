"""Cache-aware data loader with staleness detection."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from quant_scenario_engine.exceptions import DataSourceError
from quant_scenario_engine.data.validation import compute_fingerprint, validate_ohlcv


class DataLoader:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
        force_refresh: bool = False,
        allow_stale_cache: bool = False,
    ) -> pd.DataFrame:
        cache_path = self.cache_dir / interval / f"{symbol}.parquet"
        cache_meta_path = cache_path.with_suffix(".meta.json")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if force_refresh:
            df = self._fetch_from_source(symbol, start, end, interval)
            self._write_cache(cache_path, cache_meta_path, df, symbol, start, end)
            return df

        if cache_path.exists() and cache_meta_path.exists():
            try:
                cache_meta = json.loads(cache_meta_path.read_text())
                fetched_at = pd.Timestamp(cache_meta.get("fetched_at"))
                cached_start = pd.Timestamp(cache_meta.get("start"))
                cached_end = pd.Timestamp(cache_meta.get("end"))
                stale_threshold = timedelta(days=1 if interval == "1d" else 0)
                is_stale = (datetime.utcnow() - fetched_at.to_pydatetime()) > stale_threshold
                has_coverage = cached_start <= pd.Timestamp(start) and cached_end >= pd.Timestamp(end)
                if has_coverage and not is_stale:
                    df = pd.read_parquet(cache_path)
                    return df.loc[start:end]
            except Exception:
                if not allow_stale_cache:
                    raise

            if allow_stale_cache:
                try:
                    return pd.read_parquet(cache_path)
                except Exception:
                    pass

        df = self._fetch_from_source(symbol, start, end, interval)
        self._write_cache(cache_path, cache_meta_path, df, symbol, start, end)
        return df

    def _fetch_from_source(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
        raise DataSourceError("No data source configured; override _fetch_from_source in subclass")

    def _write_cache(
        self,
        cache_path: Path,
        cache_meta_path: Path,
        df: pd.DataFrame,
        symbol: str,
        start: str,
        end: str,
    ) -> None:
        validate_ohlcv(df)
        df.to_parquet(cache_path)
        meta = {
            "symbol": symbol,
            "start": start,
            "end": end,
            "fetched_at": datetime.utcnow().isoformat(),
            "fingerprint": compute_fingerprint(df),
            "last_close": float(df["close"].iloc[-1]),
        }
        cache_meta_path.write_text(json.dumps(meta, indent=2))

