"""Cache-aware data loader with staleness detection and versioning."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd

from qse.data.validation import (
    compute_fingerprint,
    validate_option_chain,
    validate_ohlcv,
)
from qse.exceptions import DataSourceError


class DataLoader:
    def __init__(
        self,
        base_dir: Path,
        category: Literal["historical", "features", "option_chains"] = "historical",
        storage_format: Literal["parquet", "pickle"] = "parquet",
        data_source=None,
    ) -> None:
        """Create a loader for OHLCV/feature data.

        `base_dir` should point to the partition root (e.g., data/historical).
        Storage format defaults to parquet; tests may use pickle to avoid optional
        parquet dependencies.
        """

        if category == "historical" and "historical" not in base_dir.parts:
            raise DataSourceError("Historical data must live under data/historical")
        if category == "features" and "features" not in base_dir.parts:
            raise DataSourceError("Feature data must live under data/features")
        if category == "option_chains" and "option_chains" not in base_dir.parts:
            raise DataSourceError("Option chains must live under data/option_chains")

        self.base_dir = base_dir
        self.category = category
        self.storage_format = storage_format
        self.data_source = data_source
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def load_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
        version: str | None = None,
        force_refresh: bool = False,
        allow_stale_cache: bool = False,
    ) -> pd.DataFrame:
        partition_dir = self._resolve_partition(symbol, f"interval={interval}", version)
        parquet_path = partition_dir / "data.parquet"
        pickle_path = partition_dir / "data.pkl"
        cache_meta_path = partition_dir / "data.meta.json"
        if parquet_path.exists():
            data_path = parquet_path
        elif pickle_path.exists():
            data_path = pickle_path
        else:
            data_path = parquet_path if self.storage_format == "parquet" else pickle_path
        cache_path = data_path
        partition_dir.mkdir(parents=True, exist_ok=True)

        if force_refresh:
            df = self._fetch_from_source(symbol, start, end, interval)
            self._write_cache(data_path, cache_meta_path, df, symbol, start, end)
            return df

        if data_path.exists() and cache_meta_path.exists():
            try:
                cache_meta = json.loads(cache_meta_path.read_text())
                fetched_at = pd.Timestamp(cache_meta.get("fetched_at"))
                cached_start = pd.Timestamp(cache_meta.get("start"))
                cached_end = pd.Timestamp(cache_meta.get("end"))
                last_close = cache_meta.get("last_close")
                stale_threshold = timedelta(days=1 if interval == "1d" else 0)
                is_stale = (datetime.utcnow() - fetched_at.to_pydatetime()) > stale_threshold
                has_coverage = cached_start <= pd.Timestamp(start) and cached_end >= pd.Timestamp(end)
                if has_coverage and not is_stale:
                    df = self._read_cache(cache_path, cache_meta)
                    return df.loc[start:end]

                if not is_stale and cached_end < pd.Timestamp(end):
                    # Corporate action detection via overlap bar
                    overlap_start = cached_end.date().isoformat()
                    overlap_end = (cached_end + timedelta(days=1)).date().isoformat()
                    overlap = self._fetch_from_source(symbol, overlap_start, overlap_end, interval)
                    if len(overlap) > 0 and last_close is not None:
                        fresh_close = float(overlap.iloc[0]["close"])
                        divergence = abs(fresh_close - last_close) / last_close
                        if divergence > 0.01:
                            # Trigger full refresh
                            df = self._fetch_from_source(symbol, start, end, interval)
                            self._write_cache(data_path, cache_meta_path, df, symbol, start, end)
                            return df

                    # No corporate action detected, fetch incremental
                    incremental = self._fetch_from_source(
                        symbol, cached_end.date().isoformat(), end, interval
                    )
                    df_cached = self._read_cache(cache_path, cache_meta)
                    df = pd.concat([df_cached, incremental]).sort_index()
                    df = df[~df.index.duplicated(keep="last")]
                    self._write_cache(
                        data_path, cache_meta_path, df, symbol, cached_start.isoformat(), end
                    )
                    return df.loc[start:end]
            except Exception:
                if not allow_stale_cache:
                    raise

            if allow_stale_cache:
                try:
                    meta = json.loads(cache_meta_path.read_text())
                    return self._read_cache(cache_path, meta)
                except Exception:
                    pass

        df = self._fetch_from_source(symbol, start, end, interval)
        self._write_cache(data_path, cache_meta_path, df, symbol, start, end)
        return df

    def load_option_chain(
        self,
        symbol: str,
        as_of: str,
        expiry: str | None = None,
        version: str | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        partition_key = f"option_chain={expiry or 'all'}"
        partition_dir = self._resolve_partition(symbol, partition_key, version, as_of=as_of)
        data_path = partition_dir / ("data.parquet" if self.storage_format == "parquet" else "data.pkl")
        cache_meta_path = partition_dir / "data.meta.json"
        partition_dir.mkdir(parents=True, exist_ok=True)

        if not force_refresh and data_path.exists() and cache_meta_path.exists():
            try:
                meta = json.loads(cache_meta_path.read_text())
                cached_expiry = meta.get("expiry")
                cached_as_of = meta.get("as_of")
                if cached_expiry == (expiry or "all") and cached_as_of == as_of:
                    return self._read_cache(data_path, meta)
            except Exception:
                if not force_refresh:
                    pass

        if not hasattr(self.data_source, "fetch_option_chain"):
            raise DataSourceError("Configured data source does not support option chains")
        df = self.data_source.fetch_option_chain(symbol=symbol, expiry=expiry)
        validate_option_chain(df)
        if data_path.suffix == ".parquet":
            df.to_parquet(data_path)
        else:
            df.to_pickle(data_path)
        meta = {
            "symbol": symbol,
            "expiry": expiry or "all",
            "as_of": as_of,
            "fetched_at": datetime.utcnow().isoformat(),
            "storage_format": self.storage_format,
            "data_source": getattr(self.data_source, "name", None),
        }
        cache_meta_path.write_text(json.dumps(meta, indent=2))
        return df

    def _fetch_from_source(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
        if self.data_source is None:
            raise DataSourceError("No data source configured; provide data_source or override")
        if hasattr(self.data_source, "fetch_ohlcv"):
            return self.data_source.fetch_ohlcv(symbol=symbol, start=start, end=end, interval=interval)
        return self.data_source.fetch(symbol=symbol, start=start, end=end, interval=interval)

    def _write_cache(
        self,
        data_path: Path,
        cache_meta_path: Path,
        df: pd.DataFrame,
        symbol: str,
        start: str,
        end: str,
    ) -> None:
        validate_ohlcv(df)
        if data_path.suffix == ".parquet":
            df.to_parquet(data_path)
        else:
            df.to_pickle(data_path)
        meta = {
            "symbol": symbol,
            "start": start,
            "end": end,
            "fetched_at": datetime.utcnow().isoformat(),
            "fingerprint": compute_fingerprint(df),
            "last_close": float(df["close"].iloc[-1]),
            "storage_format": self.storage_format,
            "data_source": getattr(self.data_source, "name", None),
        }
        cache_meta_path.write_text(json.dumps(meta, indent=2))

    def _read_cache(self, cache_path: Path, meta: dict) -> pd.DataFrame:
        storage_format = meta.get("storage_format")
        if storage_format is None:
            storage_format = "pickle" if cache_path.suffix == ".pkl" else "parquet"
        if storage_format == "pickle":
            target = cache_path if cache_path.suffix == ".pkl" else cache_path.with_suffix(".pkl")
            return pd.read_pickle(target)
        target = cache_path if cache_path.suffix == ".parquet" else cache_path.with_suffix(".parquet")
        return pd.read_parquet(target)

    def _resolve_partition(
        self, symbol: str, partition_key: str, version: str | None, *, as_of: str | None = None
    ) -> Path:
        partition_root = self.base_dir / partition_key / f"symbol={symbol}"
        if as_of is not None:
            partition_root = partition_root / f"as_of={as_of}"
        if version is None:
            version_dirs = []
            if partition_root.exists():
                version_dirs = [
                    p for p in partition_root.iterdir() if p.is_dir() and p.name.startswith("_v")
                ]
            if version_dirs:
                version = sorted(version_dirs, key=lambda p: int(p.name[2:]))[-1].name
            else:
                version = "_v1"
        if not str(version).startswith("_v"):
            version = f"_v{version}"
        return partition_root / str(version)
