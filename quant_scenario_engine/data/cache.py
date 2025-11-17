"""Helpers for fetching and caching OHLCV data in Parquet."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_symbol(symbol: str, start: str, end: str, interval: str, target: Path) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    df = df.reset_index()
    df.columns = df.columns.str.lower()
    df["symbol"] = symbol
    df["interval"] = interval

    output_dir = target / "historical" / f"interval={interval}" / f"symbol={symbol}" / "_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "data.parquet"
    df.to_parquet(output_file, index=False, engine="pyarrow", compression="snappy")
    return df


def load_or_fetch(symbol: str, start: str, end: str, interval: str, target: Path) -> pd.DataFrame:
    path = target / "historical" / f"interval={interval}" / f"symbol={symbol}" / "_v1" / "data.parquet"
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    if path.exists():
        df = pd.read_parquet(path, engine="pyarrow")
        df["date"] = pd.to_datetime(df["date"])
        min_date, max_date = df["date"].min(), df["date"].max()
        if start_ts < min_date or end_ts > max_date:
            df = fetch_symbol(symbol, start, end, interval, target)
        else:
            mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
            df = df.loc[mask].copy()
    else:
        df = fetch_symbol(symbol, start, end, interval, target)

    return df


def parse_symbol_list(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, (list, tuple)):
            return [str(v).strip("'\" ") for v in val if str(v).strip()]
    except Exception:
        pass
    cleaned = raw.strip("[]")
    return [s.strip().strip("'\"") for s in cleaned.split(",") if s.strip().strip("'\"")]


__all__ = ["fetch_symbol", "load_or_fetch", "parse_symbol_list"]

