"""Data validation utilities for OHLCV Parquet files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

from qse.exceptions import SchemaError, TimestampAnomalyError

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
OPTION_CHAIN_REQUIRED_COLUMNS = [
    "expiry",
    "strike",
    "option_type",
    "bid",
    "ask",
    "implied_volatility",
    "open_interest",
    "volume",
]


@dataclass
class ValidationResult:
    fingerprint: str
    is_stale: bool


def validate_ohlcv(df: pd.DataFrame, allow_future: bool = False) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise SchemaError("Index must be datetime-like")
    if not df.index.is_monotonic_increasing:
        raise TimestampAnomalyError("Index must be monotonically increasing")
    now_ts = pd.Timestamp.now(tz=df.index.tz)
    if not allow_future and (df.index.max() > now_ts):
        raise TimestampAnomalyError("Data contains future timestamps")


def compute_fingerprint(df: pd.DataFrame) -> str:
    """Return SHA256 hash of canonicalized OHLCV data."""
    payload = df[REQUIRED_COLUMNS].to_csv(index=True).encode()
    return hashlib.sha256(payload).hexdigest()


def fingerprints_match(fp_old: str, fp_new: str) -> bool:
    return fp_old == fp_new


def enforce_missing_tolerance(df: pd.DataFrame, max_gap: int = 3, max_ratio: float = 0.01) -> tuple[int, float]:
    """Enforce missing data tolerance.

    Returns tuple of (largest_gap, missing_ratio) for observability.
    """

    gaps = df[REQUIRED_COLUMNS].isna().any(axis=1)
    largest_gap = 0
    current_gap = 0
    for missing in gaps:
        if missing:
            current_gap += 1
            largest_gap = max(largest_gap, current_gap)
        else:
            current_gap = 0
    missing_ratio = gaps.mean()
    if largest_gap > max_gap or missing_ratio > max_ratio:
        raise TimestampAnomalyError(
            f"Missing data exceeds tolerance (gap={largest_gap}, ratio={missing_ratio:.4f})"
        )
    return largest_gap, float(missing_ratio)


def validate_option_chain(df: pd.DataFrame) -> None:
    missing = [col for col in OPTION_CHAIN_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise SchemaError(f"Missing option chain fields: {missing}")
    if df.empty:
        raise SchemaError("Option chain is empty")
