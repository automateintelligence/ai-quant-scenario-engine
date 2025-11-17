"""Data validation utilities for OHLCV Parquet files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Tuple

import pandas as pd

from quant_scenario_engine.exceptions import SchemaError, TimestampAnomalyError

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class ValidationResult:
    fingerprint: str
    is_stale: bool


def validate_ohlcv(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")
    if not df.index.is_monotonic_increasing:
        raise TimestampAnomalyError("Index must be monotonically increasing")


def compute_fingerprint(df: pd.DataFrame) -> str:
    """Return SHA256 hash of canonicalized OHLCV data."""
    payload = df[REQUIRED_COLUMNS].to_csv(index=True).encode()
    return hashlib.sha256(payload).hexdigest()


def enforce_missing_tolerance(df: pd.DataFrame, max_gap: int = 3, max_ratio: float = 0.01) -> Tuple[int, float]:
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

