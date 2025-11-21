"""Data versioning utilities for drift detection (US8)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import pandas as pd

DriftKind = Literal["none", "schema", "distribution", "count"]


@dataclass
class DataVersion:
    schema_hash: str
    row_count: int
    mean_return: float
    std_return: float


def compute_version(df: pd.DataFrame, *, return_column: str = "close") -> DataVersion:
    schema_repr = "|".join([str((c, str(df[c].dtype))) for c in df.columns])
    schema_hash = hashlib.sha256(schema_repr.encode()).hexdigest()
    row_count = len(df)
    returns = df[return_column].pct_change().dropna()
    mean_return = float(returns.mean()) if not returns.empty else 0.0
    std_return = float(returns.std()) if not returns.empty else 0.0
    return DataVersion(schema_hash=schema_hash, row_count=row_count, mean_return=mean_return, std_return=std_return)


def detect_drift(old: DataVersion, new: DataVersion, *, row_threshold: float = 0.05, stat_threshold: float = 0.2) -> DriftKind:
    if old.schema_hash != new.schema_hash:
        return "schema"
    if old.row_count == 0:
        return "none"
    row_change = abs(new.row_count - old.row_count) / max(1, old.row_count)
    if row_change > row_threshold:
        return "count"
    mean_change = abs(new.mean_return - old.mean_return) / (abs(old.mean_return) + 1e-9)
    std_change = abs(new.std_return - old.std_return) / (abs(old.std_return) + 1e-9)
    if mean_change > stat_threshold or std_change > stat_threshold:
        return "distribution"
    return "none"
