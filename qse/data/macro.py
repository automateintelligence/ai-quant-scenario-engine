"""Macro series loader and alignment helpers (US3 FR-014)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from qse.exceptions import ConfigValidationError
from qse.utils.logging import get_logger

log = get_logger(__name__, component="data.macro")


def _load_series(source: Path | str | pd.Series | pd.DataFrame, *, column: str | None = None) -> pd.Series:
    if isinstance(source, pd.Series):
        return source.sort_index()
    if isinstance(source, pd.DataFrame):
        if column is None:
            if source.shape[1] != 1:
                raise ConfigValidationError("macro DataFrame has multiple columns; specify 'column'")
            column = source.columns[0]
        return source[column].sort_index()
    path = Path(source)
    if not path.exists():
        raise ConfigValidationError(f"macro series file not found: {path}")
    df = pd.read_csv(path)
    dt_col = None
    for cand in ("date", "timestamp", "time"):
        if cand in df.columns:
            dt_col = cand
            break
    if dt_col is None:
        raise ConfigValidationError("macro CSV must include a date/timestamp column")
    if column is None:
        value_cols = [c for c in df.columns if c != dt_col]
        if len(value_cols) != 1:
            raise ConfigValidationError("macro CSV must include exactly one value column when 'column' not provided")
        column = value_cols[0]
    series = pd.Series(df[column].values, index=pd.to_datetime(df[dt_col]))
    return series.sort_index()


def align_macro_series(
    source: Path | str | pd.Series | pd.DataFrame,
    target_index: Iterable,
    *,
    column: str | None = None,
    method: str = "forward_fill",
    max_gap_multiple: int = 3,
) -> pd.Series:
    """Align macro series to target index with tolerance enforcement.

    method: forward_fill | linear
    max_gap_multiple: maximum consecutive gaps (in target frequency units) allowed for fills.
    """

    series = _load_series(source, column=column)
    target_index = pd.DatetimeIndex(target_index)
    aligned = series.reindex(target_index)

    # Infer stride count for tolerance; fallback to simple count if freq missing
    limit = max_gap_multiple if max_gap_multiple > 0 else None

    if method == "forward_fill":
        aligned = aligned.ffill(limit=limit).bfill(limit=limit)
    elif method == "linear":
        aligned = aligned.interpolate(method="time", limit=limit).ffill(limit=limit).bfill(limit=limit)
    else:
        raise ConfigValidationError("method must be 'forward_fill' or 'linear'")

    remaining_nas = aligned.isna()
    if remaining_nas.any():
        log.warning(
            "Macro alignment exceeded tolerance; gaps remain",
            extra={"remaining_gaps": int(remaining_nas.sum()), "method": method, "max_gap_multiple": max_gap_multiple},
        )

    return aligned
