"""Stage 0/1 filtering for option candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List

import pandas as pd


@dataclass(frozen=True)
class Stage0Config:
    """Configuration for expiry selection (Stage 0)."""

    min_dte: int = 7
    max_dte: int = 45
    min_expiries: int = 3
    max_expiries: int = 5


@dataclass(frozen=True)
class Stage1Config:
    """Configuration for moneyness and liquidity filters (Stage 1)."""

    moneyness_low: float = 0.85
    moneyness_high: float = 1.15
    min_volume: int = 1
    min_open_interest: int = 1
    max_bid_ask_pct: float = 0.25


REQUIRED_COLUMNS = {"expiry", "strike", "option_type", "bid", "ask", "volume", "open_interest"}


def _validate_columns(chain: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(chain.columns)
    if missing:
        raise ValueError(f"Option chain missing required columns: {sorted(missing)}")


def select_expiries(chain: pd.DataFrame, as_of: datetime, config: Stage0Config | None = None) -> List[pd.Timestamp]:
    """Select expiries within the configured DTE window.

    Raises:
        ValueError: if fewer than ``min_expiries`` are available in the window or
            when the option chain lacks required columns.
    """

    config = config or Stage0Config()
    _validate_columns(chain)

    expiries = pd.to_datetime(chain["expiry"], errors="coerce").dropna().drop_duplicates()
    dtes = (expiries - pd.Timestamp(as_of)).dt.days
    windowed = expiries[(dtes >= config.min_dte) & (dtes <= config.max_dte)].sort_values()
    selected = list(windowed[: config.max_expiries])

    if len(selected) < config.min_expiries:
        raise ValueError(
            f"Found {len(selected)} expiries in DTE window; need at least {config.min_expiries}"
        )

    return selected


def filter_strikes(
    chain: pd.DataFrame,
    spot: float,
    expiries: Iterable[pd.Timestamp],
    config: Stage1Config | None = None,
) -> pd.DataFrame:
    """Filter strikes using moneyness and liquidity thresholds.

    Returns a new ``DataFrame`` restricted to the provided ``expiries``.
    """

    config = config or Stage1Config()
    _validate_columns(chain)

    df = chain.copy()
    df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
    df = df[df["expiry"].isin(list(expiries))].dropna(subset=["expiry", "strike", "bid", "ask"])

    if df.empty:
        return df

    df = df[df["strike"] > 0]
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df = df[df["mid"] > 0]

    df["moneyness"] = df["strike"] / spot
    df = df[
        (df["moneyness"] >= config.moneyness_low)
        & (df["moneyness"] <= config.moneyness_high)
        & (df["volume"] >= config.min_volume)
        & (df["open_interest"] >= config.min_open_interest)
    ]

    spread = df["ask"] - df["bid"]
    df["spread_pct"] = (spread / df["mid"]).fillna(pd.NA)
    df = df[(df["spread_pct"] <= config.max_bid_ask_pct) & df["spread_pct"].notna()]

    return df.sort_values(["expiry", "strike", "option_type"]).reset_index(drop=True)
