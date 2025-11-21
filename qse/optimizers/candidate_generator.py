"""Stage 2 candidate structure generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import pandas as pd

from qse.optimizers.models import CandidateStructure, Leg

REQUIRED_COLUMNS = {"expiry", "strike", "option_type", "mid"}


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for structure generation."""

    min_width: int = 1
    max_width: int = 3


class CandidateGenerator:
    """Generate option structures from filtered strikes."""

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()

    def generate(self, chain: pd.DataFrame, spot: float) -> List[CandidateStructure]:
        """Create Stage 2 candidate structures.

        Returns an ordered list containing verticals, iron condors, straddles,
        and strangles for each expiry present in ``chain``.
        """

        if chain.empty:
            return []
        self._validate_columns(chain)

        df = chain.copy()
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
        df = df.dropna(subset=["expiry", "strike", "mid"])

        candidates: List[CandidateStructure] = []
        for expiry, slice_df in df.groupby("expiry"):
            candidates.extend(self._generate_verticals(slice_df))
            condor = self._generate_iron_condor(slice_df, spot)
            if condor:
                candidates.append(condor)
            straddle = self._generate_straddle(slice_df, spot)
            if straddle:
                candidates.append(straddle)
            strangle = self._generate_strangle(slice_df, spot)
            if strangle:
                candidates.append(strangle)

        return candidates

    def _generate_verticals(self, df: pd.DataFrame) -> List[CandidateStructure]:
        candidates: List[CandidateStructure] = []
        for option_type in ("call", "put"):
            subset = df[df["option_type"] == option_type].sort_values("strike")
            rows: Sequence[pd.Series] = list(subset.itertuples(index=False))
            for i, short in enumerate(rows):
                for offset in range(self.config.min_width, self.config.max_width + 1):
                    j = i + offset
                    if j >= len(rows):
                        break
                    long = rows[j]
                    width = float(offset)
                    legs = [
                        Leg(
                            option_type=option_type,
                            strike=float(short.strike),
                            expiry=short.expiry,
                            side="sell",
                            premium=float(short.mid),
                        ),
                        Leg(
                            option_type=option_type,
                            strike=float(long.strike),
                            expiry=long.expiry,
                            side="buy",
                            premium=float(long.mid),
                        ),
                    ]
                    candidates.append(
                        CandidateStructure(
                            structure_type="vertical",
                            legs=legs,
                            expiry=pd.Timestamp(short.expiry),
                            width=width,
                        )
                    )
        return candidates

    def _generate_iron_condor(self, df: pd.DataFrame, spot: float) -> CandidateStructure | None:
        puts = df[df["option_type"] == "put"].sort_values("strike", ascending=False)
        calls = df[df["option_type"] == "call"].sort_values("strike")

        if len(puts) < 2 or len(calls) < 2:
            return None

        short_put = puts.iloc[0]
        long_put = puts.iloc[1]
        short_call = calls.iloc[0]
        long_call = calls.iloc[1]

        width_put = 1.0
        width_call = 1.0
        max_width = max(width_put, width_call)

        if max_width < self.config.min_width or max_width > self.config.max_width:
            return None

        legs = [
            Leg(
                option_type="put",
                strike=float(short_put.strike),
                expiry=short_put.expiry,
                side="sell",
                premium=float(short_put.mid),
            ),
            Leg(
                option_type="put",
                strike=float(long_put.strike),
                expiry=long_put.expiry,
                side="buy",
                premium=float(long_put.mid),
            ),
            Leg(
                option_type="call",
                strike=float(short_call.strike),
                expiry=short_call.expiry,
                side="sell",
                premium=float(short_call.mid),
            ),
            Leg(
                option_type="call",
                strike=float(long_call.strike),
                expiry=long_call.expiry,
                side="buy",
                premium=float(long_call.mid),
            ),
        ]
        return CandidateStructure(
            structure_type="iron_condor",
            legs=legs,
            expiry=pd.Timestamp(short_put.expiry),
            width=max_width,
        )

    def _generate_straddle(self, df: pd.DataFrame, spot: float) -> CandidateStructure | None:
        calls = df[df["option_type"] == "call"].copy()
        puts = df[df["option_type"] == "put"].copy()
        common_strikes = sorted(set(calls["strike"]).intersection(set(puts["strike"])), key=lambda k: abs(k - spot))
        if not common_strikes:
            return None
        strike = common_strikes[0]
        call_row = calls.loc[calls["strike"] == strike].iloc[0]
        put_row = puts.loc[puts["strike"] == strike].iloc[0]
        legs = [
            Leg(
                option_type="call",
                strike=float(strike),
                expiry=call_row.expiry,
                side="buy",
                premium=float(call_row.mid),
            ),
            Leg(
                option_type="put",
                strike=float(strike),
                expiry=put_row.expiry,
                side="buy",
                premium=float(put_row.mid),
            ),
        ]
        return CandidateStructure(
            structure_type="straddle",
            legs=legs,
            expiry=pd.Timestamp(call_row.expiry),
            width=0.0,
        )

    def _generate_strangle(self, df: pd.DataFrame, spot: float) -> CandidateStructure | None:
        calls = df[df["option_type"] == "call"].sort_values("strike")
        puts = df[df["option_type"] == "put"].sort_values("strike", ascending=False)
        call_row = calls[calls["strike"] > spot].head(1)
        put_row = puts[puts["strike"] < spot].head(1)
        if call_row.empty or put_row.empty:
            return None

        call_leg = call_row.iloc[0]
        put_leg = put_row.iloc[0]
        width = float(call_leg.strike) - float(put_leg.strike)
        legs = [
            Leg(
                option_type="call",
                strike=float(call_leg.strike),
                expiry=call_leg.expiry,
                side="buy",
                premium=float(call_leg.mid),
            ),
            Leg(
                option_type="put",
                strike=float(put_leg.strike),
                expiry=put_leg.expiry,
                side="buy",
                premium=float(put_leg.mid),
            ),
        ]
        return CandidateStructure(
            structure_type="strangle",
            legs=legs,
            expiry=pd.Timestamp(call_leg.expiry),
            width=abs(width),
        )

    def _validate_columns(self, chain: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS.difference(chain.columns)
        if missing:
            raise ValueError(f"Filtered chain missing required columns: {sorted(missing)}")


__all__ = ["CandidateGenerator", "GeneratorConfig"]
