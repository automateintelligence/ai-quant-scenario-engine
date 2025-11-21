"""Dynamic indicator registry for configurable feature computation (US3)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Iterable

import pandas as pd

from qse.exceptions import ConfigValidationError
from qse.features import indicators as builtin
from qse.schema.indicators import IndicatorDefinition
from qse.utils.logging import get_logger

log = get_logger(__name__, component="features.registry")


IndicatorFn = Callable[[pd.Series, dict], pd.Series]


class IndicatorRegistry:
    """Registry for indicator callables addressable by string key.

    Supports pluggable indicator definitions supplied via config. Unknown
    indicators raise structured validation errors; missing input columns emit
    warnings and are skipped.
    """

    def __init__(self) -> None:
        self._registry: dict[str, Callable[..., pd.Series]] = {}
        self._register_builtin()

    def _register_builtin(self) -> None:
        self.register("sma", lambda series, params: builtin.compute_sma(series, length=int(params.get("length", 20))))
        self.register("rsi", lambda series, params: builtin.compute_rsi(series, length=int(params.get("length", 14))))
        self.register(
            "volume_z",
            lambda series, params: builtin.compute_volume_z(series, window=int(params.get("window", 20))),
        )

    def register(self, name: str, fn: Callable[..., pd.Series]) -> None:
        if not name:
            raise ConfigValidationError("indicator key cannot be empty")
        self._registry[name.lower()] = fn

    def apply(
        self,
        df: pd.DataFrame,
        definitions: Iterable[IndicatorDefinition],
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Apply indicator definitions to a DataFrame.

        Returns: (augmented_df, added_features, missing_features)
        """

        out = df.copy()
        added: list[str] = []
        missing: list[str] = []

        for definition in definitions:
            key = definition.function.lower()
            fn = self._registry.get(key)
            if fn is None:
                raise ConfigValidationError(f"Unknown indicator function '{definition.function}'")

            source_col = definition.source
            if source_col not in out.columns:
                log.warning(
                    "Indicator source column missing; skipping",
                    extra={"indicator": definition.name, "source": source_col},
                )
                missing.append(definition.name)
                continue

            series = out[source_col]
            try:
                result = fn(series, definition.params)
            except Exception as exc:  # pragma: no cover - protective guard rails
                log.error(
                    "Indicator computation failed",
                    extra={"indicator": definition.name, "function": definition.function, "error": str(exc)},
                )
                missing.append(definition.name)
                continue

            out[definition.name] = result
            added.append(definition.name)

        return out, added, missing


def apply_indicators(
    df: pd.DataFrame, definitions: Iterable[IndicatorDefinition], *, registry: IndicatorRegistry | None = None
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Helper to apply indicator registry with defaults."""

    reg = registry or IndicatorRegistry()
    return reg.apply(df, definitions)
