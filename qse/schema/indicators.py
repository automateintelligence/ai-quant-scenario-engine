"""Indicator definition schema for dynamic feature registration (US3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qse.exceptions import ConfigValidationError


@dataclass(slots=True)
class IndicatorDefinition:
    """Declarative indicator spec for the registry.

    Attributes:
        name: Output feature name (e.g., "rsi_14").
        function: Registry key (e.g., "rsi", "sma", "volume_z").
        source: Input column to read from (e.g., "close", "volume").
        params: Additional keyword params for the indicator function.
    """

    name: str
    function: str
    source: str = "close"
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.function:
            raise ConfigValidationError("indicator name and function are required")
        if not isinstance(self.params, dict):  # defensive: protect downstream **params
            raise ConfigValidationError("indicator params must be a mapping")
