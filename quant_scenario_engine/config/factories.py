"""Factory helpers for component creation with logging."""

from __future__ import annotations

import logging
from typing import Callable, Generic, TypeVar

T = TypeVar("T")

log = logging.getLogger(__name__)


class FactoryBase(Generic[T]):
    def __init__(self, name: str, builder: Callable[[], T]) -> None:
        self.name = name
        self.builder = builder

    def create(self, prior: str | None = None) -> T:
        component = self.builder()
        log.info(
            "Component loaded",
            extra={"type": component.__class__.__name__, "name": self.name, "prior": prior},
        )
        return component

