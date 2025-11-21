"""Profiling utilities for performance budgets (Phase 11 T125/T125a)."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

from qse.utils.logging import get_logger

log = get_logger(__name__, component="profiling")


@dataclass
class Timing:
    wall: float
    cpu: float


def _now() -> Timing:
    return Timing(wall=time.perf_counter(), cpu=time.process_time())


@contextmanager
def track_time(name: str, *, warn_budget: float | None = None, error_budget: float | None = None) -> Iterator[Timing]:
    start = _now()
    try:
        yield start
    finally:
        end = _now()
        wall_elapsed = end.wall - start.wall
        cpu_elapsed = end.cpu - start.cpu
        level = None
        if error_budget is not None and wall_elapsed >= error_budget:
            level = "error"
        elif warn_budget is not None and wall_elapsed >= warn_budget:
            level = "warning"
        if level:
            getattr(log, level)(
                "Performance budget exceeded",
                extra={"segment": name, "wall_seconds": round(wall_elapsed, 4), "cpu_seconds": round(cpu_elapsed, 4)},
            )
        else:
            log.info(
                "Segment timing",
                extra={"segment": name, "wall_seconds": round(wall_elapsed, 4), "cpu_seconds": round(cpu_elapsed, 4)},
            )


def budget_checker(total_budget: float, *, warn_ratio: float = 0.5, error_ratio: float = 0.9) -> Callable[[float], None]:
    """Return a function to check elapsed time against budget ratios."""

    warn_threshold = total_budget * warn_ratio
    error_threshold = total_budget * error_ratio

    def check(elapsed_seconds: float) -> None:
        if elapsed_seconds >= error_threshold:
            log.error(
                "Runtime exceeded 90% of budget",
                extra={"elapsed_seconds": round(elapsed_seconds, 3), "budget_seconds": total_budget},
            )
        elif elapsed_seconds >= warn_threshold:
            log.warning(
                "Runtime exceeded 50% of budget",
                extra={"elapsed_seconds": round(elapsed_seconds, 3), "budget_seconds": total_budget},
            )

    return check

