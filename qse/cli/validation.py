"""CLI validation helpers (lightweight stubs)."""

from __future__ import annotations

from qse.exceptions import ConfigValidationError


def require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ConfigValidationError(f"{name} must be > 0")


def validate_compare_inputs(
    paths: int,
    steps: int,
    seed: int | None,
    *,
    symbol: str | None = None,
    strike: float | None = None,
    maturity_days: int | None = None,
    implied_vol: float | None = None,
    distribution: str | None = None,
) -> None:
    require_positive("paths", paths)
    require_positive("steps", steps)
    if seed is None:
        raise ConfigValidationError("seed is required for reproducibility")
    if symbol is not None and not symbol.strip():
        raise ConfigValidationError("symbol is required")
    if strike is not None:
        require_positive("strike", strike)
    if maturity_days is not None:
        require_positive("maturity_days", maturity_days)
    if implied_vol is not None:
        if implied_vol <= 0 or implied_vol >= 5:
            raise ConfigValidationError("implied_vol must be between 0 and 5")
    if distribution:
        allowed = {"laplace", "student_t"}
        if distribution not in allowed:
            raise ConfigValidationError(f"distribution must be one of {sorted(allowed)}")


def validate_screen_inputs(*, horizon: int, max_workers: int) -> None:
    require_positive("horizon", horizon)
    require_positive("max_workers", max_workers)


def validate_grid_inputs(
    *,
    paths: int,
    steps: int,
    seed: int | None,
    grid: object,
    max_workers: int | None = None,
) -> None:
    require_positive("paths", paths)
    require_positive("steps", steps)
    if seed is None:
        raise ConfigValidationError("seed is required for reproducibility")
    if grid is None:
        raise ConfigValidationError("grid definitions are required")
    if isinstance(grid, list) and len(grid) == 0:
        raise ConfigValidationError("grid definitions are required")
    if max_workers is not None and max_workers <= 0:
        raise ConfigValidationError("max_workers must be > 0")
