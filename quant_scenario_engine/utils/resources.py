"""Resource estimation utilities."""

from __future__ import annotations

import shutil
from typing import Literal, Tuple

from quant_scenario_engine.exceptions import ResourceLimitError

StoragePolicy = Literal["memory", "memmap", "npz"]


def estimate_footprint_gb(n_paths: int, n_steps: int) -> float:
    """Estimate memory footprint in gigabytes with 10% overhead."""
    return n_paths * n_steps * 8 * 1.1 / 1e9


def select_storage_policy(n_paths: int, n_steps: int, total_ram_gb: float | None = None) -> Tuple[StoragePolicy, float]:
    """Return recommended storage policy and estimated footprint.

    Thresholds: <25% RAM -> memory, ≥25% -> memmap, ≥50% -> abort.
    """

    estimated_gb = estimate_footprint_gb(n_paths, n_steps)
    if total_ram_gb is None:
        total_ram_gb = shutil.disk_usage("/").total / 1e9  # best-effort fallback

    if estimated_gb >= 0.5 * total_ram_gb:
        raise ResourceLimitError(
            f"Estimated footprint {estimated_gb:.3f} GB exceeds 50% of RAM ({total_ram_gb:.3f} GB)."
        )
    if estimated_gb >= 0.25 * total_ram_gb:
        return "memmap", estimated_gb
    return "memory", estimated_gb

