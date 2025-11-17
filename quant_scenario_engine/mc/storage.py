"""Storage policy selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quant_scenario_engine.exceptions import ResourceLimitError
from quant_scenario_engine.utils.resources import estimate_footprint_gb

StorageDecision = Literal["memory", "memmap", "npz"]


@dataclass
class StoragePolicyDecision:
    policy: StorageDecision
    estimated_gb: float


def choose_storage_policy(n_paths: int, n_steps: int, total_ram_gb: float) -> StoragePolicyDecision:
    estimated_gb = estimate_footprint_gb(n_paths, n_steps)
    if estimated_gb >= 0.5 * total_ram_gb:
        raise ResourceLimitError(
            f"Estimated {estimated_gb:.3f} GB exceeds 50% of RAM ({total_ram_gb:.3f} GB)."
        )
    if estimated_gb >= 0.25 * total_ram_gb:
        return StoragePolicyDecision(policy="memmap", estimated_gb=estimated_gb)
    return StoragePolicyDecision(policy="memory", estimated_gb=estimated_gb)

