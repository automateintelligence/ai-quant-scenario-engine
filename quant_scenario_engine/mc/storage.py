"""Storage policy selection utilities and PricePath container."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np

from quant_scenario_engine.exceptions import ResourceLimitError
from quant_scenario_engine.utils.resources import estimate_footprint_gb, select_storage_policy

StorageDecision = Literal["memory", "memmap", "npz"]


@dataclass
class PricePath:
    paths: np.ndarray
    n_paths: int
    n_steps: int
    s0: float
    storage: StorageDecision
    seed: int | None
    estimated_gb: float
    path: Path | None = None  # file when memmap/npz


def store_price_paths(paths: np.ndarray, s0: float, seed: int | None, total_ram_gb: float = 16.0) -> PricePath:
    n_paths, n_steps = paths.shape
    policy, est = select_storage_policy(n_paths, n_steps, total_ram_gb=total_ram_gb)

    if policy == "memory":
        return PricePath(paths=paths, n_paths=n_paths, n_steps=n_steps, s0=s0, storage=policy, seed=seed, estimated_gb=est)

    # memmap/npz: persist to temp file
    tmpdir = Path(tempfile.mkdtemp())
    if policy == "memmap":
        file_path = tmpdir / "price_paths.dat"
        mmap = np.memmap(file_path, dtype=paths.dtype, mode="w+", shape=paths.shape)
        mmap[:] = paths[:]
        mmap.flush()
        return PricePath(paths=mmap, n_paths=n_paths, n_steps=n_steps, s0=s0, storage=policy, seed=seed, estimated_gb=est, path=file_path)

    # npz fallback
    file_path = tmpdir / "price_paths.npz"
    np.savez_compressed(file_path, paths=paths)
    return PricePath(paths=np.load(file_path)["paths"], n_paths=n_paths, n_steps=n_steps, s0=s0, storage=policy, seed=seed, estimated_gb=est, path=file_path)
