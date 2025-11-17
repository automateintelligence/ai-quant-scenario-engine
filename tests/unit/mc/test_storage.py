import numpy as np
import pytest

from quant_scenario_engine.mc.storage import PricePath, store_price_paths
from quant_scenario_engine.utils.resources import estimate_footprint_gb


def test_store_price_paths_memory():
    paths = np.ones((10, 5))
    result = store_price_paths(paths, s0=100.0, seed=42, total_ram_gb=100)
    assert isinstance(result, PricePath)
    assert result.storage == "memory"
    assert result.estimated_gb == pytest.approx(estimate_footprint_gb(10, 5))


def test_store_price_paths_memmap_when_large():
    paths = np.ones((1_000_000, 2))
    result = store_price_paths(paths, s0=100.0, seed=42, total_ram_gb=0.05)
    assert result.storage in {"memmap", "npz"}
    assert result.path is not None
