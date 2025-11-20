import json
import os
from pathlib import Path

from quant_scenario_engine.distributions.cache.cache_manager import get_cache_path
from quant_scenario_engine.distributions.integration.model_loader import load_validated_model


def _write_payload(path: Path, loc: float = 0.25, scale: float = 0.05) -> None:
    payload = {
        "symbol": path.stem.split("_")[0],
        "models": [{"name": "laplace", "cls": "LaplaceFitter", "config": {}}],
        "fit_results": [
            {
                "model_name": "laplace",
                "log_likelihood": -1.0,
                "aic": 1.0,
                "bic": 1.5,
                "params": {"loc": loc, "scale": scale},
                "n": 120,
                "converged": True,
                "heavy_tailed": True,
                "fit_success": True,
                "warnings": [],
                "error": None,
                "fit_message": None,
            }
        ],
        "tail_metrics": [],
        "var_backtests": [],
        "simulation_metrics": [],
        "scores": [
            {
                "model": "laplace",
                "total_score": 0.95,
                "components": {"aic": 0.1, "tail": 0.2, "var": 0.3, "cluster": 0.35},
            }
        ],
        "best_model": {"name": "laplace", "cls": "LaplaceFitter", "config": {}},
        "best_fit": {
            "model_name": "laplace",
            "log_likelihood": -1.0,
            "aic": 1.0,
            "bic": 1.5,
            "params": {"loc": loc, "scale": scale},
            "n": 120,
            "converged": True,
            "heavy_tailed": True,
            "fit_success": True,
            "warnings": [],
            "error": None,
            "fit_message": None,
        },
        "tail_reports": {},
        "realism_reports": {},
        "selection_report": {"winner": "laplace"},
    }
    path.write_text(json.dumps(payload))


def test_model_loader_falls_back_when_cache_missing(tmp_path):
    loaded = load_validated_model(
        symbol="XYZ",
        lookback_days=100,
        end_date="2024-01-01",
        data_source="unit",
        cache_dir=str(tmp_path),
    )

    assert loaded.metadata["model_validated"] is False
    assert loaded.metadata["fallback_reason"] == "no cached audit available"
    assert hasattr(loaded.distribution, "sample")


def test_model_loader_reads_cached_model(tmp_path):
    cache_dir = Path(tmp_path)
    cache_path = get_cache_path(cache_dir, "ABC", 200, "2024-02-02", "unit")
    _write_payload(cache_path)

    loaded = load_validated_model(
        symbol="ABC",
        lookback_days=200,
        end_date="2024-02-02",
        data_source="unit",
        cache_dir=str(cache_dir),
    )

    assert loaded.metadata["model_validated"] is True
    assert loaded.metadata["model_name"] == "laplace"
    assert loaded.distribution.loc == 0.25
    assert loaded.distribution.scale == 0.05


def test_model_loader_uses_latest_cache_when_key_unknown(tmp_path):
    cache_dir = Path(tmp_path)
    older = get_cache_path(cache_dir, "ABC", 100, "2024-01-01", "unit")
    newer = get_cache_path(cache_dir, "ABC", 200, "2024-02-02", "unit")
    _write_payload(older, loc=0.1, scale=0.02)
    os.utime(older, (0, 0))  # force old timestamp
    _write_payload(newer, loc=0.3, scale=0.08)

    loaded = load_validated_model(
        symbol="ABC",
        lookback_days=None,
        end_date=None,
        data_source=None,
        cache_dir=str(cache_dir),
    )

    assert loaded.metadata["model_name"] == "laplace"
    assert loaded.distribution.loc == 0.3
    assert loaded.metadata.get("cache_path", "").endswith(newer.name)


def test_model_loader_skips_stale_cache_when_disallowed(tmp_path):
    cache_dir = Path(tmp_path)
    stale = get_cache_path(cache_dir, "ABC", 50, "2023-01-01", "unit")
    fresh = get_cache_path(cache_dir, "ABC", 150, "2024-03-01", "unit")
    _write_payload(stale, loc=0.05, scale=0.01)
    os.utime(stale, (0, 0))
    _write_payload(fresh, loc=0.4, scale=0.09)

    loaded = load_validated_model(
        symbol="ABC",
        lookback_days=None,
        end_date=None,
        data_source=None,
        cache_dir=str(cache_dir),
        allow_stale=False,
    )

    assert loaded.distribution.loc == 0.4
    assert loaded.metadata.get("cache_path", "").endswith(fresh.name)
