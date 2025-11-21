"""Replay utilities for rerunning or reporting prior runs (US8)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qse.data.versioning import DataVersion, detect_drift
from qse.exceptions import ConfigConflictError, ConfigValidationError
from qse.schema.run_meta import RunMeta
from qse.simulation.metrics import MetricsReport


def _load_metrics_from_run_meta(meta: RunMeta) -> MetricsReport:
    if meta.metrics is None:
        raise ConfigValidationError("run_meta does not contain stored metrics for replay")
    return MetricsReport(**meta.metrics)


def _load_npz_paths(npz_path: Path) -> np.ndarray:
    if not npz_path.exists():
        raise ConfigValidationError(f"Persisted paths not found: {npz_path}")
    with np.load(npz_path) as data:
        return data["paths"]


def replay_run(
    run_meta_path: Path,
    *,
    allow_data_drift: bool = False,
    current_data_version: DataVersion | None = None,
) -> tuple[RunMeta, MetricsReport, np.ndarray | None]:
    """Replay a prior run using stored metadata.

    Returns (run_meta, metrics, price_paths or None)
    """

    meta = RunMeta.from_json(Path(run_meta_path).read_text())

    if not allow_data_drift and meta.data_drift_status not in (None, "none"):
        raise ConfigConflictError(f"Data drift detected ({meta.data_drift_status}); replay blocked. Use --allow-data-drift to proceed.")

    if current_data_version and meta.data_version:
        prev = DataVersion(**meta.data_version)
        status = detect_drift(prev, current_data_version)
        if status != "none" and not allow_data_drift:
            raise ConfigConflictError(f"Data drift detected during replay: {status}")
        meta.data_drift_status = status

    metrics = _load_metrics_from_run_meta(meta)

    paths = None
    path_hint = meta.config.get("paths_npz") if isinstance(meta.config, dict) else None
    if path_hint:
        paths = _load_npz_paths(Path(path_hint))

    meta.is_replay = True
    meta.original_run_id = meta.original_run_id or meta.run_id
    return meta, metrics, paths
