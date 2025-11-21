"""Run metadata schema with JSON serialization."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ReproducibilityContext:
    seed: int
    library_versions: dict[str, str]
    system_info: dict[str, Any]
    git_sha: str | None


@dataclass
class RunMeta:
    run_id: str
    symbol: str
    config: dict[str, Any]
    storage_policy: str
    run_type: str | None = None
    data_version: dict[str, Any] | None = None
    data_drift_status: str | None = None
    drift_status: str | None = None
    iv_source: str | None = None
    parameter_stability: str | None = None
    covariance_estimator: str | None = None
    var_method: str | None = None
    lookback_window: int | None = None
    reproducibility: ReproducibilityContext | None = None
    distribution_audit: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    is_replay: bool = False
    original_run_id: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def write_atomic(self, path: Path) -> None:
        """Write run_meta to a temporary file then move for atomicity."""
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(self.to_json())
        tmp_path.replace(path)

    @classmethod
    def from_json(cls, raw: str) -> RunMeta:
        data = json.loads(raw)
        return cls(**data)

    @classmethod
    def capture_context(
        cls,
        run_id: str,
        symbol: str,
        config: dict[str, Any],
        storage_policy: str,
        seed: int,
        covariance_estimator: str | None = None,
        var_method: str | None = None,
        lookback_window: int | None = None,
        drift_status: str | None = None,
        data_version: dict[str, Any] | None = None,
        data_drift_status: str | None = None,
        iv_source: str | None = None,
        parameter_stability: str | None = None,
        distribution_audit: dict[str, Any] | None = None,
        run_type: str | None = None,
        metrics: dict[str, Any] | None = None,
        original_run_id: str | None = None,
        is_replay: bool = False,
    ) -> RunMeta:
        reproducibility = ReproducibilityContext(
            seed=seed,
            library_versions=_capture_lib_versions(),
            system_info={
                "os": platform.platform(),
                "cpu_count": os.cpu_count(),
                "python_version": platform.python_version(),
            },
            git_sha=_capture_git_sha(),
        )
        return cls(
            run_id=run_id,
            symbol=symbol,
            config=config,
            storage_policy=storage_policy,
            run_type=run_type,
            data_version=data_version,
            data_drift_status=data_drift_status,
            drift_status=drift_status,
            iv_source=iv_source,
            parameter_stability=parameter_stability,
            covariance_estimator=covariance_estimator,
            var_method=var_method,
            lookback_window=lookback_window,
            reproducibility=reproducibility,
            distribution_audit=distribution_audit,
            metrics=metrics,
            original_run_id=original_run_id,
            is_replay=is_replay,
        )


def _capture_lib_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for lib in ["numpy", "pandas", "scipy", "numba", "statsmodels", "arch", "typer"]:
        try:
            module = __import__(lib)
            versions[lib] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[lib] = "missing"
    return versions


def _capture_git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None
