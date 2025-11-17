"""Run metadata schema with JSON serialization."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional

from quant_scenario_engine.exceptions import ConfigValidationError


@dataclass
class ReproducibilityContext:
    seed: int
    library_versions: Dict[str, str]
    system_info: Dict[str, Any]
    git_sha: Optional[str]


@dataclass
class RunMeta:
    run_id: str
    symbol: str
    config: Dict[str, Any]
    storage_policy: str
    drift_status: Optional[str] = None
    iv_source: Optional[str] = None
    parameter_stability: Optional[str] = None
    covariance_estimator: Optional[str] = None
    var_method: Optional[str] = None
    lookback_window: Optional[int] = None
    reproducibility: Optional[ReproducibilityContext] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def write_atomic(self, path: Path) -> None:
        """Write run_meta to a temporary file then move for atomicity."""
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(self.to_json())
        tmp_path.replace(path)

    @classmethod
    def from_json(cls, raw: str) -> "RunMeta":
        data = json.loads(raw)
        return cls(**data)

    @classmethod
    def capture_context(
        cls,
        run_id: str,
        symbol: str,
        config: Dict[str, Any],
        storage_policy: str,
        seed: int,
        covariance_estimator: Optional[str] = None,
        var_method: Optional[str] = None,
        lookback_window: Optional[int] = None,
        drift_status: Optional[str] = None,
        iv_source: Optional[str] = None,
        parameter_stability: Optional[str] = None,
    ) -> "RunMeta":
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
            drift_status=drift_status,
            iv_source=iv_source,
            parameter_stability=parameter_stability,
            covariance_estimator=covariance_estimator,
            var_method=var_method,
            lookback_window=lookback_window,
            reproducibility=reproducibility,
        )


def _capture_lib_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for lib in ["numpy", "pandas", "scipy", "numba", "statsmodels", "arch", "typer"]:
        try:
            module = __import__(lib)
            versions[lib] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[lib] = "missing"
    return versions


def _capture_git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None

