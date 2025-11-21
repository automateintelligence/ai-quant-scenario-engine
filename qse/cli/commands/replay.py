"""Replay CLI wiring (US8)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from qse.config.loader import load_config_with_precedence
from qse.cli.validation import require_positive
from qse.data.versioning import compute_version
from qse.exceptions import ConfigConflictError, ConfigValidationError
from qse.simulation.replay import replay_run
from qse.utils.logging import get_logger

log = get_logger(__name__, component="cli_replay")


def replay(
    config: Path | None = typer.Option(None, "--config", help="Optional YAML/JSON config path"),
    run_meta_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to run_meta.json to replay"),
    allow_data_drift: bool = typer.Option(False, "--allow-data-drift", help="Proceed even if drift detected"),
    current_data_path: Path | None = typer.Option(None, help="Optional CSV of current data for drift check"),
) -> None:
    defaults = {
        "run_meta_path": str(run_meta_path),
        "allow_data_drift": allow_data_drift,
        "current_data_path": str(current_data_path) if current_data_path else None,
    }
    casters = {
        "run_meta_path": str,
        "allow_data_drift": lambda v: str(v).lower() in {"1", "true", "yes", "on"},
        "current_data_path": str,
    }
    merged = load_config_with_precedence(
        config_path=config,
        env_prefix="QSE_",
        cli_values=defaults,
        defaults=defaults,
        casters=casters,
    )
    run_meta_path = Path(merged["run_meta_path"])
    allow_data_drift = merged["allow_data_drift"]
    current_data_path = Path(merged["current_data_path"]) if merged.get("current_data_path") else None

    current_version = None
    if current_data_path:
        df = None
        try:
            import pandas as pd

            df = pd.read_csv(current_data_path)
            if "close" not in df.columns:
                raise ConfigValidationError("current_data_path must include a 'close' column")
            current_version = compute_version(df.set_index(df.columns[0]))
        except Exception as exc:
            raise ConfigValidationError(f"Failed to load current data for drift check: {exc}")

    meta, metrics, paths = replay_run(run_meta_path, allow_data_drift=allow_data_drift, current_data_version=current_version)
    log.info("Replay loaded", extra={"run_id": meta.run_id, "original_run_id": meta.original_run_id})

    typer.echo(json.dumps({"metrics": json.loads(metrics.to_json()) if hasattr(metrics, "to_json") else meta.metrics}, indent=2))
    if paths is not None:
        typer.echo(f"Loaded persisted paths from {meta.config.get('paths_npz')}")
