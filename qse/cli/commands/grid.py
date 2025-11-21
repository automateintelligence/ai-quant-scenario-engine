"""Grid CLI wiring (US2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from qse.cli.validation import validate_grid_inputs
from qse.config.loader import load_config_with_precedence
from qse.distributions.factory import get_distribution
from qse.exceptions import ConfigValidationError
from qse.simulation.grid import ObjectiveWeights, run_grid
from qse.utils.logging import get_logger
from qse.utils.progress import ProgressReporter

log = get_logger(__name__, component="cli_grid")


def _load_grid_from_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ConfigValidationError(f"Grid file not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ConfigValidationError("pyyaml is required to load YAML grid files") from exc
        content = yaml.safe_load(path.read_text())
        return content if isinstance(content, list) else [content]
    raise ConfigValidationError("Grid file must be JSON or YAML")


def _parse_objective_weights(raw: object) -> ObjectiveWeights | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigValidationError(f"Invalid objective_weights JSON: {exc}") from exc
    if isinstance(raw, dict):
        return ObjectiveWeights(
            mean_pnl=float(raw.get("mean_pnl", 0.3)),
            sharpe=float(raw.get("sharpe", 0.3)),
            max_drawdown=float(raw.get("max_drawdown", 0.2)),
            cvar=float(raw.get("cvar", 0.2)),
        )
    raise ConfigValidationError("objective_weights must be a JSON object with weights")


def grid(
    config: Path | None = typer.Option(None, "--config", help="Optional YAML/JSON config path"),
    grid_path: Path | None = typer.Option(None, "--grid", help="Grid definition file (JSON/YAML)"),
    grid_json: str | None = typer.Option(None, "--grid-json", help="Inline grid JSON overriding file/config"),
    s0: float | None = typer.Option(None, help="Starting price"),
    paths: int | None = typer.Option(None, help="Monte Carlo paths per config"),
    steps: int | None = typer.Option(None, help="Steps per path"),
    seed: int | None = typer.Option(None, help="Random seed"),
    distribution: str | None = typer.Option(None, help="Return distribution"),
    stock_strategy: str | None = typer.Option(None, help="Stock strategy name"),
    option_strategy: str | None = typer.Option(None, help="Option strategy name"),
    strike: float | None = typer.Option(None, help="Option strike"),
    maturity_days: int | None = typer.Option(None, help="Option maturity in days"),
    iv: float | None = typer.Option(None, "--iv", help="Implied volatility"),
    rfr: float | None = typer.Option(None, "--rfr", help="Risk-free rate"),
    contracts: int | None = typer.Option(None, help="Number of option contracts"),
    max_workers: int | None = typer.Option(None, help="Maximum worker processes"),
    objective_weights: str | None = typer.Option(None, help="JSON of objective weights"),
    output: Path | None = typer.Option(None, help="Output path for grid results JSON"),
) -> None:
    defaults = {
        "s0": 100.0,
        "paths": 1000,
        "steps": 60,
        "seed": 42,
        "distribution": "laplace",
        "stock_strategy": "stock_basic",
        "option_strategy": "option_call",
        "strike": 100.0,
        "maturity_days": 30,
        "iv": 0.2,
        "rfr": 0.01,
        "contracts": 1,
        "max_workers": None,
        "grid": None,
        "objective_weights": None,
        "output": "runs/grid_results.json",
    }

    cli_grid = None
    if grid_path:
        cli_grid = _load_grid_from_file(grid_path)
    elif grid_json:
        cli_grid = json.loads(grid_json)

    cli_values = {
        "s0": s0,
        "paths": paths,
        "steps": steps,
        "seed": seed,
        "distribution": distribution,
        "stock_strategy": stock_strategy,
        "option_strategy": option_strategy,
        "strike": strike,
        "maturity_days": maturity_days,
        "iv": iv,
        "rfr": rfr,
        "contracts": contracts,
        "max_workers": max_workers,
        "grid": cli_grid,
        "objective_weights": objective_weights,
        "output": str(output) if output else None,
    }

    casters = {
        "s0": float,
        "paths": int,
        "steps": int,
        "seed": int,
        "distribution": str,
        "stock_strategy": str,
        "option_strategy": str,
        "strike": float,
        "maturity_days": int,
        "iv": float,
        "rfr": float,
        "contracts": int,
        "max_workers": int,
        "grid": lambda v: json.loads(v),
        "objective_weights": str,
        "output": str,
    }

    cfg = load_config_with_precedence(
        config_path=config,
        env_prefix="QSE_",
        cli_values=cli_values,
        defaults=defaults,
        casters=casters,
    )

    grid_def = cfg.get("grid")
    if isinstance(grid_def, str):
        grid_def = json.loads(grid_def)
    if isinstance(grid_def, dict):
        grid_def = [grid_def]

    validate_grid_inputs(
        paths=cfg["paths"],
        steps=cfg["steps"],
        seed=cfg["seed"],
        grid=grid_def,
        max_workers=cfg.get("max_workers"),
    )

    weights = _parse_objective_weights(cfg.get("objective_weights"))

    option_spec_defaults = {
        "option_type": "call",
        "strike": cfg["strike"],
        "maturity_days": cfg["maturity_days"],
        "implied_vol": cfg["iv"],
        "risk_free_rate": cfg["rfr"],
        "contracts": cfg["contracts"],
    }

    dist = get_distribution(cfg["distribution"])
    # Fit with small synthetic returns to activate parameters if not provided
    import numpy as np

    dist.fit(np.random.laplace(0, 0.01, size=500))

    progress = ProgressReporter(total=len(grid_def) if isinstance(grid_def, list) else None, log=log, component="grid")
    log.info("Starting grid run", extra={"paths": cfg["paths"], "steps": cfg["steps"]})
    results = run_grid(
        distribution=dist,
        s0=cfg["s0"],
        n_paths=cfg["paths"],
        n_steps=cfg["steps"],
        seed=cfg["seed"],
        strategy_grids=grid_def,
        option_spec_defaults=option_spec_defaults,
        default_stock_strategy=cfg["stock_strategy"],
        default_option_strategy=cfg["option_strategy"],
        max_workers=cfg.get("max_workers"),
        objective_weights=weights,
        output_path=Path(cfg["output"]),
    )
    progress.tick("Grid completed")

    typer.echo(f"Completed grid with {len(results)} configs. Top score: {results[0].objective_score:.4f}")
    for r in results:
        typer.echo(
            json.dumps(
                {
                    "config_index": r.config_index,
                    "status": r.status,
                    "objective_score": r.objective_score,
                },
                indent=2,
            )
        )
