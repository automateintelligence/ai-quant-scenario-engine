"""Compare CLI command wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from quant_scenario_engine.cli.validation import validate_compare_inputs
from quant_scenario_engine.config.loader import load_config_with_precedence
from quant_scenario_engine.distributions.factory import get_distribution
from quant_scenario_engine.distributions.integration.model_loader import load_validated_model
from quant_scenario_engine.exceptions import ConfigValidationError
from quant_scenario_engine.models.options import OptionSpec
from quant_scenario_engine.simulation.run import run_compare
from quant_scenario_engine.utils.logging import get_logger
from quant_scenario_engine.utils.progress import ProgressReporter

log = get_logger(__name__, component="cli_compare")


def compare(
    config: Path | None = typer.Option(
        None, "--config", help="Optional YAML/JSON config path"
    ),
    symbol: str | None = typer.Option(None, help="Ticker symbol"),
    s0: float | None = typer.Option(None, help="Starting price"),
    paths: int | None = typer.Option(None, help="Number of MC paths"),
    steps: int | None = typer.Option(None, help="Steps per path"),
    seed: int | None = typer.Option(None, help="Random seed"),
    distribution: str | None = typer.Option(None, help="Return distribution"),
    strategy: str | None = typer.Option(None, "--strategy", help="Stock strategy name (e.g., stock_basic)"),
    option_strategy: str | None = typer.Option(None, "--option-strategy", help="Option strategy name (e.g., call_basic)"),
    strike: float | None = typer.Option(None, help="Option strike"),
    maturity_days: int | None = typer.Option(None, help="Option maturity in days"),
    iv: float | None = typer.Option(None, help="Implied volatility"),
    rfr: float | None = typer.Option(None, help="Risk-free rate"),
    use_audit: bool = typer.Option(True, "--use-audit/--no-use-audit", help="Prefer cached validated distribution models"),
    audit_lookback_days: Optional[int] = typer.Option(None, "--audit-lookback-days", help="Override lookback days when locating cached audits"),
    audit_end_date: Optional[str] = typer.Option(None, "--audit-end-date", help="Override end date (YYYY-MM-DD) for cached audits"),
    audit_data_source: Optional[str] = typer.Option(None, "--audit-data-source", help="Override data source identifier used in the audit cache key"),
    audit_cache_dir: Optional[Path] = typer.Option(None, "--audit-cache-dir", help="Custom directory containing cached audit results"),
) -> None:
    defaults = {
        "symbol": None,
        "s0": 100.0,
        "paths": 1000,
        "steps": 60,
        "seed": 42,
        "distribution": "laplace",
        "strategy": "stock_basic",
        "option_strategy": "call_basic",
        "strike": 100.0,
        "maturity_days": 30,
        "iv": 0.2,
        "rfr": 0.01,
        "use_audit": True,
        "audit_lookback_days": None,
        "audit_end_date": None,
        "audit_data_source": None,
        "audit_cache_dir": None,
    }
    cli_values = {
        "symbol": symbol,
        "s0": s0,
        "paths": paths,
        "steps": steps,
        "seed": seed,
        "distribution": distribution,
        "strategy": strategy,
        "option_strategy": option_strategy,
        "strike": strike,
        "maturity_days": maturity_days,
        "iv": iv,
        "rfr": rfr,
        "use_audit": use_audit,
        "audit_lookback_days": audit_lookback_days,
        "audit_end_date": audit_end_date,
        "audit_data_source": audit_data_source,
        "audit_cache_dir": str(audit_cache_dir) if audit_cache_dir else None,
    }
    casters = {
        "symbol": str,
        "s0": float,
        "paths": int,
        "steps": int,
        "seed": int,
        "distribution": str,
        "strategy": str,
        "option_strategy": str,
        "strike": float,
        "maturity_days": int,
        "iv": float,
        "rfr": float,
        "use_audit": lambda v: str(v).lower() in {"1", "true", "yes", "on"},
        "audit_lookback_days": int,
        "audit_end_date": str,
        "audit_data_source": str,
        "audit_cache_dir": str,
    }

    cfg = load_config_with_precedence(
        config_path=config,
        env_prefix="QSE_",
        cli_values=cli_values,
        defaults=defaults,
        casters=casters,
    )

    if not cfg.get("symbol"):
        raise ConfigValidationError("symbol is required (CLI > ENV > YAML)")

    validate_compare_inputs(
        cfg["paths"],
        cfg["steps"],
        cfg["seed"],
        symbol=cfg["symbol"],
        strike=cfg["strike"],
        maturity_days=cfg["maturity_days"],
        implied_vol=cfg["iv"],
        distribution=cfg["distribution"],
    )

    if str(cfg.get("distribution", "")).lower() == "audit":
        cfg["use_audit"] = True

    progress = ProgressReporter(total=3, log=log, component="compare")
    log.info("Starting compare run", extra={"symbol": cfg["symbol"]})
    dist = None
    audit_metadata = None
    if cfg.get("use_audit"):
        loaded = load_validated_model(
            symbol=cfg["symbol"],
            lookback_days=cfg.get("audit_lookback_days"),
            end_date=cfg.get("audit_end_date"),
            data_source=cfg.get("audit_data_source"),
            cache_dir=cfg.get("audit_cache_dir"),
        )
        dist = loaded.distribution
        audit_metadata = loaded.metadata
        log.info(
            "Using audit-driven distribution",
            extra={
                "model": audit_metadata.get("model_name"),
                "validated": audit_metadata.get("model_validated"),
                "cache_path": audit_metadata.get("cache_path"),
            },
        )
    else:
        dist = get_distribution(cfg["distribution"])
        import numpy as np

        dist.fit(np.random.laplace(0, 0.01, size=500))
    progress.tick("Distribution ready")

    option_spec = OptionSpec(
        option_type="call",
        strike=cfg["strike"],
        maturity_days=cfg["maturity_days"],
        implied_vol=cfg["iv"],
        risk_free_rate=cfg["rfr"],
        contracts=1,
    )
    result = run_compare(
        s0=cfg["s0"],
        distribution=dist,
        n_paths=cfg["paths"],
        n_steps=cfg["steps"],
        seed=cfg["seed"],
        stock_strategy=cfg["strategy"],
        option_strategy=cfg["option_strategy"],
        option_spec=option_spec,
        audit_metadata=audit_metadata,
    )
    progress.tick("Simulation finished")
    typer.echo(result.metrics)
    if result.audit_metadata:
        typer.echo(json.dumps({"distribution_audit": result.audit_metadata}, indent=2))
        if not result.audit_metadata.get("model_validated"):
            fallback_reason = result.audit_metadata.get("fallback_reason")
            warning = "Distribution audit data is not validated"
            if result.audit_metadata.get("cache_stale"):
                warning = "Distribution audit cache is stale; consider re-running audit"
            if fallback_reason:
                warning = f"Audit fallback in effect: {fallback_reason}"
            typer.echo(f"WARNING: {warning}")
    progress.tick("Compare completed")
