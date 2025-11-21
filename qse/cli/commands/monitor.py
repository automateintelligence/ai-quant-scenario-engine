"""CLI command for live position monitoring (US8, Phase 10)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from qse.config.loader import _load_yaml
from qse.data.factory import FallbackDataSource, get_data_source
from qse.monitoring import PositionMonitor, load_position
from qse.utils.logging import get_logger

console = Console()
log = get_logger(__name__, component="cli.monitor")


def _load_config_snapshot(config_path: Path | None) -> dict:
    if config_path is None:
        return {}
    return _load_yaml(config_path)


def monitor(
    position: Path = typer.Option(..., "--position", help="Path to position JSON"),
    interval: int = typer.Option(300, "--interval", help="Polling interval seconds"),
    data_source: str = typer.Option("schwab_stub", "--data-source", help="Primary data source"),
    fallback: Optional[str] = typer.Option(None, "--fallback", help="Fallback data source"),
    config: Optional[Path] = typer.Option(
        None, "--config", help="Optional config.yml/json to load regimes + mc settings"
    ),
    iterations: Optional[int] = typer.Option(
        1,
        "--iterations",
        help="Number of monitoring iterations (<=0 runs until alert)",
    ),
) -> None:
    """Monitor a saved position and trigger alerts when thresholds are crossed."""

    try:
        position_snapshot = load_position(position)
    except Exception as exc:
        console.print(f"[red]Failed to load position: {exc}[/red]")
        raise typer.Exit(code=1)

    config_snapshot = _load_config_snapshot(config)
    regimes_cfg = config_snapshot.get("regimes", position_snapshot.config_snapshot.get("regimes", {}))
    mc_cfg = config_snapshot.get("mc", position_snapshot.config_snapshot.get("mc", {}))

    try:
        primary = get_data_source(data_source)
        provider = (
            FallbackDataSource(primary, get_data_source(fallback), logger=log)
            if fallback
            else primary
        )
    except Exception as exc:  # pragma: no cover - dependency errors shown to user
        console.print(f"[red]Failed to initialize data source: {exc}[/red]")
        raise typer.Exit(code=2)

    monitor_loop = PositionMonitor(provider, regimes_config=regimes_cfg, mc_config=mc_cfg, logger=log)
    max_iters = None if iterations is None or iterations <= 0 else iterations

    console.print(
        f"[bold cyan]Monitoring {position_snapshot.underlying}[/bold cyan] "
        f"(interval={interval}s, regime={position_snapshot.regime})"
    )

    for result in monitor_loop.monitor(position_snapshot, interval_seconds=interval, iterations=max_iters):
        table = Table(title=f"Position mark @ {result['as_of']}")
        table.add_column("Leg")
        table.add_column("Mark")
        table.add_column("PnL")
        for idx, leg in enumerate(result["leg_marks"], start=1):
            table.add_row(
                f"{idx}: {leg['side']} {leg['option_type']} {leg['strike']} {leg['expiry']}",
                f"{leg['mark']:.2f}",
                f"{leg['pnl']:.2f}",
            )
        console.print(table)
        console.print(
            f"Remaining horizon: {result['remaining_horizon']}d | Mark PnL: {result['mark_pnl']:.2f}"
        )
        sim = result.get("simulation", {})
        if sim.get("paths", 0) > 0:
            console.print(
                f"Simulated mean PnL: {sim.get('mean_pnl', 0.0):.2f} | POP: {sim.get('pop', 0.0):.2%}"
            )

        if result["alert"]["triggered"]:
            console.print(
                f"[bold green]Alert triggered[/bold green]: {result['alert']['reason']} "
                f"(PnL={result['alert']['pnl']:.2f})"
            )
            break

    log.info("Monitoring completed", extra={"iterations": max_iters or "until-alert"})


__all__ = ["monitor"]
