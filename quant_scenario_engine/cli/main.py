"""Typer CLI entrypoint with structured error handling."""

from __future__ import annotations

import sys

import typer

from quant_scenario_engine.cli.commands.compare import compare
from quant_scenario_engine.cli.commands.fetch import fetch
from quant_scenario_engine.cli.commands.screen import screen
from quant_scenario_engine.cli.commands.conditional import conditional
from quant_scenario_engine.exceptions import (
    ConfigValidationError,
    DistributionFitError,
    InsufficientDataError,
    ResourceLimitError,
)
from quant_scenario_engine.utils.logging import configure_logging, get_logger

app = typer.Typer(help="Quant Scenario Engine CLI")


app.command()(fetch)
app.command()(compare)
app.command()(screen)
app.command()(conditional)


log = get_logger(__name__, component="cli")


def main() -> None:
    configure_logging(component="cli")
    try:
        app()
    except ConfigValidationError as exc:
        log.error(str(exc))
        raise typer.Exit(code=1)
    except InsufficientDataError as exc:
        log.error(f"Data validation failed: {exc}")
        raise typer.Exit(code=2)
    except DistributionFitError as exc:
        log.error(f"Distribution fitting failed: {exc}")
        raise typer.Exit(code=3)
    except ResourceLimitError as exc:
        log.error(f"Resource limit exceeded: {exc}")
        raise typer.Exit(code=4)
    except KeyboardInterrupt:
        log.info("Shutdown requested. Finishing current tasks...")
        raise typer.Exit(code=130)
    except Exception:
        log.exception("Unhandled exception")
        raise typer.Exit(code=255)


if __name__ == "__main__":
    # Use sys.exit to ensure proper exit code propagation under raw python invocation
    sys.exit(main())
