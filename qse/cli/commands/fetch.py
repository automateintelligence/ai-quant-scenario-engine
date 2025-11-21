"""CLI command for fetching historical market data via Schwab/yfinance."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from qse.data.data_loader import DataLoader
from qse.data.factory import FallbackDataSource, get_data_source
from qse.config.loader import load_config_with_precedence
from qse.utils.logging import get_logger

console = Console()
log = get_logger(__name__, component="cli.fetch")


def fetch(
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yml file"),
    symbol: str = typer.Option(..., "--symbol", help="Stock ticker symbol (e.g., AAPL)"),
    start: str = typer.Option(
        ..., "--start", help="Start date in YYYY-MM-DD format (e.g., 2018-01-01)"
    ),
    end: str = typer.Option(
        ..., "--end", help="End date in YYYY-MM-DD format (e.g., 2024-12-31)"
    ),
    interval: str = typer.Option(
        "1d", "--interval", help="Data interval: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo"
    ),
    target: Path = typer.Option(Path("data"), "--target", help="Target directory for parquet output"),
    data_source: str = typer.Option(
        "schwab", "--data-source", help="Primary data provider (schwab|yfinance|schwab_stub)",
    ),
    allow_fallback: bool = typer.Option(
        True, "--allow-fallback/--no-fallback", help="Fallback to yfinance on Schwab errors"
    ),
    access_token: Optional[str] = typer.Option(None, "--access-token", envvar="SCHWAB_ACCESS_TOKEN"),
    timeout: float = typer.Option(10.0, "--timeout", help="HTTP timeout seconds for Schwab calls"),
    max_retries: int = typer.Option(3, "--max-retries", help="Retries for yfinance downloads"),
) -> None:
    """
    Fetch historical market data and save as Parquet.

    Downloads OHLCV data via yfinance and stores in partitioned Parquet format:
    target/historical/interval={interval}/symbol={symbol}/_v1/data.parquet

    Example:
        python -m qse.cli.fetch --symbol AAPL --start 2018-01-01 --end 2024-12-31 --interval 1d --target data/
    """
    defaults = {
        "symbol": symbol,
        "start": start,
        "end": end,
        "interval": interval,
        "target": str(target),
        "data_source": data_source,
        "allow_fallback": allow_fallback,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    casters = {
        "symbol": str,
        "start": str,
        "end": str,
        "interval": str,
        "target": str,
        "data_source": str,
        "allow_fallback": lambda v: str(v).lower() in {"1", "true", "yes", "on"},
        "timeout": float,
        "max_retries": int,
    }
    merged = load_config_with_precedence(
        config_path=config,
        env_prefix="QSE_",
        cli_values=defaults,
        defaults=defaults,
        casters=casters,
    )
    symbol = merged["symbol"]
    start = merged["start"]
    end = merged["end"]
    interval = merged["interval"]
    target = Path(merged["target"])
    data_source = merged["data_source"]
    allow_fallback = merged["allow_fallback"]
    timeout = merged["timeout"]
    max_retries = merged["max_retries"]

    # Validate dates
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        if start_dt >= end_dt:
            console.print("[red]Error: Start date must be before end date[/red]")
            raise typer.Exit(code=1)
    except ValueError as exc:
        console.print(f"[red]Error: Invalid date format - {exc}[/red]")
        raise typer.Exit(code=1)

    # Validate interval
    valid_intervals = ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
    if interval not in valid_intervals:
        console.print(
            f"[red]Error: Invalid interval '{interval}'. Must be one of: {', '.join(valid_intervals)}[/red]"
        )
        raise typer.Exit(code=1)

    # Setup output directory
    output_dir = target / "historical"

    console.print(f"[bold cyan]Fetching {symbol} data[/bold cyan]")
    console.print(f"  Period: {start} to {end}")
    console.print(f"  Interval: {interval}")
    console.print(f"  Provider: {data_source}")

    # Fetch data with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Downloading {symbol} data...", total=None)

        try:
            primary = get_data_source(
                data_source, access_token=access_token, timeout=timeout, max_retries=max_retries
            )
            provider = (
                FallbackDataSource(primary, get_data_source("yfinance", max_retries=max_retries), logger=log)
                if allow_fallback and data_source != "yfinance"
                else primary
            )

            loader = DataLoader(
                base_dir=output_dir, data_source=provider, storage_format="parquet", category="historical"
            )
            df = loader.load_ohlcv(symbol, start, end, interval=interval)

            saved_path = loader._resolve_partition(symbol, f"interval={interval}", version=None) / "data.parquet"
            progress.stop()
            console.print(f"[green]✓[/green] Saved {len(df)} rows to {saved_path}")
            log.info(
                f"Fetched {symbol} data: {len(df)} rows from {start} to {end} (interval={interval}) via {provider.name}"
            )

        except Exception as exc:
            progress.stop()
            console.print(f"[red]Error during fetch: {exc}[/red]")
            log.exception(f"Failed to fetch data for {symbol}")
            raise typer.Exit(code=3)
