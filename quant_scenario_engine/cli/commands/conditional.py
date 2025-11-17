"""Conditional backtest CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from quant_scenario_engine.cli.validation import validate_screen_inputs
from quant_scenario_engine.data.cache import load_or_fetch, parse_symbol_list
from quant_scenario_engine.features.pipeline import enrich_ohlcv
from quant_scenario_engine.schema.episode import CandidateEpisode
from quant_scenario_engine.simulation.conditional import run_conditional_backtest
from quant_scenario_engine.selectors.gap_volume import GapVolumeSelector
from quant_scenario_engine.utils.logging import get_logger

log = get_logger(__name__, component="cli_conditional")


def conditional(
    universe: str = typer.Option("", help="CSV path or list of symbols (e.g., ['AAPL','MSFT'])"),
    symbols: str = typer.Option("", help="Comma-delimited symbols (alternative to --universe)"),
    start: str = typer.Option(None, help="Start date YYYY-MM-DD when using symbols input"),
    end: str = typer.Option(None, help="End date YYYY-MM-DD when using symbols input"),
    interval: str = typer.Option("1d", help="Data interval"),
    target: Path = typer.Option(Path("data"), help="Target directory for parquet caching"),
    gap_min: float = typer.Option(0.03, help="Minimum absolute gap percentage"),
    volume_z_min: float = typer.Option(1.5, help="Minimum volume z-score"),
    horizon: int = typer.Option(10, help="Episode horizon (bars)"),
    stock_strategy: str = typer.Option("stock_basic", help="Stock strategy name"),
    option_strategy: str = typer.Option(None, help="Option strategy name (optional)"),
    option_type: str = typer.Option("call", help="Option type"),
    strike: float = typer.Option(100.0, help="Option strike"),
    maturity_days: int = typer.Option(30, help="Option maturity in days"),
    iv: float = typer.Option(0.25, help="Implied volatility"),
    rfr: float = typer.Option(0.01, help="Risk-free rate"),
) -> None:
    validate_screen_inputs(horizon=horizon, max_workers=1)
    valid_intervals = {"1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"}
    if interval not in valid_intervals:
        raise typer.Exit(code=1)

    from quant_scenario_engine.models.options import OptionSpec

    option_spec = None
    if option_strategy:
        option_spec = OptionSpec(
            option_type=option_type,
            strike=strike,
            maturity_days=maturity_days,
            implied_vol=iv,
            risk_free_rate=rfr,
            contracts=1,
        )

    grouped: dict[str, pd.DataFrame] = {}
    if universe:
        path = Path(universe)
        if path.exists():
            df = pd.read_csv(path)
            required = {"symbol", "date", "open", "high", "low", "close", "volume"}
            missing = required - set(df.columns)
            if missing:
                raise typer.Exit(code=1)
            df["date"] = pd.to_datetime(df["date"])
            grouped = {sym: g.set_index("date").sort_index() for sym, g in df.groupby("symbol")}
        else:
            symbols = universe

    if not grouped:
        symbol_list = parse_symbol_list(symbols)
        if not symbol_list:
            raise typer.Exit(code=1)
        if not start or not end:
            raise typer.Exit(code=1)
        for sym in symbol_list:
            df = load_or_fetch(sym, start=start, end=end, interval=interval, target=target)
            df = df.sort_values("date")
            grouped[sym] = df.set_index("date")

    selector = GapVolumeSelector(gap_min=gap_min, volume_z_min=volume_z_min, horizon=horizon)

    outputs = []
    for sym, df in grouped.items():
        enriched = enrich_ohlcv(df)
        episodes = selector.select(enriched)
        result = run_conditional_backtest(
            df=enriched,
            episodes=episodes,
            stock_strategy=stock_strategy,
            option_strategy=option_strategy,
            option_spec=option_spec,
        )
        outputs.append(
            {
                "symbol": sym,
                "episode_count": result.episode_count,
                "unconditional": result.unconditional.to_formatted_dict(),
                "conditional": result.conditional.to_formatted_dict() if result.conditional else None,
            }
        )

    typer.echo(json.dumps(outputs, indent=2))
    log.info("conditional command completed", extra={"symbols": list(grouped.keys())})
