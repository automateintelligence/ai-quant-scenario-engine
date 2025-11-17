"""Screen CLI command wiring."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import typer

from quant_scenario_engine.cli.validation import validate_screen_inputs
from quant_scenario_engine.features.pipeline import enrich_ohlcv
from quant_scenario_engine.selectors.gap_volume import GapVolumeSelector
from quant_scenario_engine.simulation.screen import screen_universe, run_strategy_screen
from quant_scenario_engine.data.cache import safe_load_or_fetch, parse_symbol_list
from quant_scenario_engine.selectors.loader import load_selector
from quant_scenario_engine.utils.logging import get_logger

log = get_logger(__name__, component="cli_screen")


def screen(
    universe: str = typer.Option("", help="Path to CSV file with OHLCV data"),
    symbols: str = typer.Option("", help="Symbol list: ['AAPL','MSFT'] or AAPL,MSFT"),
    start: str = typer.Option(None, help="Start date YYYY-MM-DD when using symbols input"),
    end: str = typer.Option(None, help="End date YYYY-MM-DD when using symbols input"),
    interval: str = typer.Option("1d", help="Data interval"),
    target: Path = typer.Option(Path("data"), help="Target directory for parquet caching"),
    gap_min: float = typer.Option(0.03, help="Minimum absolute gap percentage"),
    volume_z_min: float = typer.Option(1.5, help="Minimum volume z-score"),
    horizon: int = typer.Option(10, help="Episode horizon (bars)"),
    strategy: str = typer.Option(None, help="Stock strategy for screening (Mode B/C)"),
    rank_by: str = typer.Option("sharpe", help="Metric to rank symbols when strategy provided"),
    conditional_file: str = typer.Option(None, help="Optional selector YAML for conditional mode"),
    top: int | None = typer.Option(None, help="Top N candidates to keep"),
    max_workers: int = typer.Option(4, help="Max workers for screening"),
    output: Path = typer.Option(Path("runs"), help="Output directory for artifacts"),
    lookback_years: float = typer.Option(None, help="Optional lookback horizon in years for universe data"),
) -> None:
    validate_screen_inputs(horizon=horizon, max_workers=max_workers)
    valid_intervals = {"1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"}
    if interval not in valid_intervals:
        raise typer.Exit(code=1)

    grouped: dict[str, pd.DataFrame] = {}

    if universe:
        # Universe must be a CSV file path
        path = Path(universe)
        if not path.exists():
            log.error("universe CSV file not found", extra={"path": str(path)})
            raise typer.Exit(code=1)
        df = pd.read_csv(path)
        required = {"symbol", "date", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            log.error("universe CSV missing required columns", extra={"missing": list(missing)})
            raise typer.Exit(code=1)
        df["date"] = pd.to_datetime(df["date"])
        grouped = {sym: g.set_index("date").sort_index() for sym, g in df.groupby("symbol")}

    if not grouped and symbols:
        symbol_list = parse_symbol_list(symbols)
        if not symbol_list:
            log.error("no valid symbols provided")
            raise typer.Exit(code=1)
        if not start or not end:
            log.error("--start and --end are required when using --symbols")
            raise typer.Exit(code=1)
        for sym in symbol_list:
            df = safe_load_or_fetch(sym, start=start, end=end, interval=interval, target=target)
            if df is None or df.empty:
                log.warning("no data for symbol", extra={"symbol": sym})
                continue
            df = df.sort_values("date")
            grouped[sym] = df.set_index("date")

    if not grouped:
        if not universe and not symbols:
            log.error("must provide either --universe (CSV file) or --symbols (ticker list)")
            raise typer.Exit(code=1)
        log.error("no symbols with data available")
        raise typer.Exit(code=2)

    # Enrich each symbol's data with features
    enriched = {sym: enrich_ohlcv(g) for sym, g in grouped.items()}

    output.mkdir(parents=True, exist_ok=True)
    # lookback_years currently informational; hook for future slicing logic

    # Mode A: selector-only
    if not strategy:
        selector = GapVolumeSelector(gap_min=gap_min, volume_z_min=volume_z_min, horizon=horizon)
        candidates = screen_universe(universe=enriched, selector=selector, max_workers=max_workers, top_n=top)
        payload = [
            {
                "symbol": c.symbol,
                "t0": c.t0.isoformat(),
                "horizon": c.horizon,
                "selector": c.selector_name,
                "state_features": c.state_features,
                "score": c.score,
            }
            for c in candidates
        ]
        result_obj = {"candidates": payload}
        target_file = output / "screen_results_unconditional.json"
        target_file.write_text(json.dumps(result_obj, indent=2))
        typer.echo(json.dumps(result_obj, indent=2))
        log.info("screen command completed", extra={"candidates": len(payload)})
        return

    # Mode B/C: strategy screening
    selector = None
    if conditional_file:
        selector = load_selector(Path(conditional_file))
    results = run_strategy_screen(
        universe=enriched,
        strategy=strategy,
        rank_by=rank_by,
        selector=selector,
        min_episodes=10,
        top_n=top,
    )

    payload = []
    for res in results:
        entry = {
            "symbol": res.symbol,
            "rank_metric": res.rank_metric,
            "metrics_unconditional": res.metrics_unconditional.to_formatted_dict(),
            "low_confidence": res.low_confidence,
        }
        if res.metrics_conditional:
            entry["metrics_conditional"] = res.metrics_conditional.to_formatted_dict()
            entry["episode_count"] = res.episode_count
            entry["comparison"] = res.comparison
        payload.append(entry)

    result_obj = {"results": payload}
    suffix = "conditional" if selector else "unconditional"
    target_file = output / f"screen_results_{suffix}.json"
    target_file.write_text(json.dumps(result_obj, indent=2))
    typer.echo(json.dumps(result_obj, indent=2))
    log.info("strategy screen completed", extra={"symbols": len(payload), "output": str(target_file)})
