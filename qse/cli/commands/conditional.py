"""Conditional backtest and conditional Monte Carlo CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from qse.cli.validation import validate_screen_inputs
from qse.config.loader import load_config_with_precedence
from qse.data.cache import load_or_fetch, parse_symbol_list
from qse.features.pipeline import enrich_ohlcv
from qse.schema.episode import CandidateEpisode
from qse.simulation.conditional import run_conditional_backtest
from qse.simulation.conditional_mc import run_conditional_mc
from qse.selectors.gap_volume import GapVolumeSelector
from qse.utils.logging import get_logger

log = get_logger(__name__, component="cli_conditional")


def conditional(
    config: Path | None = typer.Option(None, "--config", help="Optional YAML/JSON config path"),
    universe: str = typer.Option("", help="Path to CSV file with OHLCV data"),
    symbols: str = typer.Option("", help="Symbol list: ['AAPL','MSFT'] or AAPL,MSFT"),
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
    mode: str = typer.Option("backtest", help="Mode: backtest or monte_carlo"),
    paths: int = typer.Option(1000, help="Number of MC paths (monte_carlo mode)"),
    steps: int = typer.Option(60, help="MC steps (monte_carlo mode)"),
    seed: int = typer.Option(42, help="Random seed"),
    distribution: str = typer.Option("laplace", help="Return distribution for MC"),
    state: str = typer.Option("", help="JSON string of state features for conditioning"),
    distance_threshold: float = typer.Option(2.0, help="Similarity threshold for state matching"),
    use_audit: bool = typer.Option(True, "--use-audit/--no-use-audit", help="Prefer validated distribution audit models"),
) -> None:
    defaults = {
        "universe": universe,
        "symbols": symbols,
        "start": start,
        "end": end,
        "interval": interval,
        "target": str(target),
        "gap_min": gap_min,
        "volume_z_min": volume_z_min,
        "horizon": horizon,
        "stock_strategy": stock_strategy,
        "option_strategy": option_strategy,
        "option_type": option_type,
        "strike": strike,
        "maturity_days": maturity_days,
        "iv": iv,
        "rfr": rfr,
        "mode": mode,
        "paths": paths,
        "steps": steps,
        "seed": seed,
        "distribution": distribution,
        "state": state,
        "distance_threshold": distance_threshold,
        "use_audit": use_audit,
    }
    casters = {
        "universe": str,
        "symbols": str,
        "start": str,
        "end": str,
        "interval": str,
        "target": str,
        "gap_min": float,
        "volume_z_min": float,
        "horizon": int,
        "stock_strategy": str,
        "option_strategy": str,
        "option_type": str,
        "strike": float,
        "maturity_days": int,
        "iv": float,
        "rfr": float,
        "mode": str,
        "paths": int,
        "steps": int,
        "seed": int,
        "distribution": str,
        "state": str,
        "distance_threshold": float,
        "use_audit": lambda v: str(v).lower() in {"1", "true", "yes", "on"},
    }
    merged = load_config_with_precedence(
        config_path=config,
        env_prefix="QSE_",
        cli_values=defaults,
        defaults=defaults,
        casters=casters,
    )
    universe = merged["universe"]
    symbols = merged["symbols"]
    start = merged["start"]
    end = merged["end"]
    interval = merged["interval"]
    target = Path(merged["target"])
    gap_min = merged["gap_min"]
    volume_z_min = merged["volume_z_min"]
    horizon = merged["horizon"]
    stock_strategy = merged["stock_strategy"]
    option_strategy = merged["option_strategy"]
    option_type = merged["option_type"]
    strike = merged["strike"]
    maturity_days = merged["maturity_days"]
    iv = merged["iv"]
    rfr = merged["rfr"]
    mode = merged["mode"]
    paths = merged["paths"]
    steps = merged["steps"]
    seed = merged["seed"]
    distribution = merged["distribution"]
    state = merged["state"]
    distance_threshold = merged["distance_threshold"]
    use_audit = merged["use_audit"]
    validate_screen_inputs(horizon=horizon, max_workers=1)
    valid_intervals = {"1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"}
    if interval not in valid_intervals:
        raise typer.Exit(code=1)

    from qse.models.options import OptionSpec

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

    def _parse_state_features(raw: str) -> dict[str, float]:
        if not raw:
            return {}
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise typer.Exit(code=1)
        features: dict[str, float] = {}
        for k, v in payload.items():
            try:
                features[k] = float(v)
            except (TypeError, ValueError):
                log.error("state feature must be numeric", extra={"feature": k})
                raise typer.Exit(code=1)
        return features

    state_features = _parse_state_features(state)
    if mode not in {"backtest", "monte_carlo"}:
        log.error("invalid mode", extra={"mode": mode})
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
            df = load_or_fetch(sym, start=start, end=end, interval=interval, target=target)
            df = df.sort_values("date")
            grouped[sym] = df.set_index("date")

    if not grouped:
        if not universe and not symbols:
            log.error("must provide either --universe (CSV file) or --symbols (ticker list)")
            raise typer.Exit(code=1)
        log.error("no symbols with data available")
        raise typer.Exit(code=2)

    selector = GapVolumeSelector(gap_min=gap_min, volume_z_min=volume_z_min, horizon=horizon)

    outputs = []
    for sym, df in grouped.items():
        enriched = enrich_ohlcv(df, log_output=False)
        episodes = selector.select(enriched)

        end_idx = enriched.index.max() if hasattr(enriched.index, "max") else None
        if isinstance(end_idx, pd.Timestamp):
            end_str = end_idx.isoformat()
        else:
            end_str = str(end_idx) if end_idx is not None else None

        if mode == "monte_carlo":
            result_mc = run_conditional_mc(
                df=enriched,
                episodes=episodes,
                paths=paths,
                steps=steps,
                seed=seed,
                distribution=distribution,
                stock_strategy=stock_strategy,
                option_strategy=option_strategy,
                option_spec=option_spec,
                state_features=state_features,
                distance_threshold=distance_threshold,
                use_audit=use_audit,
                symbol=sym,
                lookback_days=len(enriched),
                end_date=end_str,
                data_source=f"{('csv' if universe else 'symbols')}:{interval}",
            )
            outputs.append(
                {
                    "symbol": sym,
                    "mode": mode,
                    "episode_count": result_mc.selection.episode_count,
                    "method": result_mc.selection.method,
                    "fallback_reason": result_mc.selection.fallback_reason,
                    "state_features": state_features,
                    "metrics": result_mc.metrics.to_formatted_dict(),
                }
            )
        else:
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
    log.info("conditional command completed", extra={"symbols": list(grouped.keys()), "mode": mode})
