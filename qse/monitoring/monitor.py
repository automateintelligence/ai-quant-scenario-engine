from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping

import numpy as np

from qse.distributions.path_generator import generate_price_paths_from_regime
from qse.distributions.regime_loader import RegimeParams, load_regime_params
from qse.exceptions import PricingError, SchemaError
from qse.monitoring.position import AlertConfig, PositionLeg, PositionSnapshot
from qse.pricing.black_scholes import BlackScholesPricer


@dataclass
class AlertResult:
    triggered: bool
    reason: str | None = None
    pnl: float | None = None


class PositionMonitor:
    """Live monitoring loop for repricing and alerting (FR-064 to FR-069)."""

    def __init__(
        self,
        data_source: Any,
        *,
        pricer: Any | None = None,
        regimes_config: Mapping[str, Any] | None = None,
        mc_config: Mapping[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.data_source = data_source
        self.pricer = pricer or BlackScholesPricer()
        self.regimes_config = dict(regimes_config or {})
        self.mc_config = dict(mc_config or {})
        self.log = logger or logging.getLogger(__name__)

    def monitor(
        self,
        position: PositionSnapshot,
        *,
        interval_seconds: int = 300,
        iterations: int | None = None,
    ) -> Iterable[dict[str, Any]]:
        """
        Run monitoring loop yielding evaluation results.

        iterations: limit loop for tests; None means run indefinitely until alert triggers.
        """

        counter = 0
        while iterations is None or counter < iterations:
            result = self.evaluate_once(position)
            yield result
            counter += 1
            if result["alert"]["triggered"]:
                break
            time.sleep(interval_seconds)

    def evaluate_once(
        self, position: PositionSnapshot, *, now: datetime | None = None
    ) -> dict[str, Any]:
        current_time = now or datetime.utcnow()
        underlying_mark = self._fetch_underlying_price(position.underlying)
        leg_marks, mark_pnl = self._reprice_legs(position, current_time)
        remaining_horizon = position.remaining_horizon(current_time)
        simulation = self._simulate_remaining_paths(position, remaining_horizon, underlying_mark, current_time)
        alert = self._check_alerts(position.alerts, mark_pnl)

        return {
            "underlying": position.underlying,
            "mark_pnl": mark_pnl,
            "remaining_horizon": remaining_horizon,
            "leg_marks": leg_marks,
            "simulation": simulation,
            "alert": {
                "triggered": alert.triggered,
                "reason": alert.reason,
                "pnl": alert.pnl,
            },
            "as_of": current_time.isoformat(),
        }

    def _reprice_legs(
        self, position: PositionSnapshot, current_time: datetime
    ) -> tuple[list[dict[str, Any]], float]:
        leg_marks: list[dict[str, Any]] = []
        total_pnl = 0.0
        for leg in position.legs:
            mark = self._fetch_leg_mark(position.underlying, leg)
            leg_pnl = leg.direction * (mark - leg.entry_price) * abs(leg.quantity)
            leg_marks.append(
                {
                    "option_type": leg.option_type,
                    "side": leg.side,
                    "strike": leg.strike,
                    "expiry": leg.expiry.isoformat(),
                    "mark": mark,
                    "pnl": leg_pnl,
                }
            )
            total_pnl += leg_pnl
        self.log.debug("Repriced %s legs with total PnL=%s", len(leg_marks), total_pnl)
        return leg_marks, total_pnl

    def _simulate_remaining_paths(
        self,
        position: PositionSnapshot,
        remaining_horizon: int,
        underlying_mark: float,
        current_time: datetime,
    ) -> dict[str, Any]:
        if not self.regimes_config:
            return {"paths": 0, "mean_pnl": 0.0, "pop": None}

        try:
            params: RegimeParams = load_regime_params(
                position.regime,
                self.regimes_config,
                mode=self.mc_config.get("mode", "table"),
                overrides=self.mc_config.get("regime_overrides"),
            )
        except Exception as exc:  # pragma: no cover - defensive; validation happens earlier
            self.log.warning("Falling back to zero paths due to regime load error: %s", exc)
            return {"paths": 0, "mean_pnl": 0.0, "pop": None}

        num_paths = int(self.mc_config.get("num_paths", 250))
        bars_per_day = int(self.mc_config.get("bars_per_day", 1))
        paths = generate_price_paths_from_regime(
            s0=underlying_mark,
            regime=params,
            trade_horizon=remaining_horizon,
            bars_per_day=bars_per_day,
            n_paths=num_paths,
            seed=self.mc_config.get("seed"),
        )

        pnls: list[float] = []
        for path in paths:
            path_pnl = 0.0
            for leg in position.legs:
                spec = leg.to_option_spec(current_time, remaining_horizon)
                try:
                    price_series = self.pricer.price(path, spec)
                    leg_terminal = float(price_series[-1])
                except PricingError:
                    # Skip leg contribution on pricing error but log
                    self.log.warning("Pricing failed for leg %s", leg)
                    continue
                leg_entry_value = leg.entry_price * abs(leg.quantity)
                path_pnl += leg.direction * (leg_terminal - leg_entry_value)
            pnls.append(path_pnl)

        if not pnls:
            return {"paths": 0, "mean_pnl": 0.0, "pop": None}

        pnl_array = np.asarray(pnls, dtype=float)
        mean_pnl = float(np.mean(pnl_array))
        pop = float(np.mean(pnl_array >= 0.0))
        return {"paths": len(pnls), "mean_pnl": mean_pnl, "pop": pop}

    def _check_alerts(self, alerts: AlertConfig | None, pnl: float) -> AlertResult:
        if alerts is None or not alerts.enabled:
            return AlertResult(triggered=False)

        if alerts.profit_target is not None and pnl >= alerts.profit_target:
            return AlertResult(triggered=True, reason="profit_target", pnl=pnl)
        if alerts.stop_loss is not None and pnl <= alerts.stop_loss:
            return AlertResult(triggered=True, reason="stop_loss", pnl=pnl)
        return AlertResult(triggered=False, pnl=pnl)

    def _fetch_underlying_price(self, symbol: str) -> float:
        if hasattr(self.data_source, "get_underlying_price"):
            return float(self.data_source.get_underlying_price(symbol))
        if hasattr(self.data_source, "fetch_quotes"):
            quotes = self.data_source.fetch_quotes([symbol])
            return float(self._mid_from_quote(quotes.get(symbol, {})))
        raise SchemaError("data_source must provide get_underlying_price or fetch_quotes")

    def _fetch_leg_mark(self, underlying: str, leg: PositionLeg) -> float:
        if hasattr(self.data_source, "get_option_mark"):
            return float(
                self.data_source.get_option_mark(
                    underlying=underlying,
                    expiry=leg.expiry,
                    strike=leg.strike,
                    option_type=leg.option_type,
                    option_symbol=leg.option_symbol,
                )
            )

        symbol_key = leg.option_symbol
        if symbol_key and hasattr(self.data_source, "fetch_quotes"):
            quotes = self.data_source.fetch_quotes([symbol_key])
            return float(self._mid_from_quote(quotes.get(symbol_key, {})))

        if hasattr(self.data_source, "fetch_quotes"):
            option_key = f"{underlying}:{leg.option_type}:{leg.strike}:{leg.expiry.isoformat()}"
            quotes = self.data_source.fetch_quotes([option_key])
            return float(self._mid_from_quote(quotes.get(option_key, {})))

        raise SchemaError("data_source cannot provide option marks for monitoring")

    @staticmethod
    def _mid_from_quote(quote: Mapping[str, Any]) -> float:
        if not quote:
            raise SchemaError("Quote payload missing for requested symbol")
        bid = quote.get("bid")
        ask = quote.get("ask")
        if bid is not None and ask is not None:
            return (float(bid) + float(ask)) / 2.0
        for key in ("mark", "last", "close", "price"):
            if key in quote:
                return float(quote[key])
        raise SchemaError("Quote did not include price fields (bid/ask/mark/last)")
