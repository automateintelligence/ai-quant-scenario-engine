"""Stage 4 Monte Carlo scoring engine (US1/US2/T018)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np

from qse.distributions.regime_loader import RegimeParams, load_regime_params
from qse.interfaces import ReturnDistribution
from qse.optimizers.models import CandidateMetrics, CandidateStructure, Leg
from qse.pricing.black_scholes import BlackScholesPricer


@dataclass(frozen=True)
class MCConfig:
    """Configuration for Monte Carlo path generation."""

    num_paths: int = 5000
    bars_per_day: int = 1
    seed: int = 42


class MCEngine:
    """Monte Carlo scoring engine for option structures (Stage 4)."""

    def __init__(
        self,
        distribution: ReturnDistribution,
        pricer: BlackScholesPricer | None = None,
        config: MCConfig | None = None,
    ) -> None:
        """Initialize MC engine with distribution and pricer.

        Args:
            distribution: Fitted return distribution for path generation
            pricer: Option pricing model (defaults to Black-Scholes)
            config: MC configuration parameters
        """
        self.distribution = distribution
        self.pricer = pricer or BlackScholesPricer()
        self.config = config or MCConfig()

    def score_candidates(
        self,
        candidates: Sequence[CandidateStructure],
        spot: float,
        trade_horizon: int,
        regime_params: RegimeParams | None = None,
    ) -> list[CandidateStructure]:
        """Score candidates using full Monte Carlo simulation.

        Args:
            candidates: Survivors from Stage 3
            spot: Current underlying price
            trade_horizon: Holding period in trading days
            regime_params: Regime parameters for distribution (if applicable)

        Returns:
            Candidates with updated metrics including full MC scores
        """
        np.random.seed(self.config.seed)

        # Generate price paths for the trade horizon
        paths = self._generate_paths(spot, trade_horizon, regime_params)

        scored: list[CandidateStructure] = []
        for candidate in candidates:
            metrics = self._score_candidate(candidate, spot, paths, trade_horizon)
            candidate.metrics = metrics
            scored.append(candidate)

        return scored

    def _generate_paths(
        self, spot: float, trade_horizon: int, regime_params: RegimeParams | None
    ) -> np.ndarray:
        """Generate Monte Carlo price paths.

        Returns:
            Array of shape (num_paths, steps+1) with price paths
        """
        steps = trade_horizon * self.config.bars_per_day

        # Generate returns using the distribution's sample method
        # Distribution.sample(n_paths, n_steps, seed) -> (n_paths, n_steps)
        returns = self.distribution.sample(
            n_paths=self.config.num_paths,
            n_steps=steps,
            seed=self.config.seed
        )

        # If we have regime parameters, scale by regime mean/vol
        if regime_params is not None:
            # Scale by regime parameters
            returns = returns * regime_params.daily_vol + regime_params.mean_daily_return

        # Convert returns to price paths
        price_paths = np.zeros((self.config.num_paths, steps + 1))
        price_paths[:, 0] = spot

        for t in range(steps):
            price_paths[:, t + 1] = price_paths[:, t] * (1.0 + returns[:, t])

        return price_paths

    def _score_candidate(
        self,
        candidate: CandidateStructure,
        spot: float,
        paths: np.ndarray,
        trade_horizon: int,
    ) -> CandidateMetrics:
        """Score a single candidate using MC paths.

        Args:
            candidate: Candidate structure to score
            spot: Initial spot price
            paths: Price paths array (num_paths, steps+1)
            trade_horizon: Holding period in trading days

        Returns:
            Updated metrics with MC-based E[PnL], POP, etc.
        """
        num_paths = paths.shape[0]
        final_prices = paths[:, -1]  # Terminal prices

        # Calculate initial DTE (days to expiry from now)
        # Assume candidate.expiry is the option expiry date
        from datetime import datetime
        import pandas as pd

        # For MVP, assume "now" is when we enter the trade
        # In production, this would come from the as_of parameter
        now = pd.Timestamp(datetime.now())
        initial_dte = (candidate.expiry - now).days

        # Compute P&L for each path
        pnl_paths = np.zeros(num_paths)

        for i, final_price in enumerate(final_prices):
            pnl_paths[i] = self._compute_pnl(candidate, spot, final_price, trade_horizon, initial_dte)

        # Compute metrics
        expected_pnl = float(np.mean(pnl_paths))
        ci_lower = float(np.percentile(pnl_paths, 2.5))
        ci_upper = float(np.percentile(pnl_paths, 97.5))

        pop_breakeven = float(np.mean(pnl_paths >= 0))

        # Profit target (default $500 or from existing metrics)
        profit_target = 500.0
        if candidate.metrics and hasattr(candidate.metrics, 'profit_target'):
            profit_target = candidate.metrics.profit_target

        pop_target = float(np.mean(pnl_paths >= profit_target))

        # Capital and max loss
        capital = candidate.metrics.capital if candidate.metrics else 0.0
        max_loss = float(np.min(pnl_paths)) if len(pnl_paths) > 0 else 0.0

        # ROC
        roc = expected_pnl / capital if capital > 0 else 0.0

        # VaR and CVaR at 5%
        var_5 = float(np.percentile(pnl_paths, 5))
        cvar_5 = float(np.mean(pnl_paths[pnl_paths <= var_5]))

        # Composite score (simplified - will be replaced by IntradaySpreadsScorer)
        score = 0.35 * pop_breakeven + 0.30 * min(roc, 0.5) / 0.5

        return CandidateMetrics(
            expected_pnl=expected_pnl,
            pop_breakeven=pop_breakeven,
            pop_target=pop_target,
            capital=capital,
            max_loss=abs(max_loss),
            score=score,
            mc_paths=num_paths,
            entry_cash=candidate.metrics.entry_cash if candidate.metrics else None,
            expected_exit_cost=candidate.metrics.expected_exit_cost if candidate.metrics else None,
            commission=candidate.metrics.commission if candidate.metrics else None,
        )

    def _compute_pnl(
        self,
        candidate: CandidateStructure,
        initial_spot: float,
        final_spot: float,
        trade_horizon: int,
        initial_dte: int,
    ) -> float:
        """Compute P&L for a candidate at a terminal price.

        Args:
            candidate: Option structure
            initial_spot: Initial underlying price
            final_spot: Terminal underlying price
            trade_horizon: Days held
            initial_dte: Days to expiry at trade entry

        Returns:
            Net P&L for this path
        """
        from qse.models.options import OptionSpec

        # Entry value (net credit/debit)
        # For SELLING options: net_premium is positive (we receive money)
        # For BUYING options: net_premium is negative (we pay money)
        entry_pnl = candidate.net_premium * 100.0

        # Exit value - reprice each leg at terminal conditions
        # When we CLOSE positions:
        # - Closing a LONG (sell to close): We receive the option's current value (positive)
        # - Closing a SHORT (buy to close): We pay the option's current value (negative)
        exit_value = 0.0

        for leg in candidate.legs:
            # Calculate remaining time to expiry after holding for trade_horizon days
            remaining_dte = max(1, initial_dte - trade_horizon)

            # For simplicity, assume constant IV = 0.25 (will be enhanced later)
            iv = 0.25
            r = 0.01  # Risk-free rate

            # Create OptionSpec for pricer
            option_spec = OptionSpec(
                option_type=leg.option_type,
                strike=leg.strike,
                maturity_days=remaining_dte,
                implied_vol=iv,
                risk_free_rate=r,
                contracts=1,
                iv_source="config_default",
                early_exercise=False,
            )

            # Reprice using Black-Scholes (requires 1D array input)
            path_slice = np.array([final_spot])  # Shape: (1,)
            exit_price = self.pricer.price(path_slice, option_spec)[0]  # Extract scalar from prices array

            # P&L for this leg:
            # - If we're LONG (bought): P&L = exit_price - entry_price (sell for exit_price, paid entry_price)
            # - If we're SHORT (sold): P&L = entry_price - exit_price (received entry_price, pay exit_price to close)
            #
            # Since entry_pnl already accounts for the entry_price with correct sign,
            # we just need to add the exit cash flow:
            if leg.side == "buy":
                # Closing long: SELL the option (receive money)
                exit_value += exit_price * 100.0
            else:
                # Closing short: BUY the option (pay money)
                exit_value -= exit_price * 100.0

        # Net P&L = entry credit + exit value
        net_pnl = entry_pnl + exit_value

        # Apply transaction costs
        if candidate.metrics:
            net_pnl -= candidate.metrics.commission or 0.0

        return net_pnl


__all__ = ["MCEngine", "MCConfig"]
