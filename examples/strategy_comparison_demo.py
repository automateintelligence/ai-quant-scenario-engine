"""Demonstration of all canonical strategies with feature computation.

This script demonstrates:
1. All six strategies (stock_basic, stock_sma_trend, stock_rsi_reversion,
   call_basic, option_atm_call_momentum, option_atm_put_rsi)
2. Automatic feature computation for RSI-based strategies
3. Position sizing targeting $500-$1,000 daily P&L
4. Comparing stock vs option strategies head-to-head
"""

from __future__ import annotations

import numpy as np

from quant_scenario_engine.distributions.factory import get_distribution
from quant_scenario_engine.models.options import OptionSpec
from quant_scenario_engine.simulation.compare import run_compare


def run_strategy_comparison():
    """Run comprehensive strategy comparison."""

    # Configuration
    config = {
        "s0": 100.0,
        "n_paths": 200,
        "n_steps": 60,
        "seed": 42,
    }

    # Fit distribution
    dist = get_distribution("laplace")
    # Use synthetic heavy-tailed data to ensure fit succeeds
    synthetic_returns = np.concatenate([
        np.random.laplace(0, 0.02, size=400),
        np.random.laplace(0, 0.05, size=100),  # Add some extreme values
    ])
    dist.fit(synthetic_returns)

    # Option specifications
    call_spec = OptionSpec(
        option_type="call",
        strike=100.0,
        maturity_days=30,
        implied_vol=0.25,
        risk_free_rate=0.01,
        contracts=1,
    )

    put_spec = OptionSpec(
        option_type="put",
        strike=100.0,
        maturity_days=21,
        implied_vol=0.25,
        risk_free_rate=0.01,
        contracts=1,
    )
    
    # Strategy combinations to test
    test_cases = [
        {
            "name": "Basic Placeholders (No Features)",
            "stock": "stock_basic",
            "option": "call_basic",
            "spec": call_spec,
            "compute_features": False,
        },
        {
            "name": "SMA Trend vs ATM Call Momentum",
            "stock": "stock_sma_trend",
            "option": "option_atm_call_momentum",
            "spec": call_spec,
            "compute_features": True,
        },
        {
            "name": "RSI Reversion (Stock) vs Basic Call",
            "stock": "stock_rsi_reversion",
            "option": "call_basic",
            "spec": call_spec,
            "compute_features": True,  # RSI required for stock strategy
        },
        {
            "name": "Basic Stock vs ATM Put RSI",
            "stock": "stock_basic",
            "option": "option_atm_put_rsi",
            "spec": put_spec,
            "compute_features": True,  # RSI required for option strategy
        },
        {
            "name": "RSI Reversion vs ATM Put RSI (Both RSI)",
            "stock": "stock_rsi_reversion",
            "option": "option_atm_put_rsi",
            "spec": put_spec,
            "compute_features": True,  # RSI required for both strategies
        },
    ]

    print("=" * 80)
    print("STRATEGY COMPARISON DEMO")
    print("=" * 80)
    print(f"Configuration: {config['n_paths']} paths, {config['n_steps']} steps, seed={config['seed']}")
    print(f"Initial Price: ${config['s0']:.2f}")
    print("=" * 80)
    print()

    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"{i}. {test['name']}")
        print(f"   Stock Strategy: {test['stock']}")
        print(f"   Option Strategy: {test['option']}")
        print(f"   Features: {'Auto-computed' if test['compute_features'] else 'None'}")

        try:
            result = run_compare(
                **config,
                distribution=dist,
                stock_strategy=test["stock"],
                option_strategy=test["option"],
                option_spec=test["spec"],
                compute_features=test["compute_features"],
            )

            metrics = result.metrics
            print(f"   ✓ Mean P&L: ${metrics.mean_pnl:,.2f}")
            print(f"     Median P&L: ${metrics.median_pnl:,.2f}")
            print(f"     Sharpe Ratio: {metrics.sharpe:.4f}")
            print(f"     Max Drawdown: {metrics.max_drawdown:.2%}")
            print(f"     VaR (95%): {metrics.var:.2%}")
            print()

            results.append({
                "name": test["name"],
                "mean_pnl": metrics.mean_pnl,
                "sharpe": metrics.sharpe,
                "max_dd": metrics.max_drawdown,
            })

        except Exception as e:
            print(f"   ✗ Error: {e}")
            print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        best_pnl = max(results, key=lambda x: x["mean_pnl"])
        best_sharpe = max(results, key=lambda x: x["sharpe"])
        best_dd = max(results, key=lambda x: x["max_dd"])  # Least negative

        print(f"Best Mean P&L: {best_pnl['name']} (${best_pnl['mean_pnl']:,.2f})")
        print(f"Best Sharpe: {best_sharpe['name']} ({best_sharpe['sharpe']:.4f})")
        print(f"Best Max Drawdown: {best_dd['name']} ({best_dd['max_dd']:.2%})")
    else:
        print("No successful runs.")

    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("  ✓ Automatic RSI computation for RSI-based strategies")
    print("  ✓ Position sizing targeting $500-$1,000 daily P&L")
    print("  ✓ Mix of stock and option strategies with different logic")
    print("  ✓ Placeholder vs canonical strategy implementations")
    print("  ✓ Feature-dependent vs feature-independent strategies")


if __name__ == "__main__":
    run_strategy_comparison()
