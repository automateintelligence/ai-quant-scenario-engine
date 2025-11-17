"""Example: Analyze and inspect strategy positions from a compare run.

This script demonstrates how to:
1. Run a compare simulation
2. Extract the signal history
3. Analyze position changes
4. Display entry/exit points
"""

from __future__ import annotations

import numpy as np

from quant_scenario_engine.analysis.signals import (
    analyze_signals,
    generate_signal_summary,
    position_changes_to_dataframe,
    extract_position_changes,
    print_position_history,
)
from quant_scenario_engine.distributions.factory import get_distribution
from quant_scenario_engine.models.options import OptionSpec
from quant_scenario_engine.simulation.compare import run_compare


def main():
    """Run strategy and analyze positions."""

    print("=" * 80)
    print("STRATEGY POSITION ANALYSIS")
    print("=" * 80)
    print()

    # Configuration
    config = {
        "s0": 100.0,
        "n_paths": 100,
        "n_steps": 60,
        "seed": 42,
    }

    # Fit distribution
    dist = get_distribution("laplace")
    synthetic_returns = np.concatenate([
        np.random.laplace(0, 0.02, size=400),
        np.random.laplace(0, 0.05, size=100),
    ])
    dist.fit(synthetic_returns)

    # Option spec
    call_spec = OptionSpec(
        option_type="call",
        strike=100.0,
        maturity_days=30,
        implied_vol=0.25,
        risk_free_rate=0.01,
        contracts=1,
    )

    print("Running simulation...")
    print(f"  Paths: {config['n_paths']}")
    print(f"  Steps: {config['n_steps']}")
    print(f"  Strategy: stock_sma_trend vs option_atm_call_momentum")
    print()

    # Run compare
    result = run_compare(
        **config,
        distribution=dist,
        stock_strategy="stock_sma_trend",
        option_strategy="option_atm_call_momentum",
        option_spec=call_spec,
        compute_features=True,
    )

    # Extract signals and price paths
    signals = result.signals
    stock_signals = signals.signals_stock
    option_signals = signals.signals_option

    # Generate price paths again for analysis (alternatively, store in RunResult)
    from quant_scenario_engine.mc.generator import generate_price_paths

    price_paths = generate_price_paths(
        s0=config["s0"],
        distribution=dist,
        n_paths=config["n_paths"],
        n_steps=config["n_steps"],
        seed=config["seed"],
    )

    # Print metrics
    print("METRICS:")
    print(f"  Mean P&L: ${result.metrics.mean_pnl:,.2f}")
    print(f"  Median P&L: ${result.metrics.median_pnl:,.2f}")
    print(f"  Max Drawdown: {result.metrics.max_drawdown:.2%}")
    print(f"  Sharpe Ratio: {result.metrics.sharpe:.4f}")
    print()

    # Generate signal summary
    summary = generate_signal_summary(stock_signals, option_signals, price_paths)
    print(summary)

    # Detailed analysis for first few paths
    print("\nDETAILED POSITION HISTORY")
    print("=" * 80)

    # Stock positions for path 0
    print("\n--- Path 0: STOCK ---")
    print_position_history(
        stock_signals, price_paths, path_idx=0, signal_type="stock", max_rows=20
    )

    # Option positions for path 0
    print("\n--- Path 0: OPTION ---")
    print_position_history(
        option_signals, price_paths, path_idx=0, signal_type="option", max_rows=20
    )

    # Extract changes for first 3 paths and create DataFrame
    print("\n--- First 3 Paths: All Changes ---")
    stock_changes = extract_position_changes(stock_signals, price_paths, [0, 1, 2])
    df = position_changes_to_dataframe(stock_changes)

    if not df.empty:
        print(f"\nTotal changes across 3 paths: {len(df)}")
        print("\nFirst 10 changes:")
        print(
            df[["path", "step", "change_type", "position_before", "position_after", "price"]]
            .head(10)
            .to_string(index=False)
        )

    # Summary statistics
    print("\n" + "=" * 80)
    print("TRADING ACTIVITY SUMMARY")
    print("=" * 80)

    stock_analysis = analyze_signals(stock_signals, price_paths, "stock")
    option_analysis = analyze_signals(option_signals, price_paths, "option")

    print(f"\nStock Strategy ({config['n_paths']} paths x {config['n_steps']} steps):")
    print(f"  Total trades: {stock_analysis['total_changes']:,}")
    print(f"  Avg trades/path: {stock_analysis['mean_changes_per_path']:.1f}")
    print(f"  Time in market: {stock_analysis['pct_time_in_position']:.1f}%")
    print(f"  Avg position: {stock_analysis['mean_position_size']:.0f} shares")
    print(f"  Max position: {stock_analysis['max_position_size']:,} shares")

    print(f"\nOption Strategy ({config['n_paths']} paths x {config['n_steps']} steps):")
    print(f"  Total trades: {option_analysis['total_changes']:,}")
    print(f"  Avg trades/path: {option_analysis['mean_changes_per_path']:.1f}")
    print(f"  Time in market: {option_analysis['pct_time_in_position']:.1f}%")
    print(f"  Avg position: {option_analysis['mean_position_size']:.0f} contracts")
    print(f"  Max position: {option_analysis['max_position_size']:,} contracts")

    print("\n" + "=" * 80)
    print("\nTo save signals for further analysis:")
    print("  import numpy as np")
    print("  np.savez('signals.npz', stock=stock_signals, option=option_signals, prices=price_paths)")
    print("\nTo load later:")
    print("  data = np.load('signals.npz')")
    print("  stock_signals = data['stock']")
    print("=" * 80)


if __name__ == "__main__":
    main()
