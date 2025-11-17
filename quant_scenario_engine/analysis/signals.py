"""Signal analysis and position tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PositionChange:
    """Represents a position change event."""

    step: int
    path: int
    position_before: int
    position_after: int
    price: float
    change_type: str  # "entry", "exit", "increase", "decrease"


def analyze_signals(
    signals: np.ndarray,
    price_paths: np.ndarray,
    signal_type: str = "stock",
) -> dict[str, Any]:
    """
    Analyze trading signals to extract position statistics.

    Args:
        signals: Signal array [n_paths, n_steps], positive=long, negative=short, 0=flat
        price_paths: Price array [n_paths, n_steps]
        signal_type: "stock" or "option"

    Returns:
        Dictionary with analysis results
    """
    n_paths, n_steps = signals.shape

    # Count position changes
    changes = np.diff(signals, axis=1, prepend=0)
    n_changes = np.count_nonzero(changes, axis=1)

    # Count steps in position vs flat
    in_position = np.abs(signals) > 0
    steps_in_position = np.sum(in_position, axis=1)
    steps_flat = n_steps - steps_in_position

    # Average position size (when not flat)
    avg_position = np.zeros(n_paths)
    for i in range(n_paths):
        active_signals = signals[i, in_position[i]]
        if len(active_signals) > 0:
            avg_position[i] = np.mean(np.abs(active_signals))

    return {
        "n_paths": n_paths,
        "n_steps": n_steps,
        "total_changes": int(np.sum(n_changes)),
        "mean_changes_per_path": float(np.mean(n_changes)),
        "median_changes_per_path": float(np.median(n_changes)),
        "mean_steps_in_position": float(np.mean(steps_in_position)),
        "mean_steps_flat": float(np.mean(steps_flat)),
        "pct_time_in_position": float(np.mean(steps_in_position) / n_steps * 100),
        "mean_position_size": float(np.mean(avg_position[avg_position > 0])),
        "max_position_size": int(np.max(np.abs(signals))),
        "signal_type": signal_type,
    }


def extract_position_changes(
    signals: np.ndarray,
    price_paths: np.ndarray,
    path_indices: list[int] | None = None,
) -> list[PositionChange]:
    """
    Extract all position change events for specified paths.

    Args:
        signals: Signal array [n_paths, n_steps]
        price_paths: Price array [n_paths, n_steps]
        path_indices: Paths to analyze (default: [0] - first path only)

    Returns:
        List of PositionChange objects
    """
    if path_indices is None:
        path_indices = [0]

    changes = []

    for path_idx in path_indices:
        if path_idx >= signals.shape[0]:
            continue

        path_signals = signals[path_idx]
        path_prices = price_paths[path_idx]

        for step in range(1, len(path_signals)):
            pos_before = int(path_signals[step - 1])
            pos_after = int(path_signals[step])

            if pos_before != pos_after:
                # Determine change type
                if pos_before == 0 and pos_after != 0:
                    change_type = "entry"
                elif pos_before != 0 and pos_after == 0:
                    change_type = "exit"
                elif abs(pos_after) > abs(pos_before):
                    change_type = "increase"
                else:
                    change_type = "decrease"

                changes.append(
                    PositionChange(
                        step=step,
                        path=path_idx,
                        position_before=pos_before,
                        position_after=pos_after,
                        price=float(path_prices[step]),
                        change_type=change_type,
                    )
                )

    return changes


def position_changes_to_dataframe(changes: list[PositionChange]) -> pd.DataFrame:
    """Convert position changes to pandas DataFrame."""
    if not changes:
        return pd.DataFrame(
            columns=["path", "step", "change_type", "position_before", "position_after", "price"]
        )

    return pd.DataFrame(
        [
            {
                "path": c.path,
                "step": c.step,
                "change_type": c.change_type,
                "position_before": c.position_before,
                "position_after": c.position_after,
                "price": c.price,
            }
            for c in changes
        ]
    )


def generate_signal_summary(
    stock_signals: np.ndarray,
    option_signals: np.ndarray,
    price_paths: np.ndarray,
) -> str:
    """
    Generate human-readable summary of strategy signals.

    Args:
        stock_signals: Stock signal array [n_paths, n_steps]
        option_signals: Option signal array [n_paths, n_steps]
        price_paths: Price array [n_paths, n_steps]

    Returns:
        Formatted string summary
    """
    stock_analysis = analyze_signals(stock_signals, price_paths, "stock")
    option_analysis = analyze_signals(option_signals, price_paths, "option")

    summary = []
    summary.append("=" * 80)
    summary.append("SIGNAL ANALYSIS SUMMARY")
    summary.append("=" * 80)
    summary.append("")

    # Stock signals
    summary.append("STOCK STRATEGY:")
    summary.append(f"  Total position changes: {stock_analysis['total_changes']:,}")
    summary.append(
        f"  Average changes per path: {stock_analysis['mean_changes_per_path']:.1f}"
    )
    summary.append(
        f"  Time in position: {stock_analysis['pct_time_in_position']:.1f}% "
        f"({stock_analysis['mean_steps_in_position']:.0f}/{stock_analysis['n_steps']} steps)"
    )
    summary.append(
        f"  Average position size: {stock_analysis['mean_position_size']:.1f} shares"
    )
    summary.append(f"  Max position size: {stock_analysis['max_position_size']:,} shares")
    summary.append("")

    # Option signals
    summary.append("OPTION STRATEGY:")
    summary.append(f"  Total position changes: {option_analysis['total_changes']:,}")
    summary.append(
        f"  Average changes per path: {option_analysis['mean_changes_per_path']:.1f}"
    )
    summary.append(
        f"  Time in position: {option_analysis['pct_time_in_position']:.1f}% "
        f"({option_analysis['mean_steps_in_position']:.0f}/{option_analysis['n_steps']} steps)"
    )
    summary.append(
        f"  Average position size: {option_analysis['mean_position_size']:.1f} contracts"
    )
    summary.append(
        f"  Max position size: {option_analysis['max_position_size']:,} contracts"
    )
    summary.append("")

    summary.append("=" * 80)

    return "\n".join(summary)


def print_position_history(
    signals: np.ndarray,
    price_paths: np.ndarray,
    path_idx: int = 0,
    signal_type: str = "stock",
    max_rows: int = 50,
) -> None:
    """
    Print position history for a single path.

    Args:
        signals: Signal array [n_paths, n_steps]
        price_paths: Price array [n_paths, n_steps]
        path_idx: Path index to display
        signal_type: "stock" or "option"
        max_rows: Maximum rows to display
    """
    changes = extract_position_changes(signals, price_paths, [path_idx])

    if not changes:
        print(f"No position changes for path {path_idx}")
        return

    df = position_changes_to_dataframe(changes)

    unit = "shares" if signal_type == "stock" else "contracts"

    print(f"\nPOSITION HISTORY - Path {path_idx} ({signal_type})")
    print("=" * 80)
    print(
        f"{'Step':<6} {'Type':<10} {'Before':<12} {'After':<12} {'Price':<10} {'Change':<10}"
    )
    print("-" * 80)

    for idx, row in df.head(max_rows).iterrows():
        change_amount = row["position_after"] - row["position_before"]
        print(
            f"{row['step']:<6} {row['change_type']:<10} "
            f"{row['position_before']:>10} {unit:<12} "
            f"{row['position_after']:>10} {unit:<12} "
            f"${row['price']:>8.2f} "
            f"{change_amount:>+10} {unit}"
        )

    if len(df) > max_rows:
        print(f"... ({len(df) - max_rows} more changes)")

    print("=" * 80)


__all__ = [
    "analyze_signals",
    "extract_position_changes",
    "position_changes_to_dataframe",
    "generate_signal_summary",
    "print_position_history",
    "PositionChange",
]
