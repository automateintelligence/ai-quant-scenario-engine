"""Lightweight performance budget checks.

This script provides sanity checks for data load time, distribution fit, Monte Carlo throughput,
and strategy evaluation speed. It is intentionally simple and avoids external fixtures so it can
run in constrained CI environments.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

# Placeholders for actual implementations that will be wired as the engine matures.

def load_synthetic_ohlcv(n: int = 252) -> np.ndarray:
    """Generate synthetic OHLCV close prices for timing purposes."""
    base = 100 + np.cumsum(np.random.normal(0, 1, size=n))
    return base


def benchmark_data_load() -> float:
    start = time.perf_counter()
    _ = load_synthetic_ohlcv()
    return (time.perf_counter() - start) * 1000


def benchmark_distribution_fit() -> float:
    returns = np.random.normal(0, 0.01, size=1_000)
    start = time.perf_counter()
    # TODO: replace with Laplace/StudentT/GARCH fits
    _ = returns.mean(), returns.std(ddof=1)
    return (time.perf_counter() - start) * 1000


def benchmark_mc_throughput(n_paths: int = 1_000, n_steps: int = 60) -> float:
    start = time.perf_counter()
    _ = np.random.normal(0, 0.01, size=(n_paths, n_steps)).cumsum(axis=1)
    return (time.perf_counter() - start) * 1000


def benchmark_strategy_eval(n_paths: int = 1_000, n_steps: int = 60) -> float:
    paths = np.random.normal(0, 0.01, size=(n_paths, n_steps)).cumsum(axis=1)
    start = time.perf_counter()
    signals = (paths > np.expand_dims(paths[:, :1], axis=1)).astype(np.int8)
    _ = (signals * paths).mean()
    return (time.perf_counter() - start) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="Performance budget checks")
    parser.add_argument("--out", type=Path, default=None, help="Optional file to write timings (JSON)")
    args = parser.parse_args()

    metrics = {
        "data_load_ms": benchmark_data_load(),
        "distribution_fit_ms": benchmark_distribution_fit(),
        "mc_throughput_ms": benchmark_mc_throughput(),
        "strategy_eval_ms": benchmark_strategy_eval(),
    }

    for key, value in metrics.items():
        print(f"{key}: {value:.2f} ms")

    if args.out:
        import json

        args.out.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
