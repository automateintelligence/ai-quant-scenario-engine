"""Compare simulated vs historical metrics (US6a AS5, T162)."""

from __future__ import annotations

def compare_metrics(sim_metrics: dict, hist_metrics: dict) -> dict:
    """Return simple deltas between simulated and historical metrics."""
    deltas = {}
    for key, val in sim_metrics.items():
        if isinstance(val, dict):
            sub = {}
            for sk, sv in val.items():
                hv = hist_metrics.get(key, {}).get(sk, 0.0)
                sub[sk] = sv - hv
            deltas[key] = sub
        else:
            hv = hist_metrics.get(key, 0.0)
            deltas[key] = val - hv
    return deltas


__all__ = ["compare_metrics"]
