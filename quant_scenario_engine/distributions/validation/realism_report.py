"""Simulation realism report generator (US6a AS5, T163)."""

from __future__ import annotations

from quant_scenario_engine.distributions.validation.distribution_compare import compare_metrics


def build_realism_report(sim_metrics: dict, hist_metrics: dict) -> dict:
    deltas = compare_metrics(sim_metrics, hist_metrics)
    return {"simulated": sim_metrics, "historical": hist_metrics, "deltas": deltas}


__all__ = ["build_realism_report"]
