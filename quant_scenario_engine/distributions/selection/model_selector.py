"""Model selector using composite scores and constraints (US6a AS6, T167)."""

from __future__ import annotations

from typing import Dict, List, Optional

from quant_scenario_engine.distributions.selection.constraints import meets_constraints


def select_model(scores: List[Dict], constraints: Dict[str, Dict]) -> Optional[str]:
    """
    scores: list of {"model": name, "score": float}
    constraints: {"model": {"heavy_tailed": bool, "var_pass": bool}}
    """
    eligible = []
    for entry in scores:
        name = entry["model"]
        c = constraints.get(name, {})
        if meets_constraints(c.get("heavy_tailed", False), c.get("var_pass", False)):
            eligible.append(entry)
    if not eligible:
        return None
    eligible_sorted = sorted(eligible, key=lambda x: x["score"], reverse=True)
    return eligible_sorted[0]["model"]


__all__ = ["select_model"]
