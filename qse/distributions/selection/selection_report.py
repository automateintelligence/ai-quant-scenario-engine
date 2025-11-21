"""Selection report generator (US6a AS6, T168)."""

from __future__ import annotations

from typing import Iterable, Optional


def build_selection_report(scores: Iterable[object], chosen: Optional[str]) -> dict:
    formatted = []
    for score in scores:
        if hasattr(score, "model_name"):
            formatted.append(
                {
                    "model": getattr(score, "model_name"),
                    "total_score": getattr(score, "total_score"),
                    **getattr(score, "components", {}),
                }
            )
        elif isinstance(score, dict):
            formatted.append(score)
    formatted.sort(key=lambda entry: entry.get("total_score", 0.0), reverse=True)
    return {
        "scores": formatted,
        "chosen_model": chosen,
    }


__all__ = ["build_selection_report"]
