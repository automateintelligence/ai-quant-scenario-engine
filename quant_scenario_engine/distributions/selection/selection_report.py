"""Selection report generator (US6a AS6, T168)."""

from __future__ import annotations

from typing import Dict, List, Optional


def build_selection_report(scores: List[Dict], chosen: Optional[str]) -> dict:
    return {
        "scores": scores,
        "chosen_model": chosen,
    }


__all__ = ["build_selection_report"]
