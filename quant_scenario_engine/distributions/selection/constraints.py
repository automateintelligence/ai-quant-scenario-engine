"""Constraint validation for model selection (US6a AS6, T166)."""

from __future__ import annotations

def meets_constraints(heavy_tailed: bool, var_pass: bool) -> bool:
    return bool(heavy_tailed and var_pass)


__all__ = ["meets_constraints"]
