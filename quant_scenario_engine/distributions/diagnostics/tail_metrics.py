"""Tail error metrics for distribution audit (US6a AS3)."""

from __future__ import annotations

import numpy as np


def tail_error(empirical: np.ndarray, model_samples: np.ndarray, levels=(0.95, 0.99, 0.995)) -> dict:
    """Compute relative tail errors at specified upper-tail VaR levels."""
    errors = {}
    for lvl in levels:
        emp_q = float(np.quantile(empirical, 1.0 - lvl))
        mod_q = float(np.quantile(model_samples, 1.0 - lvl))
        if emp_q == 0:
            rel = 0.0
        else:
            rel = abs(mod_q - emp_q) / (abs(emp_q) + 1e-12)
        errors[f"var_{int(lvl*1000)/10:.1f}"] = {"empirical": emp_q, "model": mod_q, "relative_error": rel}
    return errors


__all__ = ["tail_error"]
