"""Optional quantstats report generation (Phase 11 T124)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from qse.exceptions import DependencyError


def generate_quantstats_report(equity_curve: pd.Series, output_path: Path) -> Path:
    try:
        import quantstats as qs  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise DependencyError("quantstats is required for report generation") from exc

    qs.extend_pandas()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qs.reports.html(equity_curve, output=str(output_path), title="QuantStats Report")
    return output_path
