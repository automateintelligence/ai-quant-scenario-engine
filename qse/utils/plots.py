"""Optional Plotly plotting utilities (Phase 11 T123)."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from qse.exceptions import DependencyError


def plot_equity_curves(equity_curves: Mapping[str, np.ndarray], output_path: Path) -> Path:
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise DependencyError("plotly is required for plot generation") from exc

    fig = go.Figure()
    for name, series in equity_curves.items():
        fig.add_trace(go.Scatter(y=np.asarray(series, dtype=float), mode="lines", name=name))

    fig.update_layout(title="Equity Curves", xaxis_title="Step", yaxis_title="P&L")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    return output_path
