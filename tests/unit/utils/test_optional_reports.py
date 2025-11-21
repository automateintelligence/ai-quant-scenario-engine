import numpy as np
import pandas as pd
import pytest

from qse.exceptions import DependencyError
from qse.utils.plots import plot_equity_curves
from qse.utils.quantstats_report import generate_quantstats_report


def test_plot_equity_curves_dependency(monkeypatch, tmp_path):
    monkeypatch.setitem(__import__('sys').modules, 'plotly', None)
    with pytest.raises(DependencyError):
        plot_equity_curves({"a": np.array([1, 2, 3])}, tmp_path / "plot.html")


def test_quantstats_dependency(monkeypatch, tmp_path):
    monkeypatch.setitem(__import__('sys').modules, 'quantstats', None)
    with pytest.raises(DependencyError):
        generate_quantstats_report(pd.Series([1, 2, 3]), tmp_path / "report.html")
