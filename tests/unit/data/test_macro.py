import pandas as pd

from qse.data.macro import align_macro_series


def test_align_macro_series_forward_fill_with_limit():
    source = pd.Series([1, 2], index=pd.to_datetime(["2023-01-01", "2023-01-03"]))
    target_index = pd.date_range("2023-01-01", periods=5, freq="D")

    aligned = align_macro_series(source, target_index, max_gap_multiple=2)

    assert aligned.isna().sum() == 0
    assert aligned.iloc[0] == 1
    # Forward fill through gap within tolerance
    assert aligned.iloc[1] == 1
    assert aligned.iloc[2] == 2


def test_align_macro_series_linear_interpolation():
    source = pd.Series([1, 3], index=pd.to_datetime(["2023-01-01", "2023-01-05"]))
    target_index = pd.date_range("2023-01-01", periods=5, freq="D")

    aligned = align_macro_series(source, target_index, method="linear", max_gap_multiple=4)

    assert aligned.iloc[2] == 2  # linear midpoint
