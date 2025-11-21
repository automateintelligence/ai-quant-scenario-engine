import pandas as pd

from qse.data.versioning import DataVersion, compute_version, detect_drift


def test_detect_drift_schema_change():
    df1 = pd.DataFrame({"close": [1, 2, 3]})
    df2 = pd.DataFrame({"close": [1, 2, 3], "volume": [1, 1, 1]})
    old = compute_version(df1)
    new = compute_version(df2)
    assert detect_drift(old, new) == "schema"


def test_detect_drift_distribution_change():
    old = DataVersion(schema_hash="a", row_count=100, mean_return=0.01, std_return=0.02)
    new = DataVersion(schema_hash="a", row_count=100, mean_return=0.5, std_return=0.02)
    assert detect_drift(old, new, stat_threshold=0.1) == "distribution"
