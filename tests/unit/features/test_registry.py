import pandas as pd

from qse.features.registry import IndicatorRegistry, apply_indicators
from qse.schema.indicators import IndicatorDefinition


def test_apply_indicators_adds_columns():
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5], "volume": [100, 110, 120, 130, 140]})
    defs = (
        IndicatorDefinition(name="rsi_14", function="rsi", source="close", params={"length": 3}),
        IndicatorDefinition(name="vol_z", function="volume_z", source="volume", params={"window": 3}),
    )

    out, added, missing = apply_indicators(df, defs, registry=IndicatorRegistry())

    assert "rsi_14" in out.columns and "vol_z" in out.columns
    assert "rsi_14" in added and "vol_z" in added
    assert not missing


def test_apply_indicators_warns_on_missing_source():
    df = pd.DataFrame({"close": [1, 2, 3]})
    defs = (IndicatorDefinition(name="vol_z", function="volume_z", source="volume"),)

    out, added, missing = apply_indicators(df, defs, registry=IndicatorRegistry())

    assert "vol_z" not in out.columns
    assert "vol_z" in missing
    assert not added
