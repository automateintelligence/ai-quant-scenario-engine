import pytest

from qse.cli.commands.grid import _parse_objective_weights
from qse.cli.validation import validate_grid_inputs
from qse.exceptions import ConfigValidationError


def test_validate_grid_inputs_requires_grid():
    with pytest.raises(ConfigValidationError):
        validate_grid_inputs(paths=10, steps=5, seed=1, grid=None)


def test_parse_objective_weights_from_json_string():
    weights = _parse_objective_weights('{"mean_pnl": 0.4, "sharpe": 0.2}')
    assert weights.mean_pnl == pytest.approx(0.4)
    assert weights.sharpe == pytest.approx(0.2)


def test_parse_objective_weights_raises_on_invalid():
    with pytest.raises(ConfigValidationError):
        _parse_objective_weights(123)
