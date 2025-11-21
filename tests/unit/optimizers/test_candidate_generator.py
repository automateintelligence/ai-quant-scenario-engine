from __future__ import annotations

from datetime import datetime

import pandas as pd

from qse.optimizers.candidate_generator import CandidateGenerator, GeneratorConfig


def _sample_chain() -> pd.DataFrame:
    data = [
        {"expiry": datetime(2025, 1, 20), "strike": 95, "option_type": "put", "mid": 1.0},
        {"expiry": datetime(2025, 1, 20), "strike": 100, "option_type": "put", "mid": 1.5},
        {"expiry": datetime(2025, 1, 20), "strike": 100, "option_type": "call", "mid": 1.6},
        {"expiry": datetime(2025, 1, 20), "strike": 105, "option_type": "call", "mid": 1.1},
        {"expiry": datetime(2025, 1, 20), "strike": 110, "option_type": "call", "mid": 0.9},
    ]
    return pd.DataFrame(data)


def test_generate_verticals_respects_width_range():
    chain = _sample_chain()
    generator = CandidateGenerator(GeneratorConfig(min_width=5, max_width=10))

    candidates = generator.generate(chain, spot=100.0)

    verticals = [c for c in candidates if c.structure_type == "vertical"]
    assert len(verticals) == 4  # two call spreads, two put spreads
    assert all(5.0 <= c.width <= 10.0 for c in verticals)


def test_generate_includes_multi_leg_structures():
    chain = _sample_chain()
    generator = CandidateGenerator()

    candidates = generator.generate(chain, spot=101.0)
    types = {c.structure_type for c in candidates}

    assert "iron_condor" in types
    assert "straddle" in types
    assert "strangle" in types


def test_generate_handles_empty_chain():
    generator = CandidateGenerator()

    assert generator.generate(pd.DataFrame(), spot=100.0) == []
