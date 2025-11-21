"""Unit tests for IntradaySpreadsScorer (US5/T029, SC-005)."""

from __future__ import annotations

import pytest

from qse.scorers.intraday_spreads import IntradaySpreadsScorer


def test_scorer_computes_score_with_default_weights():
    """Test scorer computes composite score using FR-039 default weights."""
    scorer = IntradaySpreadsScorer()

    candidate = {"structure_type": "Iron Condor", "legs": []}
    metrics = {
        "E[PnL]": 600.0,
        "POP_0": 0.72,
        "ROC": 0.04,
        "Theta": 25.0,
        "Delta": -0.02,
        "Gamma": -0.05,
        "Vega": -15.0,
        "MaxLoss": -750.0,
    }
    config = {
        "scoring": {
            # Use FR-039 defaults
            "w_pop": 0.35,
            "w_roc": 0.30,
            "w_theta": 0.10,
            "w_tail": 0.15,
            "w_delta": 0.05,
            "w_gamma": 0.03,
            "w_vega": 0.02,
        },
        "filters": {"max_capital": 15000.0, "max_loss_pct": 0.05},
    }

    score = scorer.score(candidate, metrics, config)

    # Verify score is in [0, 1] range
    assert 0.0 <= score <= 1.0

    # Verify POP normalization: POP=72% → POP_norm = (0.72-0.5)/0.5 = 0.44
    # pop_contrib = 0.35 * 0.44 = 0.154
    # Verify score has positive contribution from POP and ROC
    assert score > 0.0


def test_weight_override_changes_ranking():
    """Test that adjusting weights changes Top-10 rankings (SC-005, US5 Independent Test)."""
    scorer = IntradaySpreadsScorer()

    # Candidate A: Very High POP (85%), Low ROC (1%), Low Penalties
    candidate_a = {"structure_type": "Iron Condor", "legs": []}
    metrics_a = {
        "POP_0": 0.85,  # Very high POP
        "ROC": 0.01,  # Very low ROC
        "Theta": 30.0,  # Higher theta reward
        "Delta": 0.0,  # Neutral delta (no penalty)
        "Gamma": -0.01,  # Lower gamma penalty
        "Vega": -5.0,  # Lower vega penalty
        "MaxLoss": -300.0,  # Smaller max loss
    }

    # Candidate B: Lower POP (55%), Much Higher ROC (9%), Higher Penalties
    candidate_b = {"structure_type": "Bull Put Spread", "legs": []}
    metrics_b = {
        "POP_0": 0.55,  # Lower POP
        "ROC": 0.09,  # Very high ROC
        "Theta": 15.0,  # Lower theta reward
        "Delta": 0.03,  # Some delta penalty
        "Gamma": -0.04,  # Higher gamma penalty
        "Vega": -15.0,  # Higher vega penalty
        "MaxLoss": -600.0,  # Larger max loss
    }

    # Config with default weights: w_pop=0.35, w_roc=0.30
    config_default = {
        "scoring": {
            "w_pop": 0.35,
            "w_roc": 0.30,
            "w_theta": 0.10,
            "w_tail": 0.15,
            "w_delta": 0.05,
            "w_gamma": 0.03,
            "w_vega": 0.02,
        },
        "filters": {"max_capital": 15000.0, "max_loss_pct": 0.05},
    }

    # Config with adjusted weights: prioritize ROC over POP
    config_roc_focused = {
        "scoring": {
            "w_pop": 0.20,  # Reduced from 0.35
            "w_roc": 0.50,  # Increased from 0.30
            "w_theta": 0.10,
            "w_tail": 0.15,
            "w_delta": 0.05,
            "w_gamma": 0.00,
            "w_vega": 0.00,
        },
        "filters": {"max_capital": 15000.0, "max_loss_pct": 0.05},
    }

    # Score with default weights
    score_a_default = scorer.score(candidate_a, metrics_a, config_default)
    score_b_default = scorer.score(candidate_b, metrics_b, config_default)

    # Score with ROC-focused weights
    score_a_roc = scorer.score(candidate_a, metrics_a, config_roc_focused)
    score_b_roc = scorer.score(candidate_b, metrics_b, config_roc_focused)

    # With default weights, high-POP candidate A should rank higher
    assert score_a_default > score_b_default, (
        f"Default weights should favor high-POP candidate A: "
        f"A={score_a_default:.3f} vs B={score_b_default:.3f}"
    )

    # With ROC-focused weights, high-ROC candidate B should rank higher
    assert score_b_roc > score_a_roc, (
        f"ROC-focused weights should favor high-ROC candidate B: "
        f"B={score_b_roc:.3f} vs A={score_a_roc:.3f}"
    )


def test_score_decomposition_shows_components():
    """Test decompose() returns individual component contributions (FR-041)."""
    scorer = IntradaySpreadsScorer()

    candidate = {"structure_type": "Iron Condor", "legs": []}
    metrics = {
        "POP_0": 0.68,
        "ROC": 0.045,
        "Theta": 22.0,
        "Delta": -0.01,
        "Gamma": -0.04,
        "Vega": -12.0,
        "MaxLoss": -650.0,
    }
    config = {
        "scoring": {
            "w_pop": 0.35,
            "w_roc": 0.30,
            "w_theta": 0.10,
            "w_tail": 0.15,
            "w_delta": 0.05,
            "w_gamma": 0.03,
            "w_vega": 0.02,
        },
        "filters": {"max_capital": 15000.0, "max_loss_pct": 0.05},
    }

    decomp = scorer.decompose(candidate, metrics, config)

    # Verify all expected components are present
    assert "pop_contrib" in decomp
    assert "roc_contrib" in decomp
    assert "theta_contrib" in decomp
    assert "tail_penalty" in decomp
    assert "delta_penalty" in decomp
    assert "gamma_penalty" in decomp
    assert "vega_penalty" in decomp
    assert "composite_score" in decomp

    # Verify contributions are weighted correctly
    # POP=68% → POP_norm = (0.68-0.5)/0.5 = 0.36 → pop_contrib = 0.35 * 0.36 = 0.126
    assert decomp["pop_contrib"] > 0.0

    # Verify penalties are non-negative
    assert decomp["tail_penalty"] >= 0.0
    assert decomp["delta_penalty"] >= 0.0
    assert decomp["gamma_penalty"] >= 0.0
    assert decomp["vega_penalty"] >= 0.0

    # Verify composite score matches direct score() call
    score = scorer.score(candidate, metrics, config)
    assert abs(decomp["composite_score"] - score) < 1e-9


def test_pop_normalization():
    """Test POP normalization maps [50%, 100%] to [0, 1] (FR-036)."""
    scorer = IntradaySpreadsScorer()
    config = {"scoring": {}}

    # POP=50% (breakeven) → normalized to 0.0
    assert scorer._normalize_pop(0.50, config) == pytest.approx(0.0, abs=1e-6)

    # POP=75% (halfway between 50% and 100%) → normalized to 0.5
    assert scorer._normalize_pop(0.75, config) == pytest.approx(0.5, abs=1e-6)

    # POP=100% (certain profit) → normalized to 1.0
    assert scorer._normalize_pop(1.00, config) == pytest.approx(1.0, abs=1e-6)

    # POP<50% → clamped to 0.0
    assert scorer._normalize_pop(0.30, config) == pytest.approx(0.0, abs=1e-6)


def test_roc_normalization():
    """Test ROC normalization with configurable scaling (FR-036)."""
    scorer = IntradaySpreadsScorer()

    # Default scale: 10% ROC → score 1.0
    config_default = {"scoring": {"roc_scale": 0.10}}
    assert scorer._normalize_roc(0.05, config_default) == pytest.approx(0.5, abs=1e-6)
    assert scorer._normalize_roc(0.10, config_default) == pytest.approx(1.0, abs=1e-6)

    # ROC > scale → clamped to 1.0
    assert scorer._normalize_roc(0.15, config_default) == pytest.approx(1.0, abs=1e-6)


def test_theta_reward_positive_only():
    """Test Theta reward only applies to positive theta (income trades) (FR-037)."""
    scorer = IntradaySpreadsScorer()
    config = {"scoring": {"theta_scale": 50.0}}

    # Positive theta (income) → reward
    assert scorer._compute_theta_reward(25.0, config) == pytest.approx(0.5, abs=1e-6)
    assert scorer._compute_theta_reward(50.0, config) == pytest.approx(1.0, abs=1e-6)

    # Negative theta (cost) → no reward (zero contribution)
    assert scorer._compute_theta_reward(-25.0, config) == pytest.approx(0.0, abs=1e-6)


def test_tail_penalty_relative_to_capital():
    """Test tail penalty scales MaxLoss relative to capital and max_loss_pct (FR-038)."""
    scorer = IntradaySpreadsScorer()
    config = {"scoring": {}}
    filters = {"max_capital": 10000.0, "max_loss_pct": 0.05}

    # MaxLoss = $500 → penalty = 500 / (10000 * 0.05) = 500 / 500 = 1.0
    penalty = scorer._compute_tail_penalty(-500.0, filters)
    assert penalty == pytest.approx(1.0, abs=1e-6)

    # MaxLoss = $250 → penalty = 250 / 500 = 0.5
    penalty = scorer._compute_tail_penalty(-250.0, filters)
    assert penalty == pytest.approx(0.5, abs=1e-6)


def test_delta_penalty_deviation_from_target():
    """Test delta penalty measures deviation from target (default neutral=0) (FR-038)."""
    scorer = IntradaySpreadsScorer()

    # Default: delta_target=0, delta_scale=0.5
    config = {"scoring": {"delta_target": 0.0, "delta_scale": 0.5}}

    # Delta = 0.0 (neutral) → penalty = 0.0
    assert scorer._compute_delta_penalty(0.0, config) == pytest.approx(0.0, abs=1e-6)

    # Delta = 0.25 (bullish) → penalty = 0.25 / 0.5 = 0.5
    assert scorer._compute_delta_penalty(0.25, config) == pytest.approx(0.5, abs=1e-6)

    # Delta = -0.25 (bearish) → penalty = 0.25 / 0.5 = 0.5 (absolute value)
    assert scorer._compute_delta_penalty(-0.25, config) == pytest.approx(0.5, abs=1e-6)


def test_gamma_penalty_absolute_value():
    """Test gamma penalty uses absolute value (FR-038)."""
    scorer = IntradaySpreadsScorer()
    config = {"scoring": {"gamma_scale": 0.10}}

    # Gamma = -0.05 → penalty = 0.05 / 0.10 = 0.5
    assert scorer._compute_gamma_penalty(-0.05, config) == pytest.approx(0.5, abs=1e-6)

    # Gamma = 0.05 → penalty = 0.05 / 0.10 = 0.5 (same magnitude)
    assert scorer._compute_gamma_penalty(0.05, config) == pytest.approx(0.5, abs=1e-6)


def test_vega_penalty_absolute_value():
    """Test vega penalty uses absolute value (FR-038)."""
    scorer = IntradaySpreadsScorer()
    config = {"scoring": {"vega_scale": 50.0}}

    # Vega = -25.0 → penalty = 25.0 / 50.0 = 0.5
    assert scorer._compute_vega_penalty(-25.0, config) == pytest.approx(0.5, abs=1e-6)

    # Vega = 25.0 → penalty = 25.0 / 50.0 = 0.5 (same magnitude)
    assert scorer._compute_vega_penalty(25.0, config) == pytest.approx(0.5, abs=1e-6)


def test_score_clamped_to_zero_one_range():
    """Test composite score is always clamped to [0, 1] range."""
    scorer = IntradaySpreadsScorer()

    # Candidate with extremely high penalties → score should clamp to 0.0
    candidate = {"structure_type": "Test", "legs": []}
    metrics = {
        "POP_0": 0.30,  # Low POP → negative contribution
        "ROC": 0.0,  # No return
        "Theta": -50.0,  # Negative theta (cost) → no reward
        "Delta": 0.8,  # High delta misalignment
        "Gamma": 0.5,  # High gamma
        "Vega": 200.0,  # High vega
        "MaxLoss": -5000.0,  # Large loss
    }
    config = {
        "scoring": {
            "w_pop": 0.35,
            "w_roc": 0.30,
            "w_theta": 0.10,
            "w_tail": 0.15,
            "w_delta": 0.05,
            "w_gamma": 0.03,
            "w_vega": 0.02,
        },
        "filters": {"max_capital": 10000.0, "max_loss_pct": 0.05},
    }

    score = scorer.score(candidate, metrics, config)
    assert score >= 0.0, f"Score should be non-negative, got {score}"
    assert score <= 1.0, f"Score should not exceed 1.0, got {score}"
