"""Intraday-spreads composite scorer (US5/T027, FR-035 to FR-039)."""

from __future__ import annotations

from typing import Any

from qse.scorers.base import StrategyScorer


class IntradaySpreadsScorer(StrategyScorer):
    """
    Default composite scorer weighing POP, ROC, Theta, and Greek penalties.

    Scoring Formula (FR-035):
        Score = w_pop·POP_norm + w_roc·ROC_norm + w_theta·ThetaReward
                - w_tail·TailPenalty - w_delta·DeltaPenalty
                - w_gamma·GammaPenalty - w_vega·VegaPenalty

    Component Definitions:
        - POP_norm: Normalized probability of profit in [0, 1] (FR-036)
        - ROC_norm: Normalized return on capital in [0, 1] (FR-036)
        - ThetaReward: Time decay reward for income trades (FR-037)
        - TailPenalty: MaxLoss penalty relative to capital (FR-038)
        - DeltaPenalty: Directional exposure penalty (FR-038)
        - GammaPenalty: Gamma risk penalty (FR-038)
        - VegaPenalty: Vega exposure penalty (FR-038)

    Default Weights (FR-039):
        w_pop=0.35, w_roc=0.30, w_theta=0.10, w_tail=0.15,
        w_delta=0.05, w_gamma=0.03, w_vega=0.02

    Spec References: FR-035 to FR-039
    Tasks: T027
    """

    def score(
        self,
        candidate: dict[str, Any],
        metrics: dict[str, Any],
        config: dict[str, Any],
    ) -> float:
        """
        Compute composite score using weighted sum of components.

        Args:
            candidate: CandidateStructure with structure_type, legs, strikes, expiries
            metrics: MC metrics with E[PnL], POP_0, ROC, Theta, Delta, Gamma, Vega, MaxLoss
            config: Scoring config with weights, normalization scales, and target values

        Returns:
            Composite score in [0, 1] range (clamped)

        Spec: FR-035
        """
        # Load weights from config (FR-039 defaults)
        weights = self._load_weights(config)

        # Normalize POP and ROC (FR-036)
        pop_norm = self._normalize_pop(metrics.get("POP_0", 0.5), config)
        roc_norm = self._normalize_roc(metrics.get("ROC", 0.0), config)

        # Compute Theta reward (FR-037)
        theta_reward = self._compute_theta_reward(metrics.get("Theta", 0.0), config)

        # Compute penalties (FR-038)
        tail_penalty = self._compute_tail_penalty(
            metrics.get("MaxLoss", 0.0), config.get("filters", {})
        )
        delta_penalty = self._compute_delta_penalty(metrics.get("Delta", 0.0), config)
        gamma_penalty = self._compute_gamma_penalty(metrics.get("Gamma", 0.0), config)
        vega_penalty = self._compute_vega_penalty(metrics.get("Vega", 0.0), config)

        # Composite score (FR-035)
        raw_score = (
            weights["w_pop"] * pop_norm
            + weights["w_roc"] * roc_norm
            + weights["w_theta"] * theta_reward
            - weights["w_tail"] * tail_penalty
            - weights["w_delta"] * delta_penalty
            - weights["w_gamma"] * gamma_penalty
            - weights["w_vega"] * vega_penalty
        )

        # Clamp to [0, 1] range
        return max(0.0, min(1.0, raw_score))

    def decompose(
        self,
        candidate: dict[str, Any],
        metrics: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, float]:
        """
        Return score decomposition showing individual contributions (FR-041).

        Args:
            candidate: CandidateStructure dict
            metrics: MC metrics dict
            config: Scoring config section

        Returns:
            Dictionary with component contributions

        Spec: FR-041 (score decomposition for diagnostics)
        """
        weights = self._load_weights(config)

        pop_norm = self._normalize_pop(metrics.get("POP_0", 0.5), config)
        roc_norm = self._normalize_roc(metrics.get("ROC", 0.0), config)
        theta_reward = self._compute_theta_reward(metrics.get("Theta", 0.0), config)
        tail_penalty = self._compute_tail_penalty(
            metrics.get("MaxLoss", 0.0), config.get("filters", {})
        )
        delta_penalty = self._compute_delta_penalty(metrics.get("Delta", 0.0), config)
        gamma_penalty = self._compute_gamma_penalty(metrics.get("Gamma", 0.0), config)
        vega_penalty = self._compute_vega_penalty(metrics.get("Vega", 0.0), config)

        return {
            "pop_contrib": weights["w_pop"] * pop_norm,
            "roc_contrib": weights["w_roc"] * roc_norm,
            "theta_contrib": weights["w_theta"] * theta_reward,
            "tail_penalty": weights["w_tail"] * tail_penalty,
            "delta_penalty": weights["w_delta"] * delta_penalty,
            "gamma_penalty": weights["w_gamma"] * gamma_penalty,
            "vega_penalty": weights["w_vega"] * vega_penalty,
            "composite_score": self.score(candidate, metrics, config),
        }

    def _load_weights(self, config: dict[str, Any]) -> dict[str, float]:
        """Load scorer weights from config with FR-039 defaults."""
        scoring = config.get("scoring", {})
        return {
            "w_pop": scoring.get("w_pop", 0.35),
            "w_roc": scoring.get("w_roc", 0.30),
            "w_theta": scoring.get("w_theta", 0.10),
            "w_tail": scoring.get("w_tail", 0.15),
            "w_delta": scoring.get("w_delta", 0.05),
            "w_gamma": scoring.get("w_gamma", 0.03),
            "w_vega": scoring.get("w_vega", 0.02),
        }

    def _normalize_pop(self, pop: float, config: dict[str, Any]) -> float:
        """
        Normalize POP to [0, 1] range (FR-036).

        Formula: POP_norm = (POP - 0.5) / 0.5, clamped to [0, 1]
        Interpretation: POP <50% → negative contrib, POP >50% → positive contrib
        """
        pop_norm = (pop - 0.5) / 0.5
        return max(0.0, min(1.0, pop_norm))

    def _normalize_roc(self, roc: float, config: dict[str, Any]) -> float:
        """
        Normalize ROC to [0, 1] range using configurable scaling (FR-036).

        Default scale: 10% ROC = 1.0 normalized score
        Formula: ROC_norm = ROC / roc_scale, clamped to [0, 1]
        """
        scoring = config.get("scoring", {})
        roc_scale = scoring.get("roc_scale", 0.10)  # 10% ROC → score 1.0
        roc_norm = roc / roc_scale if roc_scale > 0 else 0.0
        return max(0.0, min(1.0, roc_norm))

    def _compute_theta_reward(self, theta: float, config: dict[str, Any]) -> float:
        """
        Compute Theta reward for positive time decay (FR-037).

        Formula: ThetaReward = max(0, Theta / Theta_scale), capped at 1
        Interpretation: Positive theta (income) rewarded, negative theta (cost) ignored
        """
        scoring = config.get("scoring", {})
        theta_scale = scoring.get("theta_scale", 50.0)  # $50/day theta → score 1.0
        theta_reward = max(0.0, theta / theta_scale) if theta_scale > 0 else 0.0
        return min(1.0, theta_reward)

    def _compute_tail_penalty(self, max_loss: float, filters: dict[str, Any]) -> float:
        """
        Compute tail risk penalty from MaxLoss relative to capital (FR-038).

        Formula: TailPenalty = MaxLoss_trade / (max_capital × max_loss_pct)
        Interpretation: Larger losses relative to capital → higher penalty
        """
        max_capital = filters.get("max_capital", 15000.0)
        max_loss_pct = filters.get("max_loss_pct", 0.05)

        if max_capital <= 0 or max_loss_pct <= 0:
            return 0.0

        # Absolute value since max_loss is typically negative
        loss_fraction = abs(max_loss) / (max_capital * max_loss_pct)
        return min(1.0, loss_fraction)

    def _compute_delta_penalty(self, delta: float, config: dict[str, Any]) -> float:
        """
        Compute directional exposure penalty (FR-038).

        Formula: DeltaPenalty = |Delta - Delta_target| / Delta_scale
        Default: Delta_target=0 (neutral strategies preferred)
        """
        scoring = config.get("scoring", {})
        delta_target = scoring.get("delta_target", 0.0)
        delta_scale = scoring.get("delta_scale", 0.5)  # ±0.5 delta → penalty 1.0

        if delta_scale <= 0:
            return 0.0

        delta_deviation = abs(delta - delta_target)
        return min(1.0, delta_deviation / delta_scale)

    def _compute_gamma_penalty(self, gamma: float, config: dict[str, Any]) -> float:
        """
        Compute gamma risk penalty (FR-038).

        Formula: GammaPenalty = |Gamma| / Gamma_scale
        Interpretation: Higher gamma → more sensitive to price changes → higher penalty
        """
        scoring = config.get("scoring", {})
        gamma_scale = scoring.get("gamma_scale", 0.10)  # |Gamma| = 0.10 → penalty 1.0

        if gamma_scale <= 0:
            return 0.0

        return min(1.0, abs(gamma) / gamma_scale)

    def _compute_vega_penalty(self, vega: float, config: dict[str, Any]) -> float:
        """
        Compute vega exposure penalty (FR-038).

        Formula: VegaPenalty = |Vega| / Vega_scale
        Interpretation: Higher vega → more sensitive to IV changes → higher penalty
        """
        scoring = config.get("scoring", {})
        vega_scale = scoring.get("vega_scale", 50.0)  # |Vega| = 50 → penalty 1.0

        if vega_scale <= 0:
            return 0.0

        return min(1.0, abs(vega) / vega_scale)
