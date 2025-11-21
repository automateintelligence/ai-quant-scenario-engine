"""Stage 3 analytic prefilter and hard constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from qse.optimizers.models import CandidateMetrics, CandidateStructure


@dataclass(frozen=True)
class Stage3Config:
    """Configuration for Stage 3 analytic filtering."""

    max_capital: float = 15_000.0
    max_loss_pct: float = 0.05
    min_expected_pnl: float = 50.0
    min_pop_breakeven: float = 0.55
    min_pop_target: float = 0.60
    top_k_per_type: int = 20
    profit_target: float = 500.0


class Prefilter:
    """Apply analytic prefiltering and enforce hard constraints (Stage 3)."""

    def __init__(self, config: Stage3Config | None = None) -> None:
        self.config = config or Stage3Config()

    def evaluate(self, candidates: Sequence[CandidateStructure], spot: float) -> List[CandidateStructure]:
        scored: List[CandidateStructure] = []
        for candidate in candidates:
            metrics = self._compute_metrics(candidate, spot)
            if not self._passes_constraints(metrics):
                continue
            candidate.metrics = metrics
            scored.append(candidate)

        return self._top_k(scored)

    def _compute_metrics(self, candidate: CandidateStructure, spot: float) -> CandidateMetrics:
        width = max(candidate.width, 1.0)
        net_credit = candidate.net_premium * 100.0
        capital = max(width * 100.0, abs(net_credit))
        short_legs = [leg for leg in candidate.legs if leg.side == "sell"]
        avg_short_strike = sum(leg.strike for leg in short_legs) / max(len(short_legs), 1)
        distance = abs(avg_short_strike - spot) / spot

        pop_breakeven = min(0.95, 0.55 + distance)
        pop_target = min(0.99, pop_breakeven + 0.05)
        expected_pnl = max(net_credit * 0.8, width * 20.0)
        max_loss = max(capital - net_credit, capital * (1.0 - pop_breakeven))
        score = expected_pnl / capital if capital > 0 else 0.0

        return CandidateMetrics(
            expected_pnl=expected_pnl,
            pop_breakeven=pop_breakeven,
            pop_target=pop_target,
            capital=capital,
            max_loss=max_loss,
            score=score,
        )

    def _passes_constraints(self, metrics: CandidateMetrics) -> bool:
        if metrics.capital > self.config.max_capital:
            return False
        if metrics.capital <= 0:
            return False
        loss_ratio = metrics.max_loss / self.config.max_capital
        if loss_ratio > self.config.max_loss_pct:
            return False
        if metrics.expected_pnl < self.config.min_expected_pnl:
            return False
        if metrics.pop_breakeven < self.config.min_pop_breakeven:
            return False
        if metrics.pop_target < self.config.min_pop_target:
            return False
        return True

    def _top_k(self, candidates: List[CandidateStructure]) -> List[CandidateStructure]:
        grouped: dict[str, List[CandidateStructure]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.structure_type, []).append(candidate)

        survivors: List[CandidateStructure] = []
        for bucket in grouped.values():
            sorted_bucket = sorted(bucket, key=lambda c: c.metrics.score if c.metrics else 0.0, reverse=True)
            survivors.extend(sorted_bucket[: self.config.top_k_per_type])
        return survivors


__all__ = ["Prefilter", "Stage3Config"]
