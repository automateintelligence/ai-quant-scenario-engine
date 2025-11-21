"""Stage 4 integration: trigger MC scoring for filtered candidates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd

from qse.optimizer.metrics import AdaptiveCISettings
from qse.optimizers.candidate_filter import Stage0Config, Stage1Config, filter_strikes, select_expiries
from qse.optimizers.candidate_generator import CandidateGenerator, GeneratorConfig
from qse.optimizers.models import CandidateStructure
from qse.optimizers.prefilter import Prefilter, Stage3Config


@dataclass(frozen=True)
class StrategyOptimizerResult:
    """Result bundle for staged optimization."""

    survivors: Tuple[CandidateStructure, ...]
    stage_counts: Dict[str, int]


class StrategyOptimizer:
    """Orchestrate Stage 0-4 filtering and MC scoring trigger."""

    def __init__(
        self,
        stage0_config: Stage0Config | None = None,
        stage1_config: Stage1Config | None = None,
        generator: CandidateGenerator | None = None,
        prefilter: Prefilter | None = None,
    ) -> None:
        self.stage0_config = stage0_config or Stage0Config()
        self.stage1_config = stage1_config or Stage1Config()
        self.generator = generator or CandidateGenerator(GeneratorConfig())
        self.prefilter = prefilter or Prefilter(Stage3Config())
        self.ci_settings = AdaptiveCISettings()

    def run(self, option_chain: pd.DataFrame, as_of: datetime, spot: float) -> StrategyOptimizerResult:
        """Run staged filtering and tag survivors for Monte Carlo pricing.

        Stage 4 does not perform simulations; it only annotates the survivors
        with the default Monte Carlo path budget so a downstream pricing engine
        can consume them without additional bookkeeping.
        """
        expiries = select_expiries(option_chain, as_of, self.stage0_config)
        filtered = filter_strikes(option_chain, spot, expiries, self.stage1_config)
        candidates = self.generator.generate(filtered, spot)
        survivors = self.prefilter.evaluate(candidates, spot)
        mc_prepared = self._trigger_mc_scoring(survivors)

        counts = {
            "stage0_expiries": len(expiries),
            "stage1_candidates": len(filtered),
            "stage2_structures": len(candidates),
            "stage3_survivors": len(survivors),
            "stage4_mc": len(mc_prepared),
        }
        return StrategyOptimizerResult(survivors=tuple(mc_prepared), stage_counts=counts)

    def _trigger_mc_scoring(self, survivors: Tuple[CandidateStructure, ...] | list[CandidateStructure]) -> Tuple[CandidateStructure, ...]:
        """Attach baseline Monte Carlo configuration to survivor metrics.

        The returned candidates are considered "MC-ready" because each
        carries the default path count expected by the downstream simulation
        layer. No pricing is executed here.
        """
        baseline_paths = self.ci_settings.baseline_paths
        for candidate in survivors:
            if candidate.metrics:
                candidate.metrics.mc_paths = baseline_paths
        return tuple(survivors)


__all__ = ["StrategyOptimizer", "StrategyOptimizerResult"]
