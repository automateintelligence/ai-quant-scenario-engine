"""Candidate selector interface per FR-CAND-001."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class CandidateEpisode:
    symbol: str
    start: str
    end: str
    score: float


class CandidateSelector(ABC):
    """Base selector for identifying candidate episodes."""

    @abstractmethod
    def select_candidates(self) -> List[CandidateEpisode]:
        """Return a ranked list of candidate episodes."""

