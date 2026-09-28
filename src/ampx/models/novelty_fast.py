"""
Exhaustive maximum-identity scan against the reference corpus.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
from rapidfuzz import fuzz, process

from .compliance import MAX_REFERENCE_IDENTITY
from .features import clean_sequence


def length_bounds(length: int, threshold: float) -> Tuple[int, int]:
    """
    Reference lengths that could reach `threshold` against a query of this
    length. Derived from ratio = 1 - dist/(la+lb) and dist >= |la-lb|:
        |la - lb| <= (1-T)(la + lb)  =>  lb in [la(1-r)/(1+r), la(1+r)/(1-r)]
    with r = 1 - T.
    """
    r = 1.0 - float(threshold)
    if r <= 0:
        return length, length
    low = int(np.floor(length * (1.0 - r) / (1.0 + r)))
    high = int(np.ceil(length * (1.0 + r) / (1.0 - r)))
    return max(1, low), high


class ExhaustiveNovelty:
    """Maximum identity of each query against every reference."""

    def __init__(
        self,
        references: Iterable[str],
        threshold: float = MAX_REFERENCE_IDENTITY,
        exact_above: float = 0.40,
    ):
        """
        threshold   the accept/reject cutoff (0.80, the submission ceiling).
        exact_above the identity down to which reported values are exact.

        These differ because the length prefilter is lossless only at or
        above the identity it was derived from. A band derived from 0.80
        gives a correct accept/reject decision but can under-report the
        maximum for a sequence whose true nearest neighbour sits below 0.80 --
        which would corrupt the novelty-band stratification, where 0.40-0.50
        and 0.50-0.60 are real buckets. Deriving the band from 0.40 instead
        keeps every reported value at or above 0.40 exact. Values below 0.40
        may be under-reported; they all fall in the same "below the lowest
        band" bucket regardless.
        """
        cleaned = [clean_sequence(s) for s in references]
        self.references: List[str] = [s for s in cleaned if s]
        self.threshold = float(threshold)
        self.exact_above = float(exact_above)
        self.lengths = np.array([len(s) for s in self.references], dtype=np.int32)
        self._order = np.argsort(self.lengths, kind="stable")
        self._sorted_lengths = self.lengths[self._order]
        self._sorted_references = [self.references[i] for i in self._order]

    def _slice_for(self, length: int) -> slice:
        low, high = length_bounds(length, self.exact_above)
        start = int(np.searchsorted(self._sorted_lengths, low, side="left"))
        stop = int(np.searchsorted(self._sorted_lengths, high, side="right"))
        return slice(start, stop)

    def max_identity(self, queries: Sequence[str], workers: int = -1) -> np.ndarray:
        """
        Highest identity to any reference, one value per query.

        Queries are grouped by length so each group consults only the
        reference band that could possibly reach the threshold.
        """
        queries = [clean_sequence(s) for s in queries]
        best = np.zeros(len(queries), dtype=np.float64)
        by_length: dict = {}
        for index, sequence in enumerate(queries):
            by_length.setdefault(len(sequence), []).append(index)

        for length, indices in by_length.items():
            if length == 0:
                continue
            window = self._sorted_references[self._slice_for(length)]
            if not window:
                continue
            block = [queries[i] for i in indices]
            scores = process.cdist(
                block, window, scorer=fuzz.ratio, workers=workers, dtype=np.float64,
            )
            best[np.asarray(indices)] = scores.max(axis=1) / 100.0
        return best

    def is_novel(self, queries: Sequence[str], workers: int = -1) -> np.ndarray:
        """Strictly below the threshold, matching compliance's `<` test."""
        return self.max_identity(queries, workers=workers) < self.threshold


__all__ = ["ExhaustiveNovelty", "length_bounds"]
