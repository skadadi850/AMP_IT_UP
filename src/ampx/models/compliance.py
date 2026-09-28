"""
Competition compliance and novelty filtering.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .features import (
    MAX_CYSTEINES,
    MAX_PEPTIDE_LENGTH,
    MIN_PEPTIDE_LENGTH,
    VALID_AA,
    clean_sequence,
)
import Levenshtein

MAX_REFERENCE_IDENTITY = 0.80


def read_fasta(path: str) -> List[str]:
    sequences: List[str] = []
    current: List[str] = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append("".join(current))
    return [clean_sequence(s) for s in sequences]


def write_fasta(path: str, sequences: Sequence[str], prefix: str = "seq") -> None:
    with open(path, "w") as handle:
        for index, sequence in enumerate(sequences, start=1):
            handle.write(f">{prefix}_{index}\n{sequence}\n")


def compliance_report(sequence: str) -> Dict[str, bool]:
    sequence = str(sequence).upper()
    return {
        "standard_residues": bool(sequence) and all(c in VALID_AA for c in sequence),
        "length_ok": MIN_PEPTIDE_LENGTH <= len(sequence) <= MAX_PEPTIDE_LENGTH,
        "disulfide_ok": sequence.count("C") <= MAX_CYSTEINES,
    }


def is_compliant(sequence: str) -> bool:
    return all(compliance_report(sequence).values())


def kmers(sequence: str, k: int = 3) -> frozenset:
    if len(sequence) < k:
        return frozenset({sequence})
    return frozenset(sequence[i:i + k] for i in range(len(sequence) - k + 1))

def alignment_identity(query: str, target: str, gap_penalty: float = -1.0) -> float:
    """
    Similarity exactly as the organizers' validator computes it.

    Levenshtein.ratio normalizes by the sum of both lengths, so four
    mismatches in a 20-mer against a 20-mer scores exactly 0.80 and passes
    the strict inequality. The previous Needleman-Wunsch version divided
    matches by the shorter length, which scored a short peptide fully
    contained in a longer reference as identity 1.0 even when they share
    only a fraction of their residues.
    """
    del gap_penalty
    if not query or not target:
        return 0.0
    return float(Levenshtein.ratio(query, target))

class NoveltyFilter:
    """
    Maximum identity of a candidate to a reference corpus.

    Build once from the organizers' fasta, then query per candidate. The
    same object also works against your own generated library, which is how
    you keep the top hundred from being a hundred near-duplicates of each
    other.
    """

    def __init__(self, references: Iterable[str], k: int = 3, shortlist: int = 400):
        self.k = int(k)
        self.shortlist = int(shortlist)
        self.references: List[str] = [clean_sequence(s) for s in references]
        self.references = [s for s in self.references if s]
        self.reference_kmers: List[frozenset] = [kmers(s, self.k) for s in self.references]
        self.kmer_sizes = np.array(
            [len(s) for s in self.reference_kmers], dtype=np.float64
        )
        self.index: Dict[str, List[int]] = defaultdict(list)
        for position, kmer_set in enumerate(self.reference_kmers):
            for kmer in kmer_set:
                self.index[kmer].append(position)

    def candidate_positions(self, sequence: str) -> np.ndarray:
        query = kmers(sequence, self.k)
        if not self.references:
            return np.zeros(0, dtype=np.int64)

        overlap = np.zeros(len(self.references), dtype=np.float64)
        for kmer in query:
            positions = self.index.get(kmer)
            if positions:
                overlap[positions] += 1.0

        union = len(query) + self.kmer_sizes - overlap
        jaccard = np.divide(
            overlap, union, out=np.zeros_like(overlap), where=union > 0
        )
        count = min(self.shortlist, len(self.references))
        return np.argpartition(-jaccard, count - 1)[:count]

    def max_identity(self, sequence: str) -> Tuple[float, Optional[str]]:
        sequence = clean_sequence(sequence)
        if not sequence or not self.references:
            return 0.0, None

        best_identity = 0.0
        best_reference = None
        for position in self.candidate_positions(sequence):
            reference = self.references[int(position)]
            identity = alignment_identity(sequence, reference)
            if identity > best_identity:
                best_identity = identity
                best_reference = reference
                if best_identity >= 1.0:
                    break
        return best_identity, best_reference

    def is_novel(self, sequence: str, threshold: float = MAX_REFERENCE_IDENTITY) -> bool:
        return self.max_identity(sequence)[0] < threshold


def deduplicate_library(
    sequences: Sequence[str],
    threshold: float = MAX_REFERENCE_IDENTITY,
    k: int = 3,
    shortlist: int = 200,
) -> List[str]:
    """
    Greedy internal diversity filter.

    Sequences are taken in the order given, so pass them ranked by the
    predictor: the highest-scoring member of each near-identical group
    survives. Without this the top hundred collapses onto one motif and the
    Phase 2 mean over twenty-five draws is decided by a single scaffold.
    """
    kept: List[str] = []
    kept_kmers: List[frozenset] = []
    index: Dict[str, List[int]] = defaultdict(list)

    for raw in sequences:
        sequence = clean_sequence(raw)
        if not sequence:
            continue

        query = kmers(sequence, k)
        if kept:
            overlap = defaultdict(int)
            for kmer in query:
                for position in index.get(kmer, ()):
                    overlap[position] += 1

            ranked = sorted(
                overlap.items(),
                key=lambda item: item[1] / max(len(query) + len(kept_kmers[item[0]]) - item[1], 1),
                reverse=True,
            )[:shortlist]

            too_similar = False
            for position, _ in ranked:
                if alignment_identity(sequence, kept[position]) >= threshold:
                    too_similar = True
                    break
            if too_similar:
                continue

        position = len(kept)
        kept.append(sequence)
        kept_kmers.append(query)
        for kmer in query:
            index[kmer].append(position)

    return kept


__all__ = [
    "MAX_REFERENCE_IDENTITY",
    "NoveltyFilter",
    "alignment_identity",
    "compliance_report",
    "deduplicate_library",
    "is_compliant",
    "kmers",
    "read_fasta",
    "write_fasta",
]
