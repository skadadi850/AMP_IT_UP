"""Hard-constraint gate for AMP Challenge 2027 submissions.

Every rule here is transcribed from the organizers' own
`scripts/verify_submission.py`. Run these checks *before* you ship, because a
submission that fails the automated compliance check burns your 72-hour
remediation window -- or your entry.

Two distinct standards apply:

  * The 50,000-member library: valid alphabet, length 8-50, all unique, and
    *no exact match* to any sequence in the antibacterial reference set.
  * The top-100 list: everything above, plus **no sequence may exceed 0.80
    Levenshtein ratio** against *any* reference sequence.

The similarity bar is the one that catches people. It is not "80% identity to
the nearest AMP you were thinking of" -- it is a pairwise scan against all
~39k references, and one violation invalidates the list.

Note on the real Phase 1 pipeline: the competition paper specifies MMseqs2
pairwise alignment against the MarLys AMP database (~102k sequences), which is
a *different and broader* test than the Levenshtein check in the released
validator. Passing this module is necessary, not sufficient. Run an MMseqs2
pass against MarLys as well before submitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import Levenshtein

STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
MIN_LENGTH = 8
MAX_LENGTH = 50
LIBRARY_SIZE = 50_000
TOP_SIZE = 100

#: The validator's literal rule: it fails a sequence when ratio > 0.80.
IDENTITY_CEILING = 0.80

#: What you should actually select against. Selecting right at 0.80 means a
#: candidate at exactly 0.800 passes only because the comparison is strict, and
#: the real Phase 1 check is MMseqs2 alignment identity against MarLys -- a
#: different metric on a larger database, which will not agree with Levenshtein
#: at the third decimal place. Leave headroom. Costs you almost nothing: you are
#: picking 100 from 50,000.
SAFETY_CEILING = 0.75


# --------------------------------------------------------------------------- #
# FASTA I/O
# --------------------------------------------------------------------------- #

def read_fasta(path: str | Path) -> tuple[list[str], list[str]]:
    """Read a FASTA file into (headers, sequences).

    Mirrors the organizers' parser: multi-line records are concatenated and
    sequences are upper-cased.
    """
    headers: list[str] = []
    sequences: list[str] = []
    header: str | None = None
    parts: list[str] = []

    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                headers.append(header)
                sequences.append("".join(parts))
            header, parts = line[1:], []
        else:
            parts.append(line.upper())

    if header is not None:
        headers.append(header)
        sequences.append("".join(parts))

    return headers, sequences


def write_fasta(sequences: list[str], path: str | Path, prefix: str = "seq") -> None:
    """Write sequences to FASTA with stable, 1-indexed headers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for i, seq in enumerate(sequences, start=1):
            fh.write(f">{prefix}{i}\n{seq}\n")


# --------------------------------------------------------------------------- #
# Single-sequence validity
# --------------------------------------------------------------------------- #

def is_valid_sequence(seq: str) -> bool:
    """True if `seq` satisfies alphabet and length constraints."""
    return (
        MIN_LENGTH <= len(seq) <= MAX_LENGTH
        and not (set(seq) - STANDARD_AMINO_ACIDS)
    )


def invalid_reason(seq: str) -> str | None:
    """Human-readable reason `seq` is non-compliant, or None if it is fine."""
    bad = sorted(set(seq) - STANDARD_AMINO_ACIDS)
    if bad:
        return f"non-standard residues {bad}"
    if len(seq) < MIN_LENGTH:
        return f"too short ({len(seq)} < {MIN_LENGTH})"
    if len(seq) > MAX_LENGTH:
        return f"too long ({len(seq)} > {MAX_LENGTH})"
    return None


# --------------------------------------------------------------------------- #
# Similarity screening
# --------------------------------------------------------------------------- #

def _ratio_upper_bound(len_a: int, len_b: int) -> float:
    """Cheap upper bound on Levenshtein.ratio for two lengths.

    ratio = 2*M / (len_a + len_b) where M is the number of matched characters,
    and M <= min(len_a, len_b). So the ratio can never exceed
    2*min / (len_a + len_b). When that bound is already below the ceiling we can
    skip the expensive alignment entirely -- this prunes the great majority of
    pairs, since AMP reference lengths are broadly spread over 8-50.
    """
    return 2.0 * min(len_a, len_b) / (len_a + len_b)


def max_identity(seq: str, references: list[str], ceiling: float = IDENTITY_CEILING) -> float:
    """Highest Levenshtein ratio between `seq` and any reference.

    Short-circuits as soon as the ceiling is exceeded, so a violating sequence
    is rejected quickly. Returns the running maximum, which is exact when the
    result is <= ceiling.
    """
    best = 0.0
    n = len(seq)
    for ref in references:
        if _ratio_upper_bound(n, len(ref)) <= ceiling:
            continue
        r = Levenshtein.ratio(seq, ref)
        if r > best:
            best = r
            if best > ceiling:
                return best
    return best


def passes_novelty(seq: str, references: list[str], ceiling: float = IDENTITY_CEILING) -> bool:
    """True if `seq` is dissimilar enough from every reference for the top-100."""
    return max_identity(seq, references, ceiling) <= ceiling


# --------------------------------------------------------------------------- #
# Whole-submission verification
# --------------------------------------------------------------------------- #

@dataclass
class Report:
    """Outcome of a compliance run."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self) -> str:
        lines = []
        for e in self.errors:
            lines.append(f"  FAIL  {e}")
        for w in self.warnings:
            lines.append(f"  warn  {w}")
        if not lines:
            return "All compliance checks passed."
        return "\n".join(lines)


def verify_library(
    sequences: list[str],
    reference: set[str] | None = None,
    expected_size: int = LIBRARY_SIZE,
) -> Report:
    """Check the full library against every constraint the validator applies."""
    report = Report()

    if len(sequences) != expected_size:
        report.errors.append(
            f"library has {len(sequences)} sequences, expected {expected_size}"
        )

    seen: set[str] = set()
    n_dupes = 0
    bad: dict[str, int] = {}

    for seq in sequences:
        reason = invalid_reason(seq)
        if reason:
            bad[reason] = bad.get(reason, 0) + 1
        if seq in seen:
            n_dupes += 1
        seen.add(seq)

    for reason, count in sorted(bad.items(), key=lambda kv: -kv[1]):
        report.errors.append(f"{count} sequence(s): {reason}")
    if n_dupes:
        report.errors.append(f"{n_dupes} duplicate sequence(s) in library")

    if reference is not None:
        overlap = seen & reference
        if overlap:
            report.errors.append(
                f"{len(overlap)} sequence(s) exactly match the antibacterial "
                f"reference set (library must contain none)"
            )

    return report


def verify_top(
    top: list[str],
    library: list[str],
    references: list[str] | None = None,
    top_k: int = TOP_SIZE,
    ceiling: float = IDENTITY_CEILING,
) -> Report:
    """Check the ranked top-K list, including the strict similarity ceiling."""
    report = Report()
    library_set = set(library)

    if len(top) != top_k:
        report.errors.append(f"top list has {len(top)} sequences, expected {top_k}")

    seen: set[str] = set()
    for i, seq in enumerate(top, start=1):
        if seq not in library_set:
            report.errors.append(f"top #{i} is not present in the library")
        if seq in seen:
            report.errors.append(f"top #{i} is a duplicate within the top list")
        seen.add(seq)

    if references is not None:
        for i, seq in enumerate(top, start=1):
            ident = max_identity(seq, references, ceiling)
            if ident > ceiling:
                report.errors.append(
                    f"top #{i} has {ident:.3f} identity to a reference "
                    f"(ceiling {ceiling:.2f})"
                )

    return report
