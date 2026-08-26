"""Ingest arbitrary peptide data drops into one canonical table.

The problem this solves: every AMP database exports a different shape. DBAASP
gives you TSV with MIC columns and unit strings. AMPSphere gives compressed
FASTA. Peptipedia gives CSV with activity flags. MarLys gives FASTA with
metadata crammed into the header. If you write a bespoke parser per source you
will spend a week on plumbing and still lose track of which sequence came from
where -- and provenance is a *submission requirement*.

So: drop whatever you downloaded into `data/raw/<source_id>/`, describe it once
in `sources.py`, and this module produces a single long-format table with one
row per (sequence, source) and a provenance manifest recording exactly what was
read.

Canonical output columns:

    sequence      str    upper-case, standard residues only
    length        int
    source        str    source_id from the registry
    role          str    'positive' | 'negative' | 'general' | 'reference'
    label         float  1.0 active, 0.0 inactive, NaN unknown
    mic_ugml      float  MIC in ug/mL if available, else NaN
    strain        str    strain/species string if available, else ''
    src_id        str    the source's own accession, if any
"""

from __future__ import annotations

import csv
import gzip
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

STANDARD = frozenset("ACDEFGHIKLMNPQRSTVWY")

#: Competition bounds. Sequences outside this cannot be submitted, but they can
#: still be useful for *training* the representation -- so filtering to this
#: range is a per-use decision, not something ingest should force.
MIN_LEN, MAX_LEN = 8, 50


# --------------------------------------------------------------------------- #
# Low-level readers
# --------------------------------------------------------------------------- #

def _open_maybe_gzip(path: Path) -> io.TextIOBase:
    """Open a file transparently whether or not it is gzipped."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return open(path, "r", errors="replace")


def read_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (header, sequence) pairs. Handles multi-line and gzip."""
    header: str | None = None
    parts: list[str] = []
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header, parts = line[1:], []
            else:
                parts.append(line)
    if header is not None:
        yield header, "".join(parts)


def read_table(path: Path) -> Iterator[dict[str, str]]:
    """Yield rows as dicts. Sniffs delimiter; handles gzip."""
    with _open_maybe_gzip(path) as fh:
        sample = fh.read(64_000)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            delim = dialect.delimiter
        except csv.Error:
            delim = "\t" if "\t" in sample else ","
        for row in csv.DictReader(fh, delimiter=delim):
            yield row


# --------------------------------------------------------------------------- #
# Canonicalization
# --------------------------------------------------------------------------- #

_WS = re.compile(r"\s+")


def canonical_sequence(raw: str) -> str | None:
    """Normalise a sequence string, or return None if unusable.

    Rejects anything containing a non-standard residue rather than silently
    substituting. Substitution is how you end up training on peptides that do
    not exist: `X` is not alanine, and a `B` or `Z` means the source could not
    resolve the residue. `U` (selenocysteine) and `O` (pyrrolysine) are real but
    non-proteinogenic-standard and are disallowed by the competition.
    """
    if not raw:
        return None
    seq = _WS.sub("", raw).upper().replace("-", "").replace("*", "")
    if not seq:
        return None
    if set(seq) - STANDARD:
        return None
    return seq


#: MIC strings in the wild: ">128", "8.0", "4-8", "16 ug/ml", "1.5 uM", "NA".
_MIC_NUM = re.compile(r"(\d+\.?\d*)")


def parse_mic(raw: str | None, assume_um_mw: float | None = None) -> tuple[float | None, bool]:
    """Parse a MIC cell into (value_ug_per_ml, is_censored).

    Returns censored=True for '>' style entries, which matter enormously: a
    right-censored MIC is *not* a missing value and it is *not* the number
    shown. Treating '>128' as 128 biases every regression you fit; dropping it
    throws away your strongest negative signal. Keep the flag and use interval
    /Tobit regression. See docs/PLAN.md 2.2 (iii).

    Ranges like '4-8' return the upper bound, which is the conservative choice.

    If the source reports uM and you pass the peptide's molecular weight,
    conversion is applied: ug/mL = uM * MW / 1000.
    """
    if raw is None:
        return None, False
    s = str(raw).strip()
    if not s or s.lower() in {"na", "nan", "n/a", "none", "-", "nd"}:
        return None, False

    censored = ">" in s or "≥" in s or ">=" in s
    nums = _MIC_NUM.findall(s)
    if not nums:
        return None, censored
    value = float(nums[-1])  # upper bound for ranges

    is_molar = bool(re.search(r"\b[uµ]m\b|micromolar", s, re.I))
    if is_molar and assume_um_mw:
        value = value * assume_um_mw / 1000.0

    return value, censored


# --------------------------------------------------------------------------- #
# Records and manifest
# --------------------------------------------------------------------------- #

@dataclass
class Record:
    sequence: str
    source: str
    role: str
    label: float | None = None
    mic_ugml: float | None = None
    mic_censored: bool = False
    strain: str = ""
    src_id: str = ""

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass
class IngestStats:
    """Per-source accounting. Print this; it is how you catch a bad parse."""

    source: str
    files: list[str] = field(default_factory=list)
    rows_read: int = 0
    rejected_nonstandard: int = 0
    rejected_empty: int = 0
    kept: int = 0
    with_label: int = 0
    with_mic: int = 0
    censored_mic: int = 0

    def summary(self) -> str:
        pct = 100.0 * self.kept / self.rows_read if self.rows_read else 0.0
        return (
            f"{self.source:<16} read={self.rows_read:>8,}  kept={self.kept:>8,} "
            f"({pct:5.1f}%)  nonstd={self.rejected_nonstandard:>7,}  "
            f"labels={self.with_label:>7,}  mic={self.with_mic:>7,} "
            f"(censored {self.censored_mic:,})"
        )
