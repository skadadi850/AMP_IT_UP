"""
Metrics, seed-level aggregation, and provenance for generator runs.

Design decisions that matter for reading the output.

Variance comes from two places: the training seed and the sampling seed.
With 50,000 samples the sampling noise on a median or a KS statistic is
tiny, so a single-seed difference of a few percent is almost always
training-seed variance rather than an effect of your change. Every metric
here is therefore computed per sampling seed and aggregated, and the
comparison across variants is paired on seed so the difference estimate is
tighter than two independent means.

Metric definitions follow OmegAMP's appendix H for uniqueness, diversity,
novelty and fitness, so numbers stay comparable to their published table.
Uniqueness and novelty are near-saturated for every published model, so
they are reported for comparability and should not be used to decide
anything.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .compliance import NoveltyFilter, alignment_identity, compliance_report
from .features import EISENBERG, STANDARD_AA, clean_sequence, compute_sequence_features

# Scales for OmegAMP's fitness score (their Table 10).
FITNESS_H = {
    "A": 0.25, "R": -1.80, "N": -0.64, "D": -0.72, "C": 0.04,
    "Q": -0.69, "E": -0.62, "G": 0.16, "H": -0.40, "I": 0.73,
    "L": 0.53, "K": -1.10, "M": 0.26, "F": 0.61, "P": -0.07,
    "S": -0.26, "T": -0.18, "W": 0.37, "Y": 0.02, "V": 0.54,
}
FITNESS_HX = {
    "A": 0.00, "R": 0.21, "N": 0.65, "D": 0.69, "C": 0.68,
    "Q": 0.39, "E": 0.40, "G": 1.00, "H": 0.61, "I": 0.41,
    "L": 0.21, "K": 0.26, "M": 0.24, "F": 0.54, "P": 3.16,
    "S": 0.50, "T": 0.66, "W": 0.49, "Y": 0.53, "V": 0.61,
}

EXPERT_CHARGE = (2.0, 10.0)
EXPERT_LENGTH = (10, 30)
EXPERT_HYDROPHOBICITY = (-0.5, 0.8)

DISTRIBUTION_COLUMNS = ("length", "net_charge", "eisenberg_mean", "mu_h", "gravy_raw")

# Decide on these and nothing else. Fixing the primary metric before you
# look at a run is what stops fifteen columns of noise from producing a
# story. Primary is the neutral-judge quality signal; the two secondaries
# are the ones tied to what the competition scores.
PRIMARY_METRIC = "expert_range_pct"
SECONDARY_METRICS = ("ks_mu_h", "submittable_pct")


def eisenberg_mean(sequence: str) -> float:
    if not sequence:
        return 0.0
    return sum(EISENBERG.get(a, 0.0) for a in sequence) / len(sequence)


def fitness_score(sequence: str) -> float:
    if not sequence:
        return 0.0
    theta = np.deg2rad(100.0)
    indices = np.arange(1, len(sequence) + 1)
    h = np.array([FITNESS_H.get(a, 0.0) for a in sequence])
    hx = np.array([FITNESS_HX.get(a, 0.0) for a in sequence])
    real = float(np.dot(h, np.cos(indices * theta)))
    imag = float(np.dot(h, np.sin(indices * theta)))
    denominator = float(np.sum(np.exp(hx)))
    return float(np.hypot(real, imag) / denominator) if denominator > 0 else 0.0


def sequence_properties(sequences: Iterable[str]) -> pd.DataFrame:
    unique = list(dict.fromkeys(clean_sequence(s) for s in sequences if str(s).strip()))
    frame = pd.DataFrame({"sequence": unique})
    if frame.empty:
        return frame
    features = compute_sequence_features(frame["sequence"])
    for column in features.columns:
        frame[column] = features[column].values
    frame["eisenberg_mean"] = frame["sequence"].apply(eisenberg_mean)
    frame["fitness"] = frame["sequence"].apply(fitness_score)
    return frame


def amino_acid_frequencies(sequences: Iterable[str]) -> np.ndarray:
    counts = np.zeros(len(STANDARD_AA), dtype=np.float64)
    for sequence in sequences:
        for residue in sequence:
            position = STANDARD_AA.find(residue)
            if position >= 0:
                counts[position] += 1.0
    total = counts.sum()
    return counts / total if total > 0 else counts


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    m = 0.5 * (p + q)
    return 0.5 * float(np.sum(p * np.log2(p / m))) + 0.5 * float(np.sum(q * np.log2(q / m)))


def pairwise_diversity(sequences: Sequence[str], sample: int, seed: int) -> float:
    """Mean pairwise alignment identity over a subsample; the exact
    statistic is quadratic and infeasible at library scale."""
    rng = np.random.default_rng(seed)
    unique = list(dict.fromkeys(sequences))
    if len(unique) < 2:
        return float("nan")
    if len(unique) > sample:
        unique = [unique[i] for i in rng.choice(len(unique), sample, replace=False)]
    total, pairs = 0.0, 0
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            total += alignment_identity(unique[i], unique[j])
            pairs += 1
    return total / pairs if pairs else float("nan")


def reference_set(real_frame: pd.DataFrame) -> Dict:
    """Precompute everything a comparison needs from the real data once."""
    return {
        "frame": real_frame,
        "sequences": set(real_frame["sequence"]),
        "frequencies": amino_acid_frequencies(real_frame["sequence"]),
    }


def set_metrics(
    sequences: Sequence[str],
    real: Dict,
    novelty: Optional[NoveltyFilter] = None,
    identity_sample: int = 2000,
    diversity_sample: int = 600,
    seed: int = 0,
) -> Dict[str, float]:
    """All metrics for one generated set at one sampling seed."""
    frame = sequence_properties(sequences)
    if frame.empty:
        return {"generated": len(sequences), "unique": 0}

    checks = frame["sequence"].apply(compliance_report).apply(pd.Series)
    in_expert = (
        frame["net_charge"].between(*EXPERT_CHARGE)
        & frame["length"].between(*EXPERT_LENGTH)
        & frame["eisenberg_mean"].between(*EXPERT_HYDROPHOBICITY)
    )

    record = {
        "generated": len(sequences),
        "unique": len(frame),
        "uniqueness_pct": 100.0 * len(frame) / max(len(sequences), 1),
        "novelty_pct": 100.0 * float(
            np.mean([s not in real["sequences"] for s in frame["sequence"]])
        ),
        "compliant_pct": 100.0 * float(checks.all(axis=1).mean()),
        "expert_range_pct": 100.0 * float(in_expert.mean()),
        "median_length": float(frame["length"].median()),
        "median_net_charge": float(frame["net_charge"].median()),
        "median_eisenberg": float(frame["eisenberg_mean"].median()),
        "median_mu_h": float(frame["mu_h"].median()),
        "median_gravy": float(frame["gravy_raw"].median()),
        "mean_fitness": float(frame["fitness"].mean()),
        "cys_free_pct": 100.0 * float((frame["cys_count"] == 0).mean()),
        "diversity": pairwise_diversity(list(frame["sequence"]), diversity_sample, seed),
        "aa_js_divergence": jensen_shannon(
            amino_acid_frequencies(frame["sequence"]), real["frequencies"]
        ),
    }

    for column in DISTRIBUTION_COLUMNS:
        if column in frame.columns and column in real["frame"].columns:
            record[f"ks_{column}"] = float(
                stats.ks_2samp(
                    frame[column].to_numpy(dtype=float),
                    real["frame"][column].to_numpy(dtype=float),
                ).statistic
            )

    if novelty is not None:
        rng = np.random.default_rng(seed)
        pool = list(frame["sequence"])
        if len(pool) > identity_sample:
            pool = [pool[i] for i in rng.choice(len(pool), identity_sample, replace=False)]
        identities = np.array([novelty.max_identity(s)[0] for s in pool])
        record["mean_max_reference_identity"] = float(identities.mean())
        record["submittable_pct"] = 100.0 * float((identities < 0.80).mean())

    return record


def aggregate_seeds(records: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Mean and standard error across sampling seeds.

    Standard error over seeds, not a bootstrap over sequences: with tens of
    thousands of samples the within-set bootstrap interval is far narrower
    than the across-seed spread, so quoting it would badly overstate
    precision.
    """
    if not records:
        return {}
    keys = sorted({key for record in records for key in record})
    out: Dict[str, float] = {"n_seeds": len(records)}
    for key in keys:
        values = np.array(
            [record[key] for record in records if key in record and np.isfinite(record[key])],
            dtype=float,
        )
        if values.size == 0:
            continue
        out[key] = float(values.mean())
        out[f"{key}_se"] = (
            float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        )
    return out


def paired_delta(
    baseline: List[Dict[str, float]],
    variant: List[Dict[str, float]],
    metric: str,
) -> Dict[str, float]:
    """
    Paired difference on one metric, seed by seed.

    Requires both runs to use the same sampling seeds in the same order.
    Pairing removes the shared sampling variance, which is the only way a
    three-seed comparison says anything useful.
    """
    a = np.array([r[metric] for r in baseline if metric in r], dtype=float)
    b = np.array([r[metric] for r in variant if metric in r], dtype=float)
    if a.size != b.size or a.size < 2:
        return {"delta": float(b.mean() - a.mean()) if a.size and b.size else float("nan")}
    difference = b - a
    result = stats.ttest_rel(b, a)
    return {
        "delta": float(difference.mean()),
        "delta_se": float(difference.std(ddof=1) / np.sqrt(difference.size)),
        "t": float(result.statistic),
        "p": float(result.pvalue),
        "cohens_d": float(difference.mean() / difference.std(ddof=1))
        if difference.std(ddof=1) > 0 else float("nan"),
    }


def git_provenance() -> Dict[str, str]:
    """
    Commit, branch and dirty state.

    A ledger row that cannot be traced to code is worthless three weeks
    later, and 'dirty' is the field that tells you whether the row is
    reproducible at all.
    """
    def run(*command) -> str:
        try:
            return subprocess.check_output(
                command, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return ""

    status = run("git", "status", "--porcelain")
    return {
        "git_commit": run("git", "rev-parse", "--short", "HEAD"),
        "git_branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": "yes" if status else "no",
    }


def append_ledger(path: str, row: Dict) -> None:
    """
    Append one run to the ledger, keeping the column union across runs.

    New metrics added later leave older rows blank rather than breaking the
    file, so the ledger survives schema drift.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if target.exists():
        existing = pd.read_csv(target)
        frame = pd.concat([existing, frame], ignore_index=True, sort=False)
    frame.to_csv(target, index=False)


__all__ = [
    "DISTRIBUTION_COLUMNS",
    "PRIMARY_METRIC",
    "SECONDARY_METRICS",
    "aggregate_seeds",
    "amino_acid_frequencies",
    "append_ledger",
    "eisenberg_mean",
    "fitness_score",
    "git_provenance",
    "jensen_shannon",
    "paired_delta",
    "pairwise_diversity",
    "reference_set",
    "sequence_properties",
    "set_metrics",
]