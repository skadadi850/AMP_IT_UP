"""
Sequence-derived and assay-derived conditioning features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Competition sequence bounds.
MIN_PEPTIDE_LENGTH = 8
MAX_PEPTIDE_LENGTH = 50
LENGTH_SPAN = float(MAX_PEPTIDE_LENGTH - MIN_PEPTIDE_LENGTH)

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
VALID_AA = frozenset(STANDARD_AA)

# At most one disulfide bond is inside the organizers' target space, which
# is at most two cysteines.
MAX_CYSTEINES = 2

# Upper clamp on the Eisenberg hydrophobic moment. Set above max |Eisenberg|
# (2.53, arginine) so the statistic is never actually clipped for any peptide,
# real or generated. The previous value of 1.0 was a binding constraint.
MU_H_CEILING = 2.6

KYTE_DOOLITTLE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Eisenberg consensus hydrophobicity, used for the hydrophobic moment.
EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}

# Average residue masses, for ug/mL -> uM conversion of assay values.
RESIDUE_MASS = {
    "A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886,
    "C": 103.1388, "E": 129.1155, "Q": 128.1307, "G": 57.0519,
    "H": 137.1411, "I": 113.1594, "L": 113.1594, "K": 128.1741,
    "M": 131.1926, "F": 147.1766, "P": 97.1167, "S": 87.0782,
    "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326,
}
WATER_MASS = 18.0153

# log10(concentration in uM) is mapped from this window onto [0, 1]. MIC and
# HC50 share the window deliberately: the difference of the two normalized
# axes is then proportional to log10 of the selectivity index, so asking the
# generator for "potent and non-hemolytic" is a linear displacement in
# condition space rather than two unrelated requests.
LOG_CONC_MIN = -1.0
LOG_CONC_MAX = 3.0
LOG_CONC_SPAN = LOG_CONC_MAX - LOG_CONC_MIN

GRAM_INDEX = {
    "unknown": 0,
    "negative": 1,
    "positive": 2,
    "fungal": 3,
    "mycobacterial": 4,
}

# Genus-level gram assignment. Species-level MIC coverage is sparse, so this
# axis is the backstop that lets a rare species borrow structure from its
# class instead of collapsing into the unknown token.
GENUS_GRAM = {
    "escherichia": "negative",
    "pseudomonas": "negative",
    "klebsiella": "negative",
    "acinetobacter": "negative",
    "salmonella": "negative",
    "shigella": "negative",
    "enterobacter": "negative",
    "serratia": "negative",
    "proteus": "negative",
    "burkholderia": "negative",
    "stenotrophomonas": "negative",
    "haemophilus": "negative",
    "neisseria": "negative",
    "helicobacter": "negative",
    "campylobacter": "negative",
    "vibrio": "negative",
    "yersinia": "negative",
    "citrobacter": "negative",
    "moraxella": "negative",
    "legionella": "negative",
    "bordetella": "negative",
    "brucella": "negative",
    "francisella": "negative",
    "staphylococcus": "positive",
    "streptococcus": "positive",
    "enterococcus": "positive",
    "bacillus": "positive",
    "listeria": "positive",
    "clostridium": "positive",
    "clostridioides": "positive",
    "corynebacterium": "positive",
    "micrococcus": "positive",
    "lactobacillus": "positive",
    "lactococcus": "positive",
    "cutibacterium": "positive",
    "propionibacterium": "positive",
    "nocardia": "positive",
    "mycobacterium": "mycobacterial",
    "mycobacteroides": "mycobacterial",
    "mycolicibacterium": "mycobacterial",
    "candida": "fungal",
    "cryptococcus": "fungal",
    "aspergillus": "fungal",
    "saccharomyces": "fungal",
    "fusarium": "fungal",
    "trichophyton": "fungal",
    "malassezia": "fungal",
}


def clean_sequence(sequence) -> str:
    """Uppercase and drop anything outside the twenty standard residues."""
    return "".join(c for c in str(sequence).upper() if c in VALID_AA)


def is_valid_sequence(sequence: str) -> bool:
    return (
        MIN_PEPTIDE_LENGTH <= len(sequence) <= MAX_PEPTIDE_LENGTH
        and all(c in VALID_AA for c in sequence)
    )


def peptide_mass(sequence: str) -> float:
    """Average molecular mass in Da, for unit conversion of assay values."""
    if not sequence:
        return float("nan")
    return sum(RESIDUE_MASS.get(a, 0.0) for a in sequence) + WATER_MASS


def net_charge(sequence: str, ph: float = 7.4) -> float:
    """
    Net charge including free termini. Histidine is given a partial charge
    at physiological pH rather than a full one, which matters for the
    His-rich AMP families.
    """
    if not sequence:
        return 0.0
    positive = sequence.count("K") + sequence.count("R") + 0.1 * sequence.count("H")
    negative = sequence.count("D") + sequence.count("E")
    terminal = 1.0 - 1.0  # free N-terminus and free C-terminus cancel
    del ph
    return positive - negative + terminal


def hydrophobic_moment(sequence: str, window: int = 11, delta_deg: float = 100.0) -> float:
    """
    Maximum Eisenberg hydrophobic moment over sliding windows.

    This is the amphipathicity term: an alpha-helical AMP separates polar
    from apolar faces, and the moment measures exactly that separation. It
    is the single strongest sequence-only correlate of membrane lysis, and
    unlike net charge it is not already implied by composition.
    """
    values = np.array([EISENBERG.get(a, 0.0) for a in sequence], dtype=float)
    if values.size == 0:
        return 0.0
    width = int(min(window, values.size))
    angles = np.arange(width) * np.deg2rad(delta_deg)
    cos_terms = np.cos(angles)
    sin_terms = np.sin(angles)
    best = 0.0
    for start in range(values.size - width + 1):
        chunk = values[start:start + width]
        real = float(chunk @ cos_terms)
        imag = float(chunk @ sin_terms)
        best = max(best, float(np.hypot(real, imag)) / width)
    return best


def gravy(sequence: str) -> float:
    if not sequence:
        return 0.0
    return sum(KYTE_DOOLITTLE.get(a, 0.0) for a in sequence) / len(sequence)


def cys_class_index(sequence: str) -> int:
    """1 = no cysteine, 2 = two cysteines (one disulfide), 3 = more."""
    count = sequence.count("C")
    if count == 0:
        return 1
    if count <= MAX_CYSTEINES:
        return 2
    return 3


def gram_for_species(species) -> str:
    if species is None or (isinstance(species, float) and np.isnan(species)):
        return "unknown"
    text = str(species).strip().lower()
    if not text:
        return "unknown"
    genus = text.split()[0]
    return GENUS_GRAM.get(genus, "unknown")


def normalize_log_concentration(value_um) -> float:
    """Map a concentration in uM onto [0, 1] through log10."""
    if value_um is None or not np.isfinite(value_um) or value_um <= 0:
        return float("nan")
    log_value = np.log10(float(value_um))
    return float(np.clip((log_value - LOG_CONC_MIN) / LOG_CONC_SPAN, 0.0, 1.0))


def denormalize_log_concentration(value_norm: float) -> float:
    """Inverse of normalize_log_concentration, in uM."""
    log_value = float(value_norm) * LOG_CONC_SPAN + LOG_CONC_MIN
    return float(10.0 ** log_value)


# A 1000-fold preference either way spans the axis; the observed distribution
# of log10 ratios has sd well under 1, so this clips almost nothing.
SELECTIVITY_LOG_LIMIT = 3.0


def normalize_selectivity_ratio(log_ratio) -> float:
    """
    Map log10(MIC_gram_positive / MIC_gram_negative) onto [0, 1].

    0.5 is equipotent against both classes. Above 0.5 the peptide is more
    potent against gram-negatives (it takes more of it to inhibit a
    gram-positive); below 0.5, more potent against gram-positives.
    """
    if log_ratio is None or not np.isfinite(log_ratio):
        return float("nan")
    scaled = float(log_ratio) / (2.0 * SELECTIVITY_LOG_LIMIT) + 0.5
    return float(np.clip(scaled, 0.0, 1.0))


def denormalize_selectivity_ratio(value_norm: float) -> float:
    """Inverse of normalize_selectivity_ratio, as a log10 ratio."""
    return float((float(value_norm) - 0.5) * 2.0 * SELECTIVITY_LOG_LIMIT)


def normalize_length(length) -> float:
    return float(np.clip((float(length) - MIN_PEPTIDE_LENGTH) / LENGTH_SPAN, 0.0, 1.0))


def denormalize_length(length_norm: float) -> int:
    return int(round(float(length_norm) * LENGTH_SPAN + MIN_PEPTIDE_LENGTH))


def normalize_charge_per_residue(charge: float, length: int) -> float:
    if length <= 0:
        return 0.5
    return float(np.clip((charge / length + 1.0) / 2.0, 0.0, 1.0))


def normalize_gravy(value: float) -> float:
    return float(np.clip((value + 4.5) / 9.0, 0.0, 1.0))


def compute_sequence_features(sequences) -> pd.DataFrame:
    """
    Dense, always-observed conditioning axes for a series of sequences.

    Returns a frame with the axis columns used verbatim by
    ampgen.conditioning, so no renaming happens downstream.
    """
    series = pd.Series(list(sequences), dtype=object).astype(str).str.upper()

    lengths = series.str.len()
    charges = series.apply(net_charge)

    frame = pd.DataFrame(index=series.index)
    frame["length"] = lengths
    frame["length_norm"] = lengths.apply(normalize_length)
    frame["net_charge"] = charges
    frame["charge_per_res"] = [
        normalize_charge_per_residue(c, int(n)) for c, n in zip(charges, lengths)
    ]
    # Ceiling raised from 1.0 to MU_H_CEILING: the old clamp folded the
    # entire amphipathic tail into a single bin. Real MLAMP peptides reach
    # 1.31 under this implementation (1.68 under the modlamp hmoment column
    # carried in metadata.csv), so clipping at 1.0 truncated 1.8% of the
    # reference set and understated any KS gap on this axis.
    frame["mu_h"] = series.apply(hydrophobic_moment).clip(0.0, MU_H_CEILING)
    frame["gravy_raw"] = series.apply(gravy)
    frame["gravy"] = frame["gravy_raw"].apply(normalize_gravy)
    frame["cys_count"] = series.str.count("C")
    frame["cys_class_idx"] = series.apply(cys_class_index)
    return frame


def attach_conditioning_features(
    metadata: pd.DataFrame,
    sequence_column: str = "sequence",
) -> pd.DataFrame:
    """
    Add every conditioning axis column to a metadata frame.

    Expects the sparse assay columns to already be harmonized by
    ampgen.ingest (mic_um, mic_censored, hc50_um, species). Missing assay
    values stay NaN and become masked axes at training time.
    """
    frame = metadata.copy()
    frame[sequence_column] = frame[sequence_column].apply(clean_sequence)

    dense = compute_sequence_features(frame[sequence_column])
    for column in dense.columns:
        frame[column] = dense[column].values

    if "species" not in frame.columns:
        frame["species"] = np.nan
    if "gram" not in frame.columns or frame["gram"].isna().all():
        frame["gram"] = frame["species"].apply(gram_for_species)
    frame["gram_idx"] = frame["gram"].fillna("unknown").apply(
        lambda g: GRAM_INDEX.get(str(g).strip().lower(), 0)
    )

    if "mic_um" not in frame.columns:
        frame["mic_um"] = np.nan
    frame["log_mic"] = frame["mic_um"].apply(normalize_log_concentration)

    if "hc50_um" not in frame.columns:
        frame["hc50_um"] = np.nan
    frame["log_hc50"] = frame["hc50_um"].apply(normalize_log_concentration)

    # Gram selectivity, supplied per sequence by build_dataset.py as a raw
    # log10 ratio of gram-positive to gram-negative MIC. Normalized onto
    # [0, 1] with 0.5 as equipotent, so the axis is symmetric about "no
    # preference" and clipped at a 1000-fold preference either way.
    if "gram_selectivity_raw" not in frame.columns:
        frame["gram_selectivity_raw"] = np.nan
    frame["gram_selectivity"] = frame["gram_selectivity_raw"].apply(
        normalize_selectivity_ratio
    )

    # A right-censored MIC (reported as ">128") is not a measurement of 128.
    # Encoding the censoring as its own axis stops the model from learning
    # the detection limit as a potency target.
    if "mic_censored" not in frame.columns:
        frame["mic_censored"] = np.nan
    frame["mic_censored_idx"] = (
        frame["mic_censored"]
        .map({False: 1, True: 2, 0: 1, 1: 2, "exact": 1, "right": 2})
        .fillna(0)
        .astype(int)
    )
    frame.loc[frame["log_mic"].isna(), "mic_censored_idx"] = 0

    # is_amp is never left missing: a row that carries no is_amp_idx at all
    # (the plain MLAMP/GRAMPA path, with no --negatives corpus mixed in) is
    # a validated AMP by construction, so it defaults to 2 rather than 0.
    # Rows contributed by --negatives already set is_amp_idx=1 before this
    # function runs and are untouched by the fillna.
    if "is_amp_idx" not in frame.columns:
        frame["is_amp_idx"] = np.nan
    frame["is_amp_idx"] = frame["is_amp_idx"].fillna(2).astype(int)

    # hc50_source distinguishes a real measurement from a predictor's guess.
    # Unlike mic_censored this is genuinely absent (0/unknown) whenever no
    # HC50 was observed at all, not defaulted to a positive claim.
    if "hc50_source" not in frame.columns:
        frame["hc50_source"] = np.nan
    frame["hc50_source_idx"] = (
        frame["hc50_source"]
        .map({"measured": 1, "predicted": 2})
        .fillna(0)
        .astype(int)
    )
    frame.loc[frame["log_hc50"].isna(), "hc50_source_idx"] = 0

    return frame


def build_species_vocabulary(metadata: pd.DataFrame, min_count: int = 20) -> dict:
    """
    Species index map with 0 reserved for unknown.

    Species below min_count are folded into unknown on purpose: an embedding
    fitted on eight sequences is noise, and those sequences still contribute
    through the gram axis.
    """
    counts = metadata["species"].dropna().astype(str).str.strip().str.lower().value_counts()
    kept = sorted(counts[counts >= min_count].index.tolist())
    return {name: index + 1 for index, name in enumerate(kept)}


def species_index(metadata: pd.DataFrame, species_vocab: dict) -> pd.Series:
    normalized = metadata["species"].astype(str).str.strip().str.lower()
    return normalized.map(species_vocab).fillna(0).astype(int)
