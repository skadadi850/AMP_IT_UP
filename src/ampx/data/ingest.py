"""
Harmonization of raw activity exports into one table.

Every MIC/HC50 source you add reports concentration differently: ug/mL vs
uM vs nM, ">128" vs "128" vs "128-256", per-strain vs per-species. This
module reduces all of it to a single schema so that adding a source is a
column mapping rather than a new code path.

Output schema, one row per (sequence, species, endpoint):

    sequence        cleaned, standard residues only
    species         lowercase binomial, or NaN
    strain          free text, retained as provenance only
    endpoint        'mic' or 'hc50'
    value_um        concentration in uM
    censored        False for an exact value, True for right-censored
    source          provenance label
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..models.features import clean_sequence, is_valid_sequence, peptide_mass

ENDPOINTS = ("mic", "hc50")

# Multiplicative factors onto uM for concentration units that are already
# molar. Mass-per-volume units need the peptide mass and are handled apart.
MOLAR_UNIT_FACTORS = {
    "m": 1e6,
    "mm": 1e3,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "nm": 1e-3,
    "pm": 1e-6,
    "mol/l": 1e6,
    "mmol/l": 1e3,
    "umol/l": 1.0,
    "µmol/l": 1.0,
    "nmol/l": 1e-3,
}

# Factors onto ug/mL for mass-per-volume units.
MASS_UNIT_FACTORS = {
    "ug/ml": 1.0,
    "µg/ml": 1.0,
    "μg/ml": 1.0,
    "mcg/ml": 1.0,
    "mg/l": 1.0,
    "ug/l": 1e-3,
    "mg/ml": 1e3,
    "g/l": 1e3,
    "ng/ml": 1e-3,
    "ng/ul": 1.0,
}

_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def normalize_unit(unit) -> str:
    if unit is None or (isinstance(unit, float) and np.isnan(unit)):
        return ""
    text = str(unit).strip().lower()
    text = text.replace(" ", "").replace("−", "-")
    text = text.replace("micro", "u").replace("μ", "u").replace("µ", "u")
    return text


def parse_value(raw) -> tuple:
    """
    Parse a concentration cell into (value, censored).

    Handles bare numbers, comparison prefixes and ranges. A range is
    reduced to its geometric mean, which is the right summary for a
    quantity the model consumes in log space. A ">" prefix marks right
    censoring; a "<" prefix is treated as exact at the stated bound, since
    left censoring on a potency endpoint is not informative about how much
    more potent the peptide really is.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return float("nan"), False
    if isinstance(raw, (int, float)) and np.isfinite(raw):
        return float(raw), False

    text = str(raw).strip()
    if not text:
        return float("nan"), False

    censored = text.startswith(">") or text.startswith("≥")
    numbers = [float(m) for m in _NUMBER.findall(text)]
    # Zero and non-finite parses are dropped, but negative numbers are kept:
    # a log-scale source (e.g. GRAMPA's log10(uM) value) legitimately
    # reports negatives, and positivity is enforced downstream in
    # harmonize_activity after unit conversion to uM, not here.
    numbers = [n for n in numbers if n != 0 and np.isfinite(n)]
    if not numbers:
        return float("nan"), False
    if len(numbers) == 1:
        return numbers[0], censored
    return float(np.exp(np.mean(np.log(numbers[:2])))), censored


def to_micromolar(value: float, unit: str, sequence: str) -> float:
    """
    Convert a parsed concentration to uM.

    An unrecognized or absent unit is assumed to be uM, which is what most
    curated AMP databases export; the assumption is logged by the caller
    through the count of unconverted rows rather than silently.
    """
    if not np.isfinite(value):
        return float("nan")
    key = normalize_unit(unit)
    if key == "log10_um":
        # GRAMPA's `value` column is log10(MIC in uM) despite its own `unit`
        # column reading the misleading literal "uM" -- callers must pass
        # this unit explicitly via default_unit, never read it off the file.
        return 10.0 ** value
    if key in MOLAR_UNIT_FACTORS:
        return value * MOLAR_UNIT_FACTORS[key]
    if key in MASS_UNIT_FACTORS:
        mass = peptide_mass(sequence)
        if not np.isfinite(mass) or mass <= 0:
            return float("nan")
        return value * MASS_UNIT_FACTORS[key] * 1000.0 / mass
    if key == "":
        return value
    return float("nan")


# Maps an abbreviated genus initial plus epithet onto the full genus name.
# The initial alone is ambiguous -- "A. faecalis" is Alcaligenes while
# "E. faecalis" is Enterococcus -- so the key is always the (initial,
# epithet) pair, never the initial by itself.
ABBREVIATED_GENERA = {
    ("a", "baumannii"): "acinetobacter",
    ("a", "faecalis"): "alcaligenes",
    ("b", "subtilis"): "bacillus",
    ("b", "cereus"): "bacillus",
    ("b", "anthracis"): "bacillus",
    ("c", "albicans"): "candida",
    ("c", "difficile"): "clostridioides",
    ("c", "perfringens"): "clostridium",
    ("e", "coli"): "escherichia",
    ("e", "faecalis"): "enterococcus",
    ("e", "faecium"): "enterococcus",
    ("e", "cloacae"): "enterobacter",
    ("h", "influenzae"): "haemophilus",
    ("h", "pylori"): "helicobacter",
    ("k", "pneumoniae"): "klebsiella",
    ("l", "monocytogenes"): "listeria",
    ("m", "luteus"): "micrococcus",
    ("m", "tuberculosis"): "mycobacterium",
    ("m", "smegmatis"): "mycobacterium",
    ("n", "gonorrhoeae"): "neisseria",
    ("n", "meningitidis"): "neisseria",
    ("p", "aeruginosa"): "pseudomonas",
    ("p", "mirabilis"): "proteus",
    ("p", "vulgaris"): "proteus",
    ("s", "aureus"): "staphylococcus",
    ("s", "epidermidis"): "staphylococcus",
    ("s", "pyogenes"): "streptococcus",
    ("s", "pneumoniae"): "streptococcus",
    ("s", "agalactiae"): "streptococcus",
    ("s", "typhimurium"): "salmonella",
    ("s", "enterica"): "salmonella",
    ("s", "flexneri"): "shigella",
    ("s", "marcescens"): "serratia",
    ("v", "cholerae"): "vibrio",
    ("v", "parahaemolyticus"): "vibrio",
    ("y", "pestis"): "yersinia",
    ("y", "enterocolitica"): "yersinia",
}

# Known misspellings found in the raw exports, keyed on the (already
# lowercased, period-stripped) epithet as parsed. Corrected before the
# abbreviated-genus lookup above, so a misspelled abbreviated form (e.g.
# "A. baumanii") still resolves correctly.
SPECIES_EPITHET_CORRECTIONS = {
    "baumanii": "baumannii",
}


def normalize_species(raw) -> float | str:
    """
    Reduce a target label to a lowercase binomial.

    Strain designations are stripped: DBAASP's per-strain coverage of any
    given competition panel strain is too thin to fit a per-strain
    embedding, so species is the modeling unit and strain survives only as
    provenance.

    Abbreviated genera ("E. coli") are expanded through ABBREVIATED_GENERA
    and known misspellings are corrected, so "E. coli" and "Escherichia
    coli" collapse to the same species and "A. baumanii" doesn't fragment
    off from "A. baumannii" into its own near-empty bucket.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return float("nan")
    text = str(raw).strip()
    if not text:
        return float("nan")
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"[^A-Za-z. ]", " ", text)
    tokens = [t for t in text.split() if t]
    if not tokens:
        return float("nan")
    genus = tokens[0].lower().strip(".")
    if len(tokens) == 1:
        return genus
    epithet = tokens[1].lower().strip(".")
    if epithet in {"sp", "spp", "species"} or len(epithet) < 2:
        return genus
    epithet = SPECIES_EPITHET_CORRECTIONS.get(epithet, epithet)
    if len(genus) == 1:
        genus = ABBREVIATED_GENERA.get((genus, epithet), genus)
    return f"{genus} {epithet}"


def harmonize_activity(
    frame: pd.DataFrame,
    column_map: dict,
    endpoint: str,
    source: str,
    default_unit: str | None = None,
    row_filters: dict | None = None,
) -> pd.DataFrame:
    """
    Convert one raw export into the common schema.

    column_map keys, all optional except 'sequence' and 'value':
        sequence, value, unit, species, strain, relation

    'relation' is for sources that keep the comparison operator in its own
    column instead of inside the value string.

    row_filters is an optional {column: required_value} map applied before
    anything else. GRAMPA needs this to drop YADAMP: YADAMP's
    has_unusual_modification is False for every row only because that
    source carries no modification annotation at all, not because its
    peptides are verified unmodified, so has_unusual_modification alone
    would silently admit ~4,000 rows of unknown provenance. Requiring
    datasource_has_modifications == True as well excludes them correctly.
    """
    if endpoint not in ENDPOINTS:
        raise ValueError(f"endpoint must be one of {ENDPOINTS}, got {endpoint!r}")
    for required in ("sequence", "value"):
        if required not in column_map:
            raise ValueError(f"column_map is missing {required!r}")

    if row_filters:
        keep = pd.Series(True, index=frame.index)
        for column, required_value in row_filters.items():
            keep &= frame[column] == required_value
        frame = frame[keep]

    rows = []
    unit_column = column_map.get("unit")
    species_column = column_map.get("species")
    strain_column = column_map.get("strain")
    relation_column = column_map.get("relation")

    for _, record in frame.iterrows():
        sequence = clean_sequence(record[column_map["sequence"]])
        if not is_valid_sequence(sequence):
            continue

        value, censored = parse_value(record[column_map["value"]])
        if relation_column and relation_column in record:
            relation = str(record[relation_column]).strip()
            censored = censored or relation.startswith(">") or relation.startswith("≥")

        unit = record[unit_column] if unit_column and unit_column in record else default_unit
        value_um = to_micromolar(value, unit, sequence)
        if not np.isfinite(value_um) or value_um <= 0:
            continue

        rows.append({
            "sequence": sequence,
            "species": normalize_species(record[species_column]) if species_column else float("nan"),
            "strain": str(record[strain_column]) if strain_column and strain_column in record else "",
            "endpoint": endpoint,
            "value_um": float(value_um),
            "censored": bool(censored),
            "source": source,
        })

    return pd.DataFrame(rows, columns=[
        "sequence", "species", "strain", "endpoint", "value_um", "censored", "source",
    ])


def aggregate_activity(activity: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse replicate measurements to one value per (sequence, species,
    endpoint).

    The aggregate is the geometric mean of the exact measurements, since the
    model consumes log10 concentration. Censored rows are only used when a
    sequence-species pair has no exact measurement at all, in which case the
    bound is kept and flagged, so that a peptide measured only as ">128" is
    still usable as a weak-activity example without being read as potent.
    """
    if activity.empty:
        return activity.assign(value_um=[], censored=[])

    activity = activity.copy()
    activity["species"] = activity["species"].fillna("__unknown__")

    records = []
    grouped = activity.groupby(["sequence", "species", "endpoint"], sort=False)
    for (sequence, species, endpoint), block in grouped:
        # HC50 provenance is not blended: a measured value always wins over
        # a predicted one for the same sequence, rather than being averaged
        # into it, since the predicted value carries no independent evidence
        # once a real measurement exists.
        hc50_source = np.nan
        if endpoint == "hc50":
            measured = block[block["source"] == "hemopi2_measured"]
            if len(measured) > 0:
                block = measured
                hc50_source = "measured"
            elif (block["source"] == "hemopi2_predicted").any():
                hc50_source = "predicted"

        exact = block.loc[~block["censored"], "value_um"]
        if len(exact) > 0:
            value = float(np.exp(np.mean(np.log(exact.values))))
            censored = False
        else:
            value = float(block["value_um"].max())
            censored = True
        records.append({
            "sequence": sequence,
            "species": np.nan if species == "__unknown__" else species,
            "endpoint": endpoint,
            "value_um": value,
            "censored": censored,
            "n_measurements": int(len(block)),
            "sources": ",".join(sorted(set(block["source"]))),
            "hc50_source": hc50_source,
        })

    return pd.DataFrame(records)


def build_metadata(
    positives: pd.DataFrame,
    activity: pd.DataFrame,
    sequence_column: str = "sequence",
) -> pd.DataFrame:
    """
    Join the aggregated activity onto the positive sequence set.

    One row per (sequence, species) for sequences with MIC data, plus one
    species-free row for every sequence without any. This is what makes a
    single generator serve both the 40k unannotated MLAMP positives and the
    few thousand with real MIC: the unannotated rows train the dense axes
    and the sequence prior, the annotated rows train the potency axis.

    HC50 is a peptide-level property, so it is broadcast across all species
    rows for a sequence rather than joined per species.
    """
    positives = positives.copy()
    positives[sequence_column] = positives[sequence_column].apply(clean_sequence)
    positives = positives[positives[sequence_column].str.len() > 0]
    positives = positives.drop_duplicates(subset=[sequence_column])

    aggregated = aggregate_activity(activity) if not activity.empty else activity

    if aggregated.empty:
        frame = positives.copy()
        frame["species"] = np.nan
        frame["mic_um"] = np.nan
        frame["mic_censored"] = np.nan
        frame["hc50_um"] = np.nan
        frame["hc50_source"] = np.nan
        return frame.reset_index(drop=True)

    mic = aggregated[aggregated["endpoint"] == "mic"].rename(
        columns={"value_um": "mic_um", "censored": "mic_censored"}
    )[["sequence", "species", "mic_um", "mic_censored"]]

    def _prefer_hc50_source(sources: pd.Series) -> float | str:
        values = set(sources.dropna())
        if "measured" in values:
            return "measured"
        if "predicted" in values:
            return "predicted"
        return float("nan")

    hc50 = aggregated[aggregated["endpoint"] == "hc50"].groupby("sequence", as_index=False).agg(
        hc50_um=("value_um", lambda values: float(np.exp(np.mean(np.log(values))))),
        hc50_source=("hc50_source", _prefer_hc50_source),
    )

    annotated = positives.merge(mic, on=sequence_column, how="inner")
    unannotated = positives[~positives[sequence_column].isin(set(mic["sequence"]))].copy()
    unannotated["species"] = np.nan
    unannotated["mic_um"] = np.nan
    unannotated["mic_censored"] = np.nan

    frame = pd.concat([annotated, unannotated], ignore_index=True, sort=False)
    frame = frame.merge(hc50, on=sequence_column, how="left")
    if "hc50_um" not in frame.columns:
        frame["hc50_um"] = np.nan
    if "hc50_source" not in frame.columns:
        frame["hc50_source"] = np.nan
    return frame.reset_index(drop=True)
