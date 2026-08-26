"""Source registry -- **this is the file you edit when you add data.**

For each database you download, add one `Source` entry describing where it
lives under `data/raw/` and how to read it. Nothing else in the codebase needs
to change; `scripts/build_dataset.py` walks this registry.

Fill in the `license` and `accessed` fields as you go. They are copied straight
into the provenance table, and full training-data disclosure is required for
co-authorship eligibility. Misrepresenting provenance is disqualifying, so
treat these fields as part of the submission, not bookkeeping.

## Roles, and why the distinction matters

    'positive'   experimentally active AMP
    'negative'   experimentally inactive peptide (rare and precious -- there
                 are under ~1000 of these in DBAASP)
    'general'    functional peptide of unknown antimicrobial status. NOT a
                 negative. Used to train the *representation*, and as a source
                 of hard negatives (signal/metabolic peptides) for the
                 classifier.
    'reference'  the organizers' exclusion set. Never a training positive by
                 default -- see the note in data/README.md.

Conflating 'general' with 'negative' is the most common data bug in this field:
it teaches the classifier that "not in an AMP database" means "inactive", which
is false for most of UniProt and guarantees inflated test performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Source:
    #: Short identifier. Also the folder name under data/raw/.
    id: str
    #: Human-readable name for the provenance table.
    name: str
    #: 'fasta' or 'table'.
    kind: str
    #: Default role for records from this source.
    role: str
    #: License string, e.g. 'CC BY 4.0'. Required for disclosure.
    license: str = "TODO"
    #: Version or access date, e.g. '2026-08-25'. Required for disclosure.
    accessed: str = "TODO"
    #: Where you got it, for the provenance table.
    url: str = ""
    #: Glob for the files inside data/raw/<id>/. Default: everything.
    glob: str = "*"

    # --- table sources only ---
    #: Column holding the sequence. Case-sensitive; check your header row.
    seq_col: str | None = None
    #: Column holding an activity label, if any.
    label_col: str | None = None
    #: Values in `label_col` meaning active.
    positive_values: tuple[str, ...] = ("1", "yes", "true", "active", "amp")
    #: Column holding MIC.
    mic_col: str | None = None
    #: Column holding strain/species.
    strain_col: str | None = None
    #: Column holding the source's own accession.
    id_col: str | None = None

    #: Free-text notes: filters you applied by hand, caveats, subsets taken.
    notes: str = ""


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
#
# The entries below are scaffolded with the column names these databases
# *typically* use. VERIFY THEM against your actual download -- run
#   uv run python scripts/build_dataset.py --inspect <source_id>
# to print the real header row and a few sample rows before trusting anything.
# Column names change between releases.

SOURCES: list[Source] = [
    Source(
        id="marlys",
        name="MarLys AMP database (MLAMP)",
        kind="fasta",
        role="positive",
        license="CC0",
        url="https://doi.org/10.17632/w4hb5grjwb.3",
        notes="Aggregates ~102k sequences from 13 primary AMP databases. "
              "The organizers' 80% identity ceiling is defined against this set, "
              "so it is simultaneously your best positive set and your novelty "
              "constraint. See PLAN.md 4.1.",
    ),
    Source(
        id="dbaasp",
        name="DBAASP v3",
        kind="table",
        role="positive",
        license="CC BY 4.0",
        url="https://dbaasp.org/download",
        seq_col="SEQUENCE",
        mic_col="MIC",
        strain_col="TARGET_SPECIES",
        id_col="ID",
        notes="The only large source of per-strain MIC values, and therefore the "
              "only way to train against what Phase 2 actually measures. Filter "
              "to standard residues and free N/C termini. Standardise medium and "
              "CFU as OmegAMP does (their Sec. C).",
    ),
    Source(
        id="dramp",
        name="DRAMP 3.0",
        kind="table",
        role="positive",
        license="TODO: verify academic terms",
        url="http://dramp.cpu-bioinfor.org/downloads/",
        seq_col="Sequence",
        id_col="DRAMP_ID",
    ),
    Source(
        id="dbamp",
        name="dbAMP 3.0",
        kind="table",
        role="positive",
        license="custom, free for academic use",
        url="https://awi.cuhk.edu.cn/dbAMP/",
        seq_col="Sequence",
        id_col="dbAMP_ID",
    ),
    Source(
        id="ampsphere",
        name="AMPSphere",
        kind="fasta",
        role="general",
        license="CC BY 4.0",
        url="https://ampsphere.big-data-biology.org/downloads",
        notes="~1M candidate AMPs predicted from 60k+ metagenomes (gut, marine, "
              "soil). This is your breadth corpus and the compliant substitute "
              "for raw peptidomics mining -- PLAN.md 4.2. Role is 'general' "
              "because these are *predictions*, not experimental positives. "
              "Do not label them active.",
    ),
    Source(
        id="peptipedia",
        name="Peptipedia v2.0",
        kind="table",
        role="general",
        license="ODbl",
        url="https://app.peptipedia.cl/",
        seq_col="sequence",
        label_col="activity",
        notes="Two distinct uses: (1) the ~774k general peptide corpus that "
              "OmegAMP's ablation showed improves generator quality AND "
              "controllability; (2) signal peptides and metabolic peptides as "
              "hard negatives for the classifier. Export those two subsets "
              "separately into peptipedia_signal/ and peptipedia_metabolic/.",
    ),
    Source(
        id="amplify_neg",
        name="AMPlify non-AMP set",
        kind="fasta",
        role="negative",
        license="TODO: verify",
        url="https://github.com/bcgsc/AMPlify",
        notes="~128k curated non-AMPs. Real negatives, distinct from 'general'.",
    ),
]

SOURCES_BY_ID = {s.id: s for s in SOURCES}
