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
    license: str = "terms not confirmed"
    #: Version or access date, e.g. '2026-08-25'. Required for disclosure.
    accessed: str = "not retrieved"
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
        accessed="2026-09-12",
        url="https://doi.org/10.17632/w4hb5grjwb.3",
        notes="USED BY THE SHIPPED MODELS. Aggregates ~102k sequences from 13 "
              "primary AMP databases. The positive class in BOTH generator "
              "stages: 39,448 rows at label==1, plus 723 at label==0 tiered "
              "non_amp_function which are excluded from the positive class. "
              "The organizers' 80% identity ceiling is defined against this "
              "set -- the 39,448 are byte-for-byte identical to "
              "data/antibacterial.fasta -- so the generator was trained on the "
              "set its output must stay 80% distant from. That decision is "
              "declared in docs/METHOD.md rather than left to be inferred. "
              "See PLAN.md 4.1.",
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
        license="terms not confirmed; surveyed, not used by the shipped models",
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
        license="terms not confirmed; surveyed, not used by the shipped models",
        url="https://github.com/bcgsc/AMPlify",
        notes="~128k curated non-AMPs. Real negatives, distinct from 'general'.",
    ),

    # ------------------------------------------------------------------
    # Sources actually used by the shipped models.
    #
    # The entries above describe databases surveyed during dataset design.
    # The five below are the ones that supply every label the submitted
    # library depends on, and they are recorded separately because that
    # distinction was not obvious from this file before: a reader could
    # reasonably have concluded that DBAASP or Peptipedia were trained on
    # directly, which they were not.
    #
    # Where terms could not be confirmed the field says so explicitly.
    # "terms not confirmed" is a statement of fact about what was checked;
    # a blank or an optimistic guess would not be.
    # ------------------------------------------------------------------
    Source(
        id="grampa",
        name="GRAMPA (Giant Repository of AMP Activities)",
        kind="table",
        role="positive",
        license="aggregation; constituent terms differ, see notes",
        accessed="2026-09-12",
        url="https://github.com/zswitten/Antimicrobial-Peptides",
        seq_col="sequence",
        notes="Supplies 100% of the MIC labels: 39,117 rows over 5,602 unique "
              "peptides and 659 species. An AGGREGATION, and its constituents "
              "do not share terms. Of the 41,500 rows surviving the ingest "
              "filter: DBAASP 32,880 (CC BY 4.0), DRAMP 4,459 (TERMS NOT "
              "CONFIRMED), APD 3,554 (TERMS NOT CONFIRMED), DADP 607 (TERMS "
              "NOT CONFIRMED). This unresolved mix is why the harmonized "
              "activity table is rebuilt rather than redistributed; see "
              "data/processed/README.md. Ingest keeps modified peptides "
              "(datasource_has_modifications=true) while excluding unusual "
              "modifications, so 37.4% of the MIC rows are C-terminally "
              "amidated -- a disclosed bias relative to our free-termini "
              "linear designs, see docs/METHOD.md.",
    ),
    Source(
        id="hemopi2",
        name="HemoPI-2 (Rathore et al., Commun Biol 8:176, 2025)",
        kind="table",
        role="general",
        license="GPL-3.0 -- NOT redistributable under this repository's MIT",
        accessed="2026-09-12",
        url="https://doi.org/10.5281/zenodo.14676712",
        seq_col="SEQUENCE",
        notes="Two distinct uses, with different licence consequences. "
              "(1) MEASURED HC50: 1,908 rows taken from the repository's own "
              "Dataset/ files (1,524 from cross_val_dataset.csv, 380 from "
              "independent_dataset.csv). This is their data under GPL-3.0 and "
              "is NOT committed here. (2) PREDICTED HC50: 39,448 values we "
              "generated by running their hemopi2_regression.py over the MLAMP "
              "positives. Program output is not covered by the program's "
              "licence. Their lenchk() truncates sequences over 40 residues "
              "before featurization, so 1,367 of those predictions (3.47%) are "
              "computed on a truncated peptide. Repo HEAD 2b67a5c, 2026-07-13.",
    ),
    Source(
        id="sorfdb",
        name="sORFdb (Zenodo record 10688271)",
        kind="fasta",
        role="general",
        license="see Zenodo record; terms not confirmed",
        accessed="2026-08-30",
        url="https://doi.org/10.5281/zenodo.10688271",
        notes="The sole source of the 269,824 general small proteins "
              "that form the generator pretrain's negative class. Real "
              "translated open reading frames, NOT shuffled or mutated "
              "decoys. All five files MD5-verified against the Zenodo record; "
              "protein FASTA holds 34,007,166 records before filtering to "
              "8-50 residues. Role is 'general', not 'negative': absence from "
              "an AMP database is not evidence of inactivity.",
    ),
    Source(
        id="smprot_v2",
        name="SmProt v2",
        kind="fasta",
        role="unused",
        license="terms not confirmed",
        accessed="2026-08-30",
        url="http://bigdata.ibp.ac.cn/SmProt/",
        notes="OBTAINED BUT NEVER USED. Downloaded and staged on 2026-08-30 "
              "in anticipation of pooling it with sORFdb into the pretrain "
              "general-peptide class; the pooling step was never implemented "
              "and no SmProt sequence entered any corpus. The negatives build "
              "(02_negatives.sbatch) reads sORFdb alone, and its record count "
              "closes on sORFdb alone: 318,268 length-filtered minus 48,435 "
              "AMP-like equals the 269,833 in negatives.fasta. Retained in "
              "this registry so the disclosure records what was acquired, not "
              "only what was used. See docs/METHOD.md, 'SmProt v2 was "
              "obtained but never used'.",
    ),
]

SOURCES_BY_ID = {s.id: s for s in SOURCES}
