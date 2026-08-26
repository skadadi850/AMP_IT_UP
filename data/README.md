# Data

## Where to put what

```
data/
├── raw/                    ← YOUR DOWNLOADS GO HERE. Never edited by code.
│   ├── marlys/             one folder per source id from src/ampx/data/sources.py
│   ├── dbaasp/
│   ├── dramp/
│   ├── dbamp/
│   ├── ampsphere/
│   ├── peptipedia/
│   ├── peptipedia_signal/
│   ├── peptipedia_metabolic/
│   └── amplify_neg/
├── interim/                scratch (mmseqs work dirs). Disposable.
├── processed/              generated. Delete and rebuild any time.
│   ├── peptides.csv        canonical table, one row per (sequence, source)
│   ├── negatives.csv       synthetic negatives
│   ├── splits.csv          cluster-disjoint train/val/test
│   └── manifest.md         provenance, generated for disclosure
└── reference/
    └── antibacterial.fasta the organizers' exclusion set (39,448 seqs)
```

`raw/`, `interim/` and `processed/` are gitignored -- they are large and
regenerable. `reference/` is tracked, because `generate.py` needs it at
inference time and the organizers run from a fresh clone.

**The rule: `raw/` is append-only and never modified in place.** Every
transformation happens in code and lands in `processed/`. If you hand-edit a
raw file you lose the ability to reproduce your own dataset, and reproducibility
is a submission requirement.

## Adding a source

1. Make the folder: `mkdir -p data/raw/<source_id>/`
2. Drop the download in. Gzip is fine, no need to decompress.
3. Look at what you actually got:
   `uv run python scripts/build_dataset.py --inspect <source_id>`
4. Edit **`src/ampx/data/sources.py`** -- set `seq_col`, `mic_col`, `license`,
   `accessed`. This is the only file you edit to add data.
5. Rebuild: `uv run python scripts/build_dataset.py`
6. Check the keep-rate in the report. A low rate means wrong column names.

## Provenance

`processed/manifest.md` is generated on every build. Full training-data
disclosure is required for co-authorship eligibility, and misrepresenting
provenance is grounds for disqualification. Fill in every `license` and
`accessed` field in `sources.py` as you download, not at the end.

Any non-public data you use must be released publicly under a permissive
license at submission time.

## A note on the reference set

`reference/antibacterial.fasta` is 39,448 unique sequences, all within 8-50
residues, with charge / disulfide / source-database / activity metadata in the
FASTA headers -- parse the headers, they are useful.

It serves two conflicting purposes, and this tension is worth being deliberate
about: it is a high-quality curated AMP set (good training data) *and* it
defines your novelty constraint (the top-100 must stay below 80% identity to
all of it). Training hard on it pushes generations toward exactly the sequences
you must then stay away from. Options: train on it but sample with novelty
pressure; or hold it out of the generator's corpus and use the other positive
sources. Decide explicitly and record the decision in `docs/METHOD.md`.

## Unit hazard

MIC values arrive in both µg/mL and µM depending on source and row. `parse_mic`
records the number it found but **only converts µM to µg/mL if you pass the
molecular weight**. Mixing units silently is the kind of bug that produces a
plausible-looking model that ranks peptides wrongly. Either normalise to one
unit with computed MW, or carry a unit column and never compare across it.

Related: MIC values like `>128` are **right-censored, not missing and not 128**.
`peptides.csv` keeps a `mic_censored` flag. Use interval/Tobit regression rather
than dropping or naively imputing them -- censored rows are your strongest
inactivity signal.
