# Rebuilding the harmonized activity table

`activity_harmonized.csv` is **not redistributed in this repository.** It
aggregates sources whose redistribution terms differ and, in two cases, are not
confirmed — including HemoPI-2 data released under GPL-3.0, which cannot be
shipped under this repository's MIT licence. Every upstream source is public,
so nothing here is withheld: what follows is the exact path to rebuild the
table, plus a checksum so a rebuild can be compared byte-for-byte against the
one the models were trained on.

## What the table is

The harmonized activity table is the output of `scripts/build_dataset.py`. It
carries one row per (peptide, species, endpoint) measurement with the columns
`sequence, species, strain, endpoint, value_um, censored, source`. Replicate
measurements are collapsed upstream by geometric mean in log10 space, so a
peptide assayed repeatedly against one species contributes a single row.

It feeds two things: the MIC arm trains the XGBoost oracle
(`checkpoint/mic_regressor.json`), and both arms supply the generator's
`log_mic` and `log_hc50` conditioning axes.

## Upstream sources

Place each download at the path in the third column, then run the rebuild
command below. Retrieval dates are the dates these files were fetched for the
shipped models.

| Source | Identifier | Place at | Retrieved | Terms |
|---|---|---|---|---|
| **MLAMP** (positives) | `doi.org/10.17632/w4hb5grjwb.3` | `data/raw/mlamp_positives.csv` | 2026-09-12 | CC0 |
| **GRAMPA** (MIC) | Witten & Witten, aggregates DBAASP / DRAMP / APD / DADP | `data/raw/grampa.csv` | 2026-09-12 | see note below |
| **HemoPI-2 measured** (HC50) | `doi.org/10.5281/zenodo.14676712`, repo HEAD 2026-07-13 | `data/raw/hemopi2_measured.csv` | 2026-09-12 | **GPL-3.0** |
| **HemoPI-2 predicted** (HC50) | generated, see below | `data/raw/hemopi2_predicted.csv` | 2026-09-12 | model output |

**GRAMPA is an aggregation** and its constituent terms are not uniform. Of the
41,500 rows surviving the ingest filter: DBAASP 32,880 (CC BY 4.0), DRAMP 4,459
(**terms not confirmed**), APD 3,554 (**terms not confirmed**), DADP 607
(**terms not confirmed**). This is the reason the harmonized table is not
redistributed even with the HemoPI-2 rows removed — see
`src/ampx/data/sources.py`, where each is recorded with its verification state.

**The predicted HC50 column is model output, not HemoPI-2's data.** It is
produced by running HemoPI-2's own `hemopi2_regression.py` over all 39,448
MLAMP positives. Program output is not covered by the program's licence, but
it is also not measurement: see the disclosure in the Data disclosure section of the top-level `README.md` on what the
`log_hc50` axis actually steers toward, and on the `lenchk()` truncation that
affects the 1,367 peptides longer than 40 residues.

## Rebuild command

From the repository root, with the four files placed as above:

```bash
uv run python scripts/build_dataset.py \
  --positives data/raw/mlamp_positives.csv \
  --sources   data/activity_sources.json \
  --output    data/processed
```

`data/activity_sources.json` **is** committed: it is our own specification,
containing the per-source column mappings and row filters and no third-party
data. It is what makes this rebuild reproducible rather than approximate. The
MIC filter it encodes (`datasource_has_modifications: true`,
`has_unusual_modification: false`) is deliberate and is discussed in
the Data disclosure section of the top-level `README.md`.

## Expected output, so the rebuild is checkable

`scripts/build_dataset.py` writes `activity_harmonized.csv` alongside
`metadata.csv`. The harmonized table the shipped models were trained on has:

| Partition | Rows |
|---|---:|
| Total | 80,473 |
| `endpoint == "mic"` | 39,117 |
| `endpoint == "hc50"` | 41,356 |
| `source == "grampa"` | 39,117 |
| `source == "hemopi2_measured"` | 1,908 |
| `source == "hemopi2_predicted"` | 39,448 |

Unique peptides in the MIC arm: 5,602, across 659 species. The row-to-peptide
ratio is why the regressor's split is clustered on sequence rather than drawn
at random.

## Checksum of the table we trained on

```
sha256  c1e5b83425291cb7930d88a044e4b6846c60d037f35993c6ae5fa0043f606c5d
bytes   5511497
lines   80474   (80,473 rows plus header)
```

Verify a rebuild with:

```bash
sha256sum data/processed/activity_harmonized.csv
```

A byte-identical result confirms the rebuild reproduces our training input
exactly. A mismatch with matching row counts most likely means an upstream
database has been revised since the retrieval dates above, which is worth
knowing and is the reason those dates are recorded.
