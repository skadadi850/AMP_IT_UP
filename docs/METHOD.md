# Method

> Fill this in as you build. It is a **submission requirement**: the abstract,
> training-data disclosure, filters applied, and selection procedure are all
> required for benchmark participation, and the LLM-assistance disclosure is
> required by the NeurIPS Main Track Handbook.

## Abstract

_(150-250 words. What the model is, what it is conditioned on, how the top-100
was selected.)_

## Training data

See [../data/README.md](../data/README.md) for the full provenance table.

- **Generator corpus:** _(sources, record counts, filters)_
- **Predictor corpus:** _(EV positives/negatives, synthetic negatives, counts)_
- **Splits:** MMseqs2 clustering at 40% identity, cluster-disjoint.

## Generative model

- Architecture:
- Representation:
- Conditioning:
- Sampler:
- Checkpoint: `checkpoints/`

## Surrogate predictors

- Members of the ensemble:
- Calibration method:
- Held-out performance (clustered split): FPR, Prec@100, AUPRC

## Filters applied to the library

1. Alphabet restricted to the 20 standard residues; length 8-50.
2. Deduplication.
3. Exact-match exclusion against `data/reference/antibacterial.fasta`.
4. _(physicochemical windows, synthesizability, etc.)_

## Top-100 selection procedure

_(This is explicitly requested by the organizers. Describe the objective, the
uncertainty estimate, the diversity constraint, and the novelty screen.)_

- Objective:
- Uncertainty:
- Diversity constraint:
- Novelty screen: Levenshtein ratio <= 0.80 vs all reference sequences, plus
  MMseqs2 identity <= 80% vs the MarLys AMP database.

## Manual interventions

_(State plainly. "None" is a valid and strong answer -- the competition is
partly about separating model quality from human curation.)_

## Reproducibility

Fixed seed 42. `uv sync && uv run generate` produces byte-identical output.
Verified on: _(OS, Python, GPU)_

## Use of AI assistants

_(Required disclosure. State which tools were used and for what.)_
