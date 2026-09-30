# CMD_AMP

This repo contains our implementation of Conditioned Masked Discrete Diffusion for Antimicrobial Peptide Design (AMP Challenge - NeurIPS 2026 Competition Track).

Generative design of linear antimicrobial peptides for the
[AMP Challenge](https://szczurek-lab.github.io/amp-challenge-website/)
(NeurIPS 2026 Competition Track).

## Abstract

Antimicrobial peptides (AMPs) have emerged as promising therapeutic candidates, with advances in computational approaches and high-throughput screening accelerating their discovery and optimization. However, designing effective de novo AMPs requires addressing potency, selectivity, and toxicity. Existing generative approaches often rely on post-hoc filtering, which selects candidates only after generation, or indirect conditioning through latent representations from protein language models (PLMs), which may limit simultaneous control over these design objectives. Here, we introduce CMD-AMP, a masked discrete diffusion model conditioned on structural and functional features to directly generate AMP sequences. To explicitly learn amphipathic structural organization, CMD-AMP masks residues on the same face of an α-helix, directing the model to reconstruct these membrane-interacting surfaces. CMD-AMP further incorporates sequence-derived features as well as experimental data, including MIC and hemolytic activity, using observation masking. Across 50,000 generated peptides, no sequence exceeds 80% identity to any of 39,448 antibacterial peptides used for training, and a held-out MIC regressor predicts potency comparable to that of known AMPs. These results indicate that CMD-AMP can be used for de novo AMP generation for concurrent design constraints. 


## Quick start

```bash
uv sync
uv run generate
```

This writes:

```
generate/library.fasta   50,000 designed peptides
generate/top.fasta       100 ranked candidates
```

Both files are byte-identical across runs (fixed seed, default 42).

## Options

| Flag | Default | Description |
|---|---|---|
| `--n-sequences` | `50000` | Library size |
| `--top-k` | `100` | Ranked candidates |
| `--seed` | `42` | Random seed |
| `--length` | `50` | Maximum peptide length in residues. Template flag; the competition ceiling and the default are both 50 |
| `--device` | auto | `cuda`, `cpu`, or omitted to resolve automatically |
| `--identity-ceiling` | `0.75` | Maximum Levenshtein ratio to any reference sequence. The organizers' validator fails above `0.80`; the default leaves margin |
| `--overflow-k` | `500` | Length of the ranked overflow list written to `docs/`. The organizers replace an invalid top-100 entry with the next valid candidate, so the ordering past 100 is used |
| `--out` | entry-point name | Output directory |
| `--kappa` | `1.5` | **Inert.** Risk aversion in the LCB ranking. The oracle is a single model, so `score_std` is 0 and `lcb` collapses to the mean; changing this flag does not change the output |

## Validating before you submit

```bash
# Structural + similarity checks, run locally
uv run generate

# Full organizer validation against your pushed repo
uv run python scripts/verify_submission.py https://github.com/skadadi850/AMP_IT_UP
```

## Layout

```
src/ampx/
  generate.py      entry point; deterministic sampling + ranking
  compliance.py    hard-constraint gate (alphabet, length, novelty ceiling)
  ranking.py       risk-aware top-100 selection
  predictor.py     calibrated activity / MIC surrogates
  models/          generator architectures
scripts/           dataset build, training, evaluation
checkpoint/        trained weights, fetched from the tagged release
data/              organizers' exclusion set (antibacterial.fasta)
docs/METHOD.md     method abstract + data disclosure
```

## Method

See [docs/METHOD.md](docs/METHOD.md).

## License

MIT. See [LICENSE](LICENSE).
