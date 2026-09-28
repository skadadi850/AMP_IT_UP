# CMD_AMP

This repo contains our implementation of Conditioned Masked Discrete Diffusion for Antimicrobial Peptide Design (AMP Challenge - NeurIPS 2026 Competition Track).

Generative design of linear antimicrobial peptides for the
[AMP Challenge](https://szczurek-lab.github.io/amp-challenge-website/)
(NeurIPS 2026 Competition Track).

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
| `--kappa` | `1.5` | Risk aversion in top-K selection |
| `--max-per-cluster` | `4` | Cap on candidates per structural cluster |
| `--skip-verify` | off | Skip similarity scan (development only) |

## Validating before you submit

```bash
# Structural + similarity checks, run locally
uv run generate

# Full organizer validation against your pushed repo
uv run python scripts/verify_submission.py https://github.com/<you>/<repo>
```

## Layout

```
src/ampx/
  generate.py      entry point; deterministic sampling + ranking
  compliance.py    hard-constraint gate (alphabet, length, novelty ceiling)
  ranking.py       risk-aware top-100 selection
  featurize.py     descriptor computation for the surrogate ensemble
  predictor.py     calibrated activity / MIC surrogates
  models/          generator architectures
scripts/           dataset build, training, evaluation
checkpoints/       trained weights (git-lfs)
data/reference/    organizers' exclusion set
docs/METHOD.md     method abstract + data disclosure
```

## Method

See [docs/METHOD.md](docs/METHOD.md).

## License

MIT. See [LICENSE](LICENSE).
