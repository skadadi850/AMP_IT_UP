# Scripts

| Script | Purpose |
|---|---|
| `verify_submission.py` | The organizers' official validator, vendored unchanged. Run against your **pushed public repo** URL. |
| `build_dataset.py` | Assemble and clean the training corpora; write provenance. |
| `cluster_split.py` | MMseqs2 clustering at 40% identity, cluster-disjoint splits. |
| `make_negatives.py` | Random / shuffled / mutated / add-delete synthetic negatives. |
| `train_predictor.py` | Fit and calibrate the surrogate ensemble. |
| `train_generator.py` | Train the conditional generator. |
| `eval_seqme.py` | Score a library with `seqme` against the baselines. |

Usage:

```bash
uv run python scripts/verify_submission.py https://github.com/<you>/<repo>
```
