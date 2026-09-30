# Scripts

Build, training, calibration and analysis scripts. `uv run generate` needs none of them.

| Script | Purpose |
|---|---|
| `verify_submission.py` | The organizers' validator, vendored unchanged. Run it against the pushed repository URL: `uv run python scripts/verify_submission.py https://github.com/skadadi850/AMP_IT_UP` |
| `fetch_weights.py` | Downloads the two Release-hosted weight files and verifies them against `checkpoint/SHA256SUMS`. Thin wrapper over `ampx.weights`. |
| `build_dataset.py` | Builds the training metadata table and the harmonized activity table from the sources in `data/activity_sources.json`. Rebuild steps: `data/processed/README.md`. |
| `build_mic_table.py` | Collapses the harmonized table to one MIC row per (sequence, species) and writes the unique-sequence FASTA for the ESM-2 embedding job, in fixed order. |
| `cluster_split.py` | MMseqs2 clustering at 40% identity and cluster-disjoint train, validation and test splits. |
| `make_negatives.py` | Synthetic negatives: random, shuffled, mutated and add-delete variants. |
| `train_generator.py` | Trains the masked discrete diffusion model. |
| `train_predictor.py` | Fits the MIC regressor. |
| `generate_masked.py` | Samples from a masked diffusion checkpoint, including conditioning sweeps. |
| `calibrate_length.py` | Fits the single PAD-logit bias by minimizing the KS distance between generated and reference lengths. |
| `clean_libraries.py` | Filters generated libraries to competition scope and records attrition. Sequences are dropped, never repaired. |
| `compute_metrics.py` | Core metrics per seed and aggregated, including the paired face-versus-independent-masking test. |
| `stratified_reselect.py` | Re-scores a library and re-selects the top 100 stratified by length, for the analysis of the long-peptide skew. The submitted list comes from the entry point. |
| `eval_seqme.py` | Placeholder. Not implemented. |
| `hpc/` | SLURM job scripts. |
