#!/bin/bash
#SBATCH -A biochem
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mail-user=skadadi@purdue.edu
#SBATCH --mail-type=FAIL
#SBATCH --job-name=amp_restrat
#SBATCH --output=logs/%x_%j.out

# Re-score the generated library and re-select the top 100 stratified by
# length. CPU-only and no GPU cycle: score_candidates uses only the oracle
# (XGBoost plus the ESM-2 encoder), never the 58M-parameter generator or the
# 128-step sampler, so this runs on the cpu partition and leaves the ai queue
# free for the organizers' validator.
#
# The gate job persisted only FASTA, so every per-candidate score it computed
# was discarded. This run writes the full table -- per-species predicted MIC
# across all nine panel species, panel score, eligibility, length and band --
# so no future re-selection has to recompute anything.
#
# Threads are capped at 16 rather than the 32 requested. ESM-2 on CPU scaled
# INVERSELY past 16 threads when measured on this cluster (32 threads was
# slower than 16, 2 seq/s against 5), so the extra cores are requested for
# rapidfuzz's novelty screen, which does parallelise, and deliberately not
# handed to torch.

starts=$(date +"%s")
set -euo pipefail

module load conda

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

echo "[cfg] REPO=$REPO"
echo "[cfg] cpus-per-task=${SLURM_CPUS_PER_TASK:-?}  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "[cfg] library=run_a/library.fasta (the determinism-gated library, 16748221)"
md5sum run_a/library.fasta

mkdir -p results/selection logs

uv run python scripts/stratified_reselect.py \
  --library "$REPO/run_a/library.fasta" \
  --reference "$REPO/data/antibacterial.fasta" \
  --scores-out "$REPO/results/selection/candidate_scores.csv" \
  --top-out "$REPO/results/selection/top_stratified.fasta" \
  --overflow-out "$REPO/results/selection/top500_stratified.fasta" \
  --k 100 --overflow-k 500 --device cpu

ends=$(date +"%s"); diff_s=$((ends-starts))
printf "\n\telapsed %02d:%02d:%02d\n" $((diff_s/3600)) $(((diff_s%3600)/60)) $((diff_s%60))
sacct --jobs=$SLURM_JOBID --format=jobid,jobname,state,exitcode,maxrss,elapsed
