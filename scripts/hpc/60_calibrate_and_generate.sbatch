#!/bin/bash
#SBATCH -A kazemian
#SBATCH -N 1
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -t 05:00:00
#SBATCH --requeue
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=14
#SBATCH --mail-user=skadadi@purdue.edu
#SBATCH --mail-type=FAIL
#SBATCH --job-name=amp_gate
#SBATCH --output=logs/%x_%j.out

# Phase 3.5 determinism gate, with the stage-5b pad-bias calibration folded in
# ahead of it so this is one queue entry rather than three. The `ai` partition
# is small and heavily oversubscribed, so a second queue entry can cost more
# wall clock than the job itself.
#
# Stages, in order:
#   1. pad-bias calibration grid against the SHIPPED checkpoint
#      (skipped with --skip-calibration; see the note on that flag below)
#   2. STOP if --calibrate-only, so the grid can be reviewed before any
#      library is generated
#   3. full generation, twice, into separate directories
#   4. byte-diff of the two runs -- the gate, and this job's exit condition
#   5. the organizers' validator on the first run's output
#
# Any stage failing fails the job: `set -euo pipefail` plus explicit exits.
#
# Walltime is 5h, raised from 3h. Job 16736825 was cancelled at the 3h limit
# with run B 126/157 batches in: sampling alone is ~1h28m per run, so two runs
# plus calibration, environment sync, weight fetch and oracle scoring of
# 50,000 sequences does not fit. Keep this close to real runtime -- the `ai`
# queue is deep and padded requests backfill badly -- but 3h was under it.

starts=$(date +"%s")
start=$(date +"%r, %m-%d-%Y")

set -euo pipefail

module load conda
export HDF5_USE_FILE_LOCKING=FALSE

# Defined inline rather than sourced from scripts/hpc/env.sh: that file lives
# in the research repository and is deliberately not part of this submission,
# and both are overridable so this script is not pinned to one account.
export D="${D:-/home/skadadi/scratch/Antimicrobial_Peptide_Data}"
# sbatch runs a spooled copy of this file, so BASH_SOURCE is not the repo.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"

# A cache directory shared across concurrent jobs corrupts (interleaved
# autotune best_config JSON) and kills every job with a JSONDecodeError, so
# each job autotunes independently on its own node.
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/inductor_${SLURM_JOB_ID}
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton_${SLURM_JOB_ID}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
trap 'rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"' EXIT

CALIBRATE_ONLY=0
SKIP_CALIBRATION=0
for arg in "$@"; do
  case "$arg" in
    --calibrate-only)    CALIBRATE_ONLY=1 ;;
    --skip-calibration)  SKIP_CALIBRATION=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
if [[ "$CALIBRATE_ONLY" -eq 1 && "$SKIP_CALIBRATION" -eq 1 ]]; then
  echo "--calibrate-only and --skip-calibration are mutually exclusive" >&2
  exit 2
fi

cd "$REPO"

# The calibration reference is the corpus the SHIPPED checkpoint was trained
# on. checkpoint/length_calibration.json was fitted against face_only/, a
# different set of weights, so its -0.2 is an assumption until re-derived
# here. Length distribution is directly scored in Phase 1, so a pad bias
# carried over from the wrong checkpoint would skew all 50,000 sequences on a
# scored axis.
#
# The grid is POSITIVE, which inverts calibrate_length.py's default. That
# default (0.0 down to -3.0) encodes the face_only checkpoint's behaviour,
# which was short-biased: median 16 against a training median of 18, so it
# needed lengthening, and pad_bias is added to the PAD logit -- negative
# suppresses PAD and lengthens. Stage-5b is the opposite. Job 16735793
# measured it at median 21 / mean 22.98 at zero bias against a target of
# 18 / 19.01, and KS rose monotonically across every negative value tested
# (0.2153 at 0.0 up to 0.4397 at -0.75). The minimum sat on the grid
# boundary, which is a truncation rather than a fit, so the optimum is at
# positive bias. Linear extrapolation of that run puts it near +0.3; the grid
# below brackets that and runs to +1.0 so the minimum is interior and can be
# seen to be one.
CKPT="$REPO/checkpoint/masked_diffusion_best.pt"
META="$D/processed/selectivity/metadata.csv"
SPLITS="$D/raw/splits_with_negatives.csv"
OUTDIR="$REPO/results/calibration"
RUN_A="$REPO/run_a"
RUN_B="$REPO/run_b"

echo "[cfg] REPO=$REPO D=$D"
echo "[cfg] CKPT=$CKPT"
echo "[cfg] META=$META SPLITS=$SPLITS"
echo "[cfg] CALIBRATE_ONLY=$CALIBRATE_ONLY SKIP_CALIBRATION=$SKIP_CALIBRATION GRID=0.0 +0.1 +0.2 +0.3 +0.4 +0.5 +0.75 +1.0"
echo "[cfg] RUN_A=$RUN_A RUN_B=$RUN_B"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

mkdir -p "$OUTDIR" logs

# uv provides the environment, not conda: the submission must run under the
# same `uv sync` the organizers use, so validating against anything else
# would not test what they will run.
echo
echo "== stage 0: sync the submission environment =="
uv sync
uv run python scripts/fetch_weights.py

echo
if [[ "$SKIP_CALIBRATION" -eq 1 ]]; then
# The fitted pad_bias no longer reaches generation: generate.py passes
# free_length=False and pad_bias=0.0 at the sampling call site, so the value in
# checkpoint/length_calibration_stage5b.json is printed for provenance and
# nothing else. Re-running the grid costs wall clock that run B needs and would
# rewrite that file, so the flag skips the stage rather than treating it as a
# cheap no-op. The file must already exist; the check below is the guard.
echo "== stage 1: SKIPPED (--skip-calibration) =="
if [[ ! -f "$REPO/checkpoint/length_calibration_stage5b.json" ]]; then
  echo "FAIL: --skip-calibration requires an existing" >&2
  echo "      checkpoint/length_calibration_stage5b.json; run without the" >&2
  echo "      flag to fit it." >&2
  exit 1
fi
echo "  using existing checkpoint/length_calibration_stage5b.json:"
uv run python -c "import json;d=json.load(open('checkpoint/length_calibration_stage5b.json'));print(f\"    pad_bias {d['best_pad_bias']:+.2f}, KS {d['ks_length_vs_train']:.4f}, reveal {d['reveal']}\")"
echo "  (pad_bias is inert under free_length=False; recorded, not applied)"
else
echo "== stage 1: pad-bias calibration grid (stage-5b checkpoint) =="
uv run python scripts/calibrate_length.py \
  --checkpoint "$CKPT" \
  --metadata "$META" \
  --splits "$SPLITS" \
  --num-samples 4000 \
  --batch-size 512 \
  --steps 128 \
  --reveal structured \
  --seed 7 \
  --grid 0.0 0.1 0.2 0.3 0.4 0.5 0.75 1.0 \
  --out "$OUTDIR/length_calibration_stage5b.csv" \
  --device cuda

# Record the fitted value next to the old one rather than overwriting it, so
# the face_only figure stays available for comparison.
uv run python - <<'PY'
import json, pandas as pd
from pathlib import Path

csv = Path("results/calibration/length_calibration_stage5b.csv")
table = pd.read_csv(csv)
# calibrate_length.py already wrote a sidecar next to the csv; extend it with
# the provenance this submission needs rather than recomputing the fit.
payload = json.loads(csv.with_suffix(".json").read_text())
payload.update({
    "calibration_reference":
        "training folds only (fold != 0), selectivity corpus",
    "checkpoint": "checkpoint/masked_diffusion_best.pt",
    "fitted_against": "stage-5b selectivity fine-tune (the shipped checkpoint)",
    "supersedes": "checkpoint/length_calibration.json, fitted on face_only/",
    "grid": [float(v) for v in table["pad_bias"]],
})
Path("checkpoint/length_calibration_stage5b.json").write_text(
    json.dumps(payload, indent=2) + "\n")

print("\n=== pad-bias grid (stage-5b) ===")
print(table.to_string(index=False))
print(f"\nfitted pad_bias = {payload['best_pad_bias']:+.2f} "
      f"(KS {payload['ks_length_vs_train']:.4f})")
print("old face_only value was -0.20; agreement here is confirmation, "
      "disagreement means the carried-over value would have skewed length.")
print("wrote checkpoint/length_calibration_stage5b.json")
PY
fi

if [[ "$CALIBRATE_ONLY" -eq 1 ]]; then
  echo
  echo "== stopping after calibration (--calibrate-only) =="
  echo "Review results/calibration/length_calibration_stage5b.csv, then"
  echo "resubmit without the flag to run the generation gate."
  ends=$(date +"%s"); echo "elapsed $((ends-starts))s"
  exit 0
fi

echo
echo "== stage 2: generation run A =="
rm -rf "$RUN_A" "$RUN_B"
uv run generate --out "$RUN_A"

echo
echo "== stage 3: generation run B (same seed, same defaults) =="
uv run generate --out "$RUN_B"

echo
echo "== stage 4: determinism gate =="
# The gate. Two runs with identical inputs must be byte-identical; this is
# exactly what the organizers check, so the job fails here rather than
# reporting success and leaving it to be discovered later.
gate_failed=0
for f in library.fasta top.fasta; do
  if diff -q "$RUN_A/$f" "$RUN_B/$f" >/dev/null; then
    echo "  DETERMINISTIC  $f  ($(grep -c '^>' "$RUN_A/$f") sequences)"
  else
    echo "  NONDETERMINISTIC  $f"
    diff "$RUN_A/$f" "$RUN_B/$f" | head -20
    gate_failed=1
  fi
done
if [[ "$gate_failed" -ne 0 ]]; then
  echo "FAIL: generation is not reproducible; not running the validator."
  exit 1
fi

echo
echo "== stage 5: organizers' validator on run A =="
# verify_submission.py clones a GitHub URL and runs the entry point there, so
# it validates the pushed repository rather than this working tree. Run it
# only when a URL is supplied; otherwise check run A's files directly, which
# is the part that does not need a remote.
if [[ -n "${SUBMISSION_URL:-}" ]]; then
  uv run python scripts/verify_submission.py "$SUBMISSION_URL" \
    --dir "${TMPDIR:-/tmp}/submission_${SLURM_JOB_ID}" \
    --antibacterial-fasta "$REPO/data/antibacterial.fasta"
else
  echo "SUBMISSION_URL unset: validating run A's output locally instead."
  echo "Set SUBMISSION_URL=<github url> to run the full clone-and-verify path."
  uv run python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from verify_submission import _read_fasta, _verify_sequences, _verify_top, \
    _verify_no_overlap, _veritfy_max_simularity
run = Path("run_a")
_, refs = _read_fasta(Path("data/antibacterial.fasta"))
refs = set(refs)
library = _verify_sequences(run / "library.fasta")
_verify_top(run / "top.fasta", library, 100)
_verify_no_overlap(library, refs)
_, top = _read_fasta(run / "top.fasta")
_veritfy_max_simularity(set(top), refs)
print("local validation passed")
PY
fi

echo
echo "== all gates passed =="

ends=$(date +"%s")
end=$(date +"%r, %m-%d-%Y")
diff_s=$(($ends-$starts))
hours=$(($diff_s / 3600)); dif=$(($diff_s % 3600))
minutes=$(($dif / 60)); seconds=$(($dif % 60))
printf "\n\t===========Time Stamp===========\n"
printf "\tStart\t:$start\n\tEnd\t:$end\n\tTime\t:%02d:%02d:%02d\n" "$hours" "$minutes" "$seconds"
printf "\t================================\n\n"
sacct --jobs=$SLURM_JOBID --format=jobid,jobname,qos,nnodes,ncpu,maxrss,cputime,avecpu,elapsed
