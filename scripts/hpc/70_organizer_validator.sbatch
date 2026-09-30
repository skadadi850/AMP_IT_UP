#!/bin/bash
#SBATCH -A kazemian
#SBATCH -N 1
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -t 04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=14
#SBATCH --mail-user=skadadi@purdue.edu
#SBATCH --mail-type=FAIL
#SBATCH --job-name=amp_validator
#SBATCH --output=logs/%x_%j.out

# The organizers' own verify_submission.py, run in its URL-taking form against
# the pushed repository. This is their exact check and the last gate we do not
# control: it clones the public repo with no credentials, uv syncs it, runs the
# entry point, checks alphabet / length / duplicates / overlap / similarity,
# then runs the entry point A SECOND TIME and byte-compares. It is therefore a
# full-scale determinism gate in its own right, not just a compliance pass.
#
# Walltime is 4h rather than 3h deliberately. Two 50,000-sequence generations
# at ~1h28m of sampling each, plus dependency resolution, a 225 MB weight
# download and ESM-2 scoring of 100,000 candidates, leaves no slack in three
# hours; job 16736825 was already killed once at a 3h limit with run B at 80%.
# A walltime kill at 95% costs a queue cycle we cannot spare this close to the
# deadline.
#
# Nothing here reads the working tree. The repository under test is fetched
# fresh from GitHub, which is the point: it validates what the organizers will
# actually clone, not what happens to be on this filesystem.

starts=$(date +"%s")
start=$(date +"%r, %m-%d-%Y")

set -euo pipefail

module load conda

SUBMISSION_URL="${1:-https://github.com/skadadi850/AMP_IT_UP}"
TEMPLATE_URL="https://github.com/szczurek-lab/amp-challenge-2027.git"

# Node-local scratch. The login node's /tmp is not visible here, so the
# organizers' script is cloned fresh rather than copied from a previous
# session's scratch directory.
WORK="${TMPDIR:-/tmp}/validator_${SLURM_JOB_ID}"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

export TORCHINDUCTOR_CACHE_DIR="$WORK/inductor"
export TRITON_CACHE_DIR="$WORK/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "[cfg] SUBMISSION_URL=$SUBMISSION_URL"
echo "[cfg] WORK=$WORK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo
echo "== fetching the organizers' validator =="
git clone -q --depth 1 "$TEMPLATE_URL" "$WORK/template"

# Attribution: a pass or a failure is a statement about a SPECIFIC version of
# the organizers' checker. Cloning fresh means we validate against what they
# ship today rather than a copy taken earlier, so if they have patched the
# validator since, this job finds out -- but only if the log says which
# version answered. Record the commit, its date, and the checker's own digest.
echo "template repo   : $TEMPLATE_URL"
echo "template HEAD   : $(git -C "$WORK/template" rev-parse HEAD)"
echo "template date   : $(git -C "$WORK/template" log -1 --format=%cd --date=iso)"
echo "template subject: $(git -C "$WORK/template" log -1 --format=%s)"
echo "verify_submission.py sha256: $(sha256sum "$WORK/template/scripts/verify_submission.py" | cut -d' ' -f1)"
echo "verify_submission.py last changed in: $(git -C "$WORK/template" log -1 --format='%h %cd' --date=short -- scripts/verify_submission.py)"

echo
echo "== running verify_submission.py in URL form =="
echo "   (clones, syncs, generates, checks, regenerates, byte-compares)"
cd "$WORK/template"

# Their own environment, not ours: the validator is a two-dependency project
# and running it under the submission's environment would not be their check.
uv sync

set +e
uv run python scripts/verify_submission.py "$SUBMISSION_URL" \
  --dir "$WORK/submission" \
  --antibacterial-fasta "$WORK/template/data/antibacterial.fasta"
rc=$?
set -e

echo
if [[ "$rc" -eq 0 ]]; then
  echo "== VALIDATOR PASSED (exit 0) =="
else
  echo "== VALIDATOR FAILED (exit $rc) =="
fi

# Surface the generated artifacts' shape even on failure, since a failure
# after generation is far more informative with the counts in the log.
if [[ -d "$WORK/submission/generate" ]]; then
  for f in library.fasta top.fasta; do
    p="$WORK/submission/generate/$f"
    [[ -f "$p" ]] && echo "  $f: $(grep -c '^>' "$p") sequences, sha256 $(sha256sum "$p" | cut -c1-16)"
  done
fi

ends=$(date +"%s")
end=$(date +"%r, %m-%d-%Y")
diff_s=$((ends-starts))
printf "\n\t===========Time Stamp===========\n"
printf "\tStart\t:$start\n\tEnd\t:$end\n\tTime\t:%02d:%02d:%02d\n" \
  $((diff_s/3600)) $(((diff_s%3600)/60)) $((diff_s%60))
printf "\t================================\n\n"
sacct --jobs=$SLURM_JOBID --format=jobid,jobname,qos,nnodes,ncpu,maxrss,cputime,avecpu,elapsed

exit "$rc"
