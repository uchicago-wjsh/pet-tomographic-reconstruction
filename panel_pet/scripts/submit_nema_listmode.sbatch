#!/bin/bash
#SBATCH --job-name=lm_nema
#SBATCH --array=1-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lmnema_%A_%a.out
#SBATCH --error=logs/lmnema_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from panel_pet}"
source ../scripts/rcc_env.sh
mkdir -p logs results/listmode_nema

PYTHON="${CONDA_PREFIX:?Activate the reconstruction environment}/bin/python"
CTRS=(40 80 120)
CTR=${CTRS[$((SLURM_ARRAY_TASK_ID - 1))]}
# The ROOT set MUST be complete. --total-seconds 600 asserts what the whole
# directory represents, so a short set would silently yield a listmode labelled
# 600 s that actually holds less -- every count-rate and SNR number derived from
# it would then be wrong with nothing to flag it. Job 53354379 lost 7 of 60
# tasks to a slow node, which is exactly how that happens.
root_dir="${OUTDIR:-$PET_ROOT_DIR/nema}"
shopt -s nullglob
chunks=("$root_dir"/*.root)
NROOT=${#chunks[@]}
if [[ "$NROOT" -ne 600 ]]; then
    echo "ERROR: $root_dir has $NROOT .root files, expected 600." >&2
    echo "       Re-run the missing GATE array tasks before building listmode." >&2
    exit 1
fi
echo "$root_dir: $NROOT/600 chunks present"

OUT="results/listmode_nema/lm_nema_600s_ctr${CTR}ps.npz"

echo "[$(date)] NEMA box, 600 s @ CTR ${CTR} ps -> $OUT"
if [[ -s "$OUT" ]]; then echo "  exists, skipping"; exit 0; fi

"$PYTHON" reconstruction/make_listmode.py \
    --root-dir "$root_dir" --total-seconds 600 --seconds 600 \
    --ctr-ps "$CTR" --energy-window 459.9 562.1 \
    --out "$OUT"
echo "[$(date)] done"
