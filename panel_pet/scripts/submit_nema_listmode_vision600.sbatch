#!/bin/bash
#SBATCH --job-name=lm_nema_v600
#SBATCH --array=1-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lmnemav600_%A_%a.out
#SBATCH --error=logs/lmnemav600_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from panel_pet}"
source ../scripts/rcc_env.sh
mkdir -p logs results/listmode_nema_v600

PYTHON="${CONDA_PREFIX:?Activate the reconstruction environment}/bin/python"
export PYTHONPATH="$PWD/reconstruction:${PYTHONPATH:-}"

CTRS=(210)
CTR=${CTRS[$((SLURM_ARRAY_TASK_ID - 1))]}

# Same guard as the box: --total-seconds 600 ASSERTS what the directory holds,
# so a short ROOT set would silently produce a listmode labelled 600 s that
# contains less, and every count-rate number downstream would be wrong.
root_dir="${OUTDIR:-$PET_ROOT_DIR/nema_v600}"
shopt -s nullglob
chunks=("$root_dir"/*.root)
NROOT=${#chunks[@]}
if [[ "$NROOT" -ne 600 ]]; then
    echo "ERROR: $root_dir has $NROOT .root files, expected 600." >&2
    exit 1
fi
echo "$root_dir: $NROOT/600 chunks present"

OUT="results/listmode_nema_v600/lm_nema_600s_ctr${CTR}ps.npz"
echo "[$(date)] NEMA Vision 600, 600 s @ CTR ${CTR} ps -> $OUT"
if [[ -s "$OUT" ]]; then echo "  exists, skipping"; exit 0; fi

"$PYTHON" reconstruction/make_listmode_vision600.py \
    --root-dir "$root_dir" --total-seconds 600 --seconds 600 \
    --ctr-ps "$CTR" --energy-window 459.9 562.1 \
    --min-rsector-difference 0 \
    --lut-json reconstruction/vision600_lut.json \
    --out "$OUT"
echo "[$(date)] done: $(ls -lh "$OUT" | awk '{print $5}')"
