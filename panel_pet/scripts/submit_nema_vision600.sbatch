#!/bin/bash
#SBATCH --job-name=nema_v600
#SBATCH --array=1-60%60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3840M
#SBATCH --time=02:00:00
#SBATCH --output=logs/nemav600_%A_%a.out
#SBATCH --error=logs/nemav600_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from panel_pet}"
source ../scripts/rcc_env.sh

SIM_SECONDS=600         # TOTAL simulated acquisition across the whole array
CHUNK_SECONDS=1.0       # ~276k coincidences -> ~1.9 GB per GATE process
NCHUNK=10               # -> 10 s of acquisition per task, so 60 tasks

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: not running as an array job; the header sets --array=1-60%60." >&2
    echo "  Submitted bare, this would produce 1/60th of the acquisition." >&2
    exit 1
fi
TASK="$SLURM_ARRAY_TASK_ID"

for f in data/nema_activity.mhd data/nema_activity.raw macros/main_nema_vision600.mac; do
    [[ -s "$f" ]] || { echo "ERROR: $f missing (run scripts/make_nema_phantom.py)" >&2; exit 1; }
done

OUTDIR="${OUTDIR:-$PET_ROOT_DIR/nema_v600}"
mkdir -p "$OUTDIR" logs

TASK_SECONDS=$(awk -v n="$NCHUNK" -v c="$CHUNK_SECONDS" 'BEGIN{printf "%.9g", n*c}')
TASK_START=$(awk -v k="$TASK" -v s="$TASK_SECONDS" 'BEGIN{printf "%.9g", (k-1)*s}')

NEEDED=$(awk -v t="$SIM_SECONDS" -v n="$NCHUNK" -v c="$CHUNK_SECONDS" \
             'BEGIN{printf "%.0f", t/(n*c)}')
NTASK="${SLURM_ARRAY_TASK_COUNT:-1}"
if [[ "$NTASK" -ne "$NEEDED" ]]; then
    echo "WARNING: array has $NTASK tasks but SIM_SECONDS=$SIM_SECONDS needs $NEEDED." >&2
fi

echo "host    : $(hostname)"
echo "scanner : Biograph Vision 600"
echo "phantom : NEMA IEC body, spheres 2642 Bq/cm3, background 4:1, 6.30 MBq"
echo "task    : $TASK   covers t = $TASK_START .. +$TASK_SECONDS s of $SIM_SECONDS s"
echo "output  : $OUTDIR"
echo "started : $(date)"

for ((C = 0; C < NCHUNK; C++)); do
    SEED=$(( 700000 + TASK * 100 + C ))
    TSTART=$(awk -v b="$TASK_START" -v c="$C" -v d="$CHUNK_SECONDS" \
                 'BEGIN{printf "%.9g", b + c*d}')
    OUT="$OUTDIR/nemav600_t$(printf '%04d' "$TASK")_c$(printf '%04d' "$C")"

    ../scripts/gate.sh \
        -a "[SEED,${SEED}][TSTART,${TSTART}][DURATION,${CHUNK_SECONDS}][OUT,${OUT}]" \
        macros/main_nema_vision600.mac > /dev/null

    echo "  chunk $((C+1))/$NCHUNK  t=${TSTART}s  seed $SEED  $(date +%H:%M:%S)"
done

echo "finished: $(date)"
