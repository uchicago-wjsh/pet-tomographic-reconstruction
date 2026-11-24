#!/bin/bash
#SBATCH --job-name=rec_nema_v600mc
#SBATCH --array=1-1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/recnemav600mc_%A_%a.out
#SBATCH --error=logs/recnemav600mc_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from panel_pet}"
source ../scripts/rcc_env.sh
mkdir -p logs

CONDA_ENV="${CONDA_PREFIX:?Activate the reconstruction environment}"
PYTHON=$CONDA_ENV/bin/python
export PARALLELPROJ_CUDA_LIB="$CONDA_ENV/lib/libparallelproj_cuda.so"
export PARALLELPROJ_CUDA_KERNEL_FILE="$CONDA_ENV/lib/projector_kernels.cu.1.10.2"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/reconstruction:${PYTHONPATH:-}"

CTRS=(210)
CTR=${CTRS[$((SLURM_ARRAY_TASK_ID - 1))]}
case "$CTR" in
    210) TOF_BIN_PS=50; TOF_NBINS=123 ;;
    *)   echo "no TOF binning defined for CTR=$CTR" >&2; exit 1 ;;
esac

LM="results/listmode_nema_v600/lm_nema_600s_ctr${CTR}ps.npz"
MC_MAP="${MC_MAP:-results/sensitivity/mc_sens_v600_alpha1_128x128x128_2.0mm.npy}"
OUT_DIR="results/recon_nema_v600_mc/600s_ctr${CTR}ps"
[[ -s "$LM"     ]] || { echo "ERROR: $LM missing" >&2; exit 1; }
[[ -s "$MC_MAP" ]] || { echo "ERROR: $MC_MAP missing" >&2; exit 1; }

echo "[$(date)] NEMA Vision 600 + MC correction, CTR ${CTR} ps -> $OUT_DIR"
nvidia-smi --query-gpu=name --format=csv,noheader || true

"$PYTHON" - <<'PY'
import sys, torch, cupy, pytomography, parallelproj.backend as ppb
print(f"  device {pytomography.device} | pp cuda devices {ppb.num_visible_cuda_devices}")
if str(pytomography.device) == "cpu": sys.exit("ERROR: on CPU")
try: torch.ones(8, device="cuda").sum().item(); cupy.arange(8).sum().item()
except Exception as e: sys.exit(f"ERROR: kernel launch failed: {e}")
PY

"$PYTHON" reconstruction/recon_mlem.py \
    --listmode "$LM" \
    --lut reconstruction/vision600_lut.npy \
    --output-dir "$OUT_DIR" \
    --n-iters 200 --save-iters 2 4 6 8 10 30 50 100 200 \
    --grid-size 160 --grid-size-z 96 --voxel-size-mm 2.0 \
    --tof-num-bins "$TOF_NBINS" --tof-bin-width-ps "$TOF_BIN_PS" --tof-fwhm-ps "$CTR" \
    --n-sensitivity-lors 0 \
    --mc-sensitivity-map "$MC_MAP" \
    --n-splits 48
echo "[$(date)] done"
