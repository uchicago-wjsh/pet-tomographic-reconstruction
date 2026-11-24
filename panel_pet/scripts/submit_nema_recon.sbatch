#!/bin/bash
#SBATCH --job-name=rec_nema
#SBATCH --array=1-3
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/recnema_%A_%a.out
#SBATCH --error=logs/recnema_%A_%a.err

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

CTRS=(40 80 120)
CTR=${CTRS[$((SLURM_ARRAY_TASK_ID - 1))]}
case "$CTR" in
    40)  TOF_BIN_PS=10; TOF_NBINS=521 ;;
    80)  TOF_BIN_PS=20; TOF_NBINS=261 ;;
    120) TOF_BIN_PS=30; TOF_NBINS=175 ;;
esac

LM="results/listmode_nema/lm_nema_600s_ctr${CTR}ps.npz"
OUT_DIR="results/recon_nema/600s_ctr${CTR}ps"
MC_MAP="${MC_MAP:-results/sensitivity/mc_sens_alpha1_200x200x200_2.0mm.npy}"
for f in "$LM" "$MC_MAP" "${MC_MAP%.npy}.json"; do
    [[ -s "$f" ]] || { echo "ERROR: $f missing" >&2; exit 1; }
done

echo "[$(date)] NEMA box, 600 s @ CTR ${CTR} ps -> $OUT_DIR"
echo "  grid 160x160x96 @ 2 mm = 320 x 320 x 192 mm"
nvidia-smi --query-gpu=name --format=csv,noheader || true
"$PYTHON" - <<'PY'
import sys, torch, cupy, pytomography, parallelproj.backend as ppb
problems = []
if str(pytomography.device) == "cpu": problems.append("pytomography on CPU")
else:
    print(f"GPU {torch.cuda.get_device_name(0)}  cc {torch.cuda.get_device_capability(0)}")
    try: torch.ones(8, device="cuda").sum().item()
    except Exception as e: problems.append(f"torch kernel launch failed: {e}")
    try: cupy.arange(8).sum().item()
    except Exception as e: problems.append(f"cupy kernel launch failed: {e}")
if ppb.num_visible_cuda_devices < 1: problems.append("parallelproj sees no CUDA device")
if problems: sys.exit("ERROR: GPU stack not usable: " + "; ".join(problems))
PY

"$PYTHON" reconstruction/recon_mlem.py \
    --listmode "$LM" --lut reconstruction/scanner_lut.npy \
    --output-dir "$OUT_DIR" \
    --n-iters 200 --save-iters 10 30 50 100 200 \
    --grid-size 160 --grid-size-z 96 --voxel-size-mm 2.0 \
    --tof-num-bins "$TOF_NBINS" --tof-bin-width-ps "$TOF_BIN_PS" --tof-fwhm-ps "$CTR" \
    --n-sensitivity-lors 0 \
    --mc-sensitivity-map "$MC_MAP" \
    --n-splits 16
echo "[$(date)] done"
