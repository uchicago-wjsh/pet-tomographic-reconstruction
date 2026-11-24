#!/usr/bin/env bash
# Reconstruct the 40 ps four-panel point-source example from a GATE ROOT file.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 POINT.root OUTPUT_DIRECTORY" >&2
    exit 2
fi

root_file=$1
out=$2
recon=$(cd "$(dirname "$0")/../reconstruction" && pwd)
python_bin=${PYTHON:-python}
"$python_bin" "$recon/scanner_lut.py" --out "$out/scanner_lut.npy" --no-plot
"$python_bin" "$recon/root_to_listmode.py" "$root_file" \
    --lut-json "$out/scanner_lut.json" --time-source energy-weighted \
    --ctr-ps 40 --out "$out/events.npz"
"$python_bin" "$recon/tof_backproject.py" --listmode "$out/events.npz" \
    --lut "$out/scanner_lut.npy" --kernel delta --grid-size 25 --grid-size-z 25 \
    --voxel-size-mm 2 --out "$out/tof_counts.npy"
"$python_bin" "$recon/recon_mlem.py" --listmode "$out/events.npz" \
    --lut "$out/scanner_lut.npy" --output-dir "$out/mlem" \
    --grid-size 25 --voxel-size-mm 2 --tof-num-bins 521 --tof-bin-width-ps 10 \
    --tof-fwhm-ps 40 --n-iters 5 --n-sensitivity-lors 1000000 --n-splits 8
