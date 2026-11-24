# PET tomographic reconstruction

Reconstruct 3D activity images from simulated PET coincidences using time-of-flight (TOF) list-mode MLEM in PyTomography. The source comes from the `panel_pet/reconstruction` workflow in `PET_simulation`. It supports the four-panel scanner and the Biograph Vision 600 ring. A direct TOF backprojection provides a count image on the same grid.

## Run a point-source reconstruction

The example generates a point at the scanner origin with GATE 9.4.1, converts the coincidences to list mode, and reconstructs a 50 mm field of view. A Linux machine with Conda and Apptainer can run the example on CPU.

```bash
# From the repository root
conda env create -f environment.yml
conda activate pet-reconstruction
export PARALLELPROJ_C_LIB="$CONDA_PREFIX/lib/libparallelproj_c.so"
export OMP_NUM_THREADS=2

scripts/build_container.sh
cd panel_pet
mkdir -p work/point
../scripts/gate.sh -a '[SEED,77][NPRIM,100000][OUT,work/point/acquisition]' \
    macros/main_point_center.mac
scripts/reconstruct_point.sh work/point/acquisition.root work/point/image
```

If you already have the GATE image, set `GATE_SIF` to its absolute path and skip the build. If you already have a four-panel point-source ROOT file with `Hits` and `Coincidences`, start with the reconstruction script. Its two arguments are the input file and the output directory. The scanner geometry and the 40 ps timing model are fixed for this example.

| Output inside `work/point/image/` | Contents |
| --- | --- |
| `scanner_lut.npy` and `scanner_lut.json` | Crystal coordinates in mm and the detector ranges for each panel |
| `events.npz` | Detector pairs and signed time differences in seconds |
| `tof_counts.npy` | Direct TOF histogram on the 25 × 25 × 25 grid |
| `mlem/reconstruction_mlem_tof.nii` and `.npy` | Five-iteration TOF-MLEM image at 2 mm spacing |
| `mlem/reconstruction_meta.json` | Event count, grid, timing settings, projector and sensitivity settings |

The example samples one million cross-panel lines of response to estimate the geometric sensitivity. This keeps a CPU run small. The production NEMA jobs use exact geometric normalization and a scanner-specific Monte Carlo efficiency map.

## Reconstruct another acquisition

Run the commands below from `panel_pet/`. Select the geometry that acquired the data. The detector lookup table (LUT), its JSON sidecar and the list-mode IDs must use the same indexing.

| Stage | Four-panel scanner | Vision 600 |
| --- | --- | --- |
| Build geometry | `reconstruction/scanner_lut.py` reads the GATE macro and module placements | `reconstruction/vision600_lut.py` maps the cylindricalPET hierarchy and axial orientation |
| Convert timed ROOT chunks | `reconstruction/make_listmode.py` selects true cross-panel coincidences | `reconstruction/make_listmode_vision600.py` uses rsector/module/submodule/crystal IDs |
| Reconstruct | `reconstruction/recon_mlem.py` uses cross-panel pairs for normalization | The same entry point reads the ring sidecar and accepts all distinct detector pairs for the default sector cut |
| Direct TOF image | `reconstruction/tof_backproject.py` | Same entry point with the ring LUT |

For example, these commands process a directory of equal-duration four-panel ROOT chunks. Set `root_dir` and `total_seconds` to the directory and its actual combined acquisition time. Each chunk must contain the GATE `Coincidences` tree. The converter reads the detector IDs, event IDs, energy and readout time on both arms, applies a 459.9–562.1 keV window, and adds 40 ps coincidence timing resolution to the simulated ideal timing.

```bash
root_dir=/path/to/four-panel/chunks
total_seconds=600
python reconstruction/scanner_lut.py --no-plot
python reconstruction/make_listmode.py --root-dir "$root_dir" \
    --total-seconds "$total_seconds" --ctr-ps 40 --out work/events.npz
python reconstruction/recon_mlem.py --listmode work/events.npz \
    --lut reconstruction/scanner_lut.npy --output-dir work/mlem \
    --grid-size 160 --grid-size-z 96 --voxel-size-mm 2 \
    --tof-num-bins 521 --tof-bin-width-ps 10 --tof-fwhm-ps 40 \
    --n-iters 30 --save-iters 10 30 --n-sensitivity-lors 0 --n-splits 16
```

Exact normalization enumerates 1,003,622,400 panel pairs. Use a host with memory comparable to the 96 GB reconstruction allocation for that command. For an initial small run, use the point-source example above. For a 20 mm panel detector, pass `--macro macros/geometry_20mm.mac` and a separate `--out` LUT name, then pass its matching `--lut-json` to the converter.

For Vision 600, run `python reconstruction/vision600_lut.py` and use `make_listmode_vision600.py` with `--ctr-ps 210 --min-rsector-difference 0`. Reconstruct with `--lut reconstruction/vision600_lut.npy --tof-num-bins 123 --tof-bin-width-ps 50 --tof-fwhm-ps 210`. The ring normalization in this recipe matches a minimum sector difference of zero.

For existing list-mode data, start at the reconstruction step. The NPZ must contain `detector_ids` of shape `(N, 2)` with zero-based detector IDs and `tof_seconds` of shape `(N,)` using `time1 - time2` in the stored detector order. Match `--tof-fwhm-ps` to the coincidence timing resolution applied to the acquisition. The converters add timing blur for simulated data with ideal electronics.

## NEMA production jobs

The Slurm scripts run the project's 600-second NEMA acquisition, conversion, and 200-iteration GPU reconstruction. The panel array converts the same acquisition at 40, 80, and 120 ps. The Vision 600 jobs use 210 ps. Each reconstruction uses a 160 × 160 × 96 grid at 2 mm.

From `panel_pet/`, activate the Python environment, prepare the phantom and LUT, and provide the scanner's calibrated Monte Carlo sensitivity array with its matching JSON sidecar. The GPU jobs need CUDA-enabled PyTorch, CuPy compatible with the installed CUDA runtime, and the parallelproj CUDA library and kernels in the active Conda environment. Supply the account and partitions for your cluster.

```bash
mkdir -p logs
python scripts/make_nema_phantom.py
python reconstruction/scanner_lut.py --no-plot
export MC_MAP=/path/to/panel/mc_sens_alpha1_200x200x200_2.0mm.npy

gate_job=$(sbatch --parsable -A "$ACCOUNT" -p "$CPU_PARTITION" scripts/submit_nema.sbatch)
lm_job=$(sbatch --parsable -A "$ACCOUNT" -p "$CPU_PARTITION" \
    --dependency=afterok:"$gate_job" scripts/submit_nema_listmode.sbatch)
sbatch -A "$ACCOUNT" -p "$GPU_PARTITION" --dependency=afterok:"$lm_job" \
    scripts/submit_nema_recon.sbatch
```

For the ring, generate `vision600_lut.npy`, set `MC_MAP` to the Vision calibration, and submit `submit_nema_vision600.sbatch`, `submit_nema_listmode_vision600.sbatch`, and `submit_nema_recon_vision600_mc.sbatch` with the same dependencies.

The calibration uses separate uniform-source acquisitions. The four-panel source macro is `macros/main.mac` (400 mm cube) and the ring macro is `macros/main_sens_vision600.mac` (256 mm cube). For example, to reduce an existing set of panel calibration chunks and write the array and JSON sidecar, run the commands below. For the ring calibration, use its own chunks and `--fov 256`.

```bash
python scripts/sensitivity.py '/path/to/panel-calibration/*.root' --fov 400 --voxel 2 \
    --fold -o work/panel_efficiency.npz
python reconstruction/unfold_mc_sensitivity.py --npz work/panel_efficiency.npz \
    --alpha 1 --out work/panel_efficiency.npy
```

## Other reconstruction tools

- `root_to_listmode.py` takes explicit panel ROOT files for fixed-primary runs. Its energy-weighted timing uses the `Hits` tree. Use `--time-source readout` for coincidence-only files.
- `compute_voxel_sensitivity.py` and `compute_voxel_sensitivity_vision600.py` precompute geometric sensitivity. `reconstruct_mlem_pytomography.py` supports cached geometric sensitivity and sparse system matrices.
- Both MLEM entry points support finite-crystal endpoint sampling. The package documentation covers geometry export, the line-of-response model, CLI flags, and tests. [Finite-crystal projector](panel_pet/reconstruction/pet_projector/README.md)

## Validation

The point-source recipe above ran on CPU with GATE 9.4.1, PyTomography 3.4.0, and parallelproj 1.10.2. It retained 3,931 coincidences. The direct histogram summed to 3,931 and the five-iteration NIfTI image peaked at (0, 0, 0) mm. A separate Vision 600 ROOT acquisition yielded 69,324 events and a finite one-iteration image. These checks used sampled geometric sensitivity.

To run the numerical projector tests from `panel_pet/reconstruction/` in the activated environment:

```bash
PARALLELPROJ_SILENT_IMPORT=1 OMP_NUM_THREADS=1 \
    python -m unittest discover -s tests -p 'test_pet_projector.py' -v
```

The suite checks forward/backward adjointness, TOF bin handling, sampled sensitivity, geometry export and the MLEM entry points. The CUDA comparison runs when a GPU is available.
