# Cache, temporary storage and ROOT locations for the GATE and Slurm wrappers.
export PET_SIM_ROOT="${PET_SIM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"

# Bulk but transient: ROOT files that get histogrammed and deleted, scratch
# copies, per-run logs. On /project so a run is never at the mercy of the
# scratch quota, but nothing in here is precious.
export PET_TMPDIR="${PET_TMPDIR:-$PET_SIM_ROOT/tmp}"
export PET_CACHE_DIR="${PET_CACHE_DIR:-$PET_SIM_ROOT/cache}"
export PET_ROOT_DIR="${PET_ROOT_DIR:-$PET_SIM_ROOT/panel_pet/root}"

# Override the /home default described above. TMP/TEMP too: some tools read
# those instead, and a half-redirected TMPDIR is worse than none.
export TMPDIR="$PET_TMPDIR"
export TMP="$PET_TMPDIR"
export TEMP="$PET_TMPDIR"

# Package and tool caches, likewise off the home quota.
export CONDA_PKGS_DIRS="$PET_CACHE_DIR/conda_pkgs"
export PIP_CACHE_DIR="$PET_CACHE_DIR/pip"
export XDG_CACHE_HOME="$PET_CACHE_DIR/xdg"
export MPLCONFIGDIR="$PET_CACHE_DIR/mpl"

# The exception. RCC's Apptainer guidance
# (<https://docs.rcc.uchicago.edu/software/apps-and-envs/singularity/>) is that
# the cache and tmp directories must be local disk, not a network filesystem:
# image builds unpack hundreds of thousands of small files and are painfully
# slow (and can fail on ownership) over GPFS. /tmp is node-local and wiped
# with the job, so it is neither /home nor /scratch.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/tmp/$USER/apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/$USER/apptainer_tmp}"

# Sourced interactively as often as not, so a failure here must not take the
# caller's shell down with it under `set -e`.
mkdir -p "$PET_TMPDIR" "$PET_CACHE_DIR" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" \
         "$MPLCONFIGDIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" \
         2>/dev/null || true
