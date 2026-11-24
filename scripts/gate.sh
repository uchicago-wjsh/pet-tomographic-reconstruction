#!/bin/bash
# ---------------------------------------------------------------------------
# gate.sh -- run GATE 9.4.1 inside the Apptainer container on RCC Midway3
#
# Usage:
#   ./scripts/gate.sh <macro.mac> [extra Gate args...]
#   ./scripts/gate.sh -a '[nb,1000]' macros/simple_pet.mac
#   ./scripts/gate.sh --shell              # interactive shell inside container
#
# The directory you run this from becomes the container's working directory,
# so relative paths in your macro resolve exactly as they do on the host.
#
# Follows RCC's Apptainer guidance (docs.rcc.uchicago.edu/software/apps-and-envs/singularity):
#   - apptainer provided via `module load apptainer`
#   - APPTAINER_CACHEDIR / APPTAINER_TMPDIR point at node-local /tmp, not NFS
#   - host paths made visible with explicit --bind
#
# scripts/rcc_env.sh sets those two, plus TMPDIR, and is the single place where
# this project's storage layout is decided -- see the comments in it.
# ---------------------------------------------------------------------------
set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/rcc_env.sh"

SIF="${GATE_SIF:-$PET_SIM_ROOT/containers/gate_9.4.1.sif}"

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: GATE container not found at $SIF" >&2
    echo "       Rebuild it with: $PET_SIM_ROOT/scripts/build_container.sh" >&2
    exit 1
fi

# `module` is a shell function; make it available in non-interactive shells.
if ! command -v module &>/dev/null; then
    source /etc/profile.d/modules.sh 2>/dev/null || true
fi
command -v apptainer &>/dev/null || module load apptainer/1.4.1

# Bind the host trees GATE may need to read or write. Apptainer auto-mounts the
# working directory, but not the paths a symlink out of it resolves to --
# panel_pet/GateMaterials.db -> ../GateMaterials.db is exactly that -- so the
# whole project tree has to be bound explicitly. /project2 is where this project
# actually lives; the rest are bound only when they exist, so pointing OUTDIR at
# scratch or at node-local job storage still works.
binds=(-B "$PET_SIM_ROOT")
for extra in /project /project2 "/scratch/midway3/$USER" \
             "${SLURM_JOB_ID:+/scratch/local/jobs/$SLURM_JOB_ID}"; do
    if [[ -n "$extra" && -d "$extra" ]]; then
        binds+=(-B "$extra")
    fi
done

WORKDIR="$(pwd -P)"

# --cleanenv keeps the host environment (and ~/.bashrc) from leaking into the
# container; /etc/mybashrc inside the image sets up Geant4 + ROOT + Gate.
if [[ "${1:-}" == "--shell" ]]; then
    exec apptainer exec --cleanenv "${binds[@]}" --pwd "$WORKDIR" "$SIF" \
        bash --rcfile /etc/mybashrc -i
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <macro.mac> [extra Gate args...]" >&2
    echo "       $0 --shell" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# GATE 9.4.1 (Geant4 11.3.0) in this image reliably raises SIGSEGV inside
# G4RunManager::DeleteUserInitializations() while tearing down, i.e. *after*
# "End of macro" and after every output file has been flushed and closed.
# Verified: the ROOT trees written by such a run open cleanly in uproot with
# the full expected entry counts. Left alone it surfaces as exit 139, which
# Slurm reports as a FAILED job.
#
# So: only when the run actually reached the end of the macro AND Gate logged
# no macro-command error do we treat a SIGSEGV as success. Any other failure
# keeps its original exit code.
# ---------------------------------------------------------------------------
# Re-create here, not just at the top: on multi-hour jobs a node tmp cleaner
# can remove /tmp/$USER between invocations, which fails mktemp mid-run.
mkdir -p "$APPTAINER_TMPDIR" 2>/dev/null || true
LOG="$(mktemp "${APPTAINER_TMPDIR}/gate_run_XXXXXX.log" 2>/dev/null \
       || mktemp "${TMPDIR:-/tmp}/gate_run_XXXXXX.log")"
trap 'rm -f "$LOG"' EXIT

set +e
apptainer exec --cleanenv "${binds[@]}" --pwd "$WORKDIR" "$SIF" \
    bash -c 'source /etc/mybashrc >/dev/null 2>&1; exec Gate "$@"' _ "$@" \
    2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

if [[ $rc -eq 139 ]] \
   && grep -q "End of macro" "$LOG" \
   && ! grep -q "Sorry, error in a macro command" "$LOG"; then
    echo "" >&2
    echo "[gate.sh] NOTE: GATE completed the macro and wrote its output, then" >&2
    echo "[gate.sh] segfaulted during Geant4 shutdown (a known GATE 9.4.1 /" >&2
    echo "[gate.sh] Geant4 11.3 teardown bug). Output files are complete." >&2
    echo "[gate.sh] Reporting success (exit 0) instead of 139." >&2
    rc=0
fi

exit $rc
