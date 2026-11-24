#!/bin/bash
# ---------------------------------------------------------------------------
# build_container.sh -- (re)create the GATE 9 Apptainer image on RCC Midway3.
#
# Run this on a LOGIN node: compute nodes have no outbound internet access,
# so the Docker layers cannot be fetched from there.
#
# Follows RCC guidance (docs.rcc.uchicago.edu/software/apps-and-envs/singularity):
# apptainer comes from a module, and its cache/tmp must live on local disk
# (/tmp) rather than on NFS-backed /home or /project.
# ---------------------------------------------------------------------------
set -euo pipefail

# Sets PET_SIM_ROOT, TMPDIR and the APPTAINER_* cache/tmp pair (node-local, as
# RCC requires for image builds). See scripts/rcc_env.sh.
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/rcc_env.sh"

GATE_TAG="${GATE_TAG:-9.4.1-docker}"
SIF="$PET_SIM_ROOT/containers/gate_${GATE_TAG%-docker}.sif"

if ! command -v module &>/dev/null; then
    source /etc/profile.d/modules.sh 2>/dev/null || true
fi
command -v apptainer >/dev/null || module load apptainer/1.4.1

mkdir -p "$PET_SIM_ROOT/containers"

echo "Pulling docker://opengatecollaboration/gate:$GATE_TAG -> $SIF"
apptainer pull --force "$SIF" "docker://opengatecollaboration/gate:$GATE_TAG"

echo
echo "Built: $SIF"
apptainer exec "$SIF" bash -c 'source /etc/mybashrc >/dev/null 2>&1; Gate --version'

# GATE looks for these two files in the directory the macro runs from, and the
# copy shipped inside the image is the canonical one.
cd "$PET_SIM_ROOT"
if [[ ! -f GateMaterials.db ]]; then
    apptainer exec "$SIF" cat /software/gate/src/GateMaterials.db > GateMaterials.db
    # GATE 9.4.1 ships this database without an "Air" entry even though most
    # PET/SPECT macros reference it; add the standard NIST dry-air composition.
    cat >> GateMaterials.db <<'EOF'

Air: d=1.29 mg/cm3 ; n=4 ; state=gas
        +el: name=Nitrogen   ; f=0.755268
        +el: name=Oxygen     ; f=0.231781
        +el: name=Argon      ; f=0.012827
        +el: name=Carbon     ; f=0.000124
EOF
    echo "Wrote GateMaterials.db (+ Air)"
fi
if [[ ! -f Materials.xml ]]; then
    apptainer exec "$SIF" cat /software/gate/src/Materials.xml > Materials.xml
    echo "Wrote Materials.xml"
fi
