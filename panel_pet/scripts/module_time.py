#!/usr/bin/env python
"""
Energy-weighted module event time -- the definition GATE cannot produce.

The task specifies:
    measured event energy = sum of all energy deposited in the module
    measured event time   = energy-weighted interaction time of all
                            interactions in the module

GATE 9.4.1 gives the first but not the second. Its two readout policies are
(verified in GateVDigitizerModule::CentroidMerge and GateReadout.cc):

    TakeEnergyWinner   -> time = min(contributing times)
    TakeEnergyCentroid -> time = unweighted mean of contributing times

Neither weights by energy. This script rebuilds module-level events directly
from the Hits tree, where the correct quantities are:

    E_module = sum_i edep_i
    t_module = sum_i (edep_i * t_i) / sum_i edep_i

Because the electronics time resolution is assumed perfect, t_module can then
be resampled to emulate any coincidence time resolution (CTR): add Gaussian
noise of sigma = CTR_FWHM / (2.355 * sqrt(2)) to each of the two arms, so the
difference of the two carries the requested CTR.

Run on output from macros/main_hits.mac (small statistics -- the Hits tree is
far too large at production scale):

    ../scripts/gate.sh -a '[SEED,1][ACTIVITY,2000][DURATION,1][OUT,output/hits]' \
                       macros/main_hits.mac
    python scripts/module_time.py output/hits.root
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import uproot


def module_events(path: str):
    """Collapse Hits into one record per (event, module)."""
    with uproot.open(path) as f:
        if "Hits" not in [k.split(";")[0] for k in f.keys()]:
            raise SystemExit("no Hits tree -- run macros/main_hits.mac")
        h = f["Hits"]
        if h.num_entries == 0:
            raise SystemExit("Hits tree is empty -- enable setRootHitFlag 1")
        a = h.arrays(["eventID", "runID", "time", "edep", "level1ID", "level2ID"],
                     library="np")

    keep = a["edep"] > 0
    ev, run = a["eventID"][keep], a["runID"][keep]
    t, e = a["time"][keep], a["edep"][keep]
    mod, cry = a["level1ID"][keep], a["level2ID"][keep]

    # one group per (run, event, module)
    key = np.stack([run, ev, mod], axis=1)
    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    key, t, e, cry = key[order], t[order], e[order], cry[order]
    new = np.ones(len(key), dtype=bool)
    new[1:] = np.any(key[1:] != key[:-1], axis=1)
    gid = np.cumsum(new) - 1
    n_groups = gid[-1] + 1

    esum = np.bincount(gid, weights=e, minlength=n_groups)
    et = np.bincount(gid, weights=e * t, minlength=n_groups)
    tmin = np.full(n_groups, np.inf)
    np.minimum.at(tmin, gid, t)
    tsum = np.bincount(gid, weights=t, minlength=n_groups)
    nhit = np.bincount(gid, minlength=n_groups)
    ncry = np.zeros(n_groups, dtype=np.int64)
    # count distinct crystals per group
    first = np.flatnonzero(new)
    bounds = np.append(first, len(key))
    for g in range(n_groups):
        ncry[g] = len(np.unique(cry[bounds[g]:bounds[g + 1]]))

    return {
        "run": key[first, 0], "event": key[first, 1], "module": key[first, 2],
        "energy": esum,                       # summed module energy (MeV)
        "t_eweighted": et / esum,             # <-- the required definition
        "t_min": tmin,                        # what GATE's TakeEnergyWinner gives
        "t_mean": tsum / nhit,                # what TakeEnergyCentroid gives
        "n_hits": nhit, "n_crystals": ncry,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", help="GATE ROOT file containing a Hits tree")
    p.add_argument("--ctr", type=float, default=None,
                   help="demo: resample times to this coincidence time "
                        "resolution in ps FWHM")
    args = p.parse_args()

    m = module_events(args.root)
    n = len(m["energy"])
    multi = m["n_hits"] > 1

    print(f"module-level events            : {n:,}")
    print(f"  with >1 hit (time differs)   : {multi.sum():,} "
          f"({multi.mean()*100:.1f}%)")
    print(f"  spanning >1 crystal          : {(m['n_crystals']>1).sum():,}")
    print(f"summed module energy (keV)     : "
          f"{m['energy'].min()*1000:.1f} .. {m['energy'].max()*1000:.1f}")

    if multi.any():
        d_min = (m["t_eweighted"] - m["t_min"])[multi] * 1e12    # s -> ps
        d_mean = (m["t_eweighted"] - m["t_mean"])[multi] * 1e12
        print("\nenergy-weighted time minus GATE's alternatives, "
              "multi-hit modules only (ps):")
        print(f"  vs t_min  (TakeEnergyWinner)   mean {d_min.mean():+8.2f}  "
              f"rms {d_min.std():7.2f}  max |{np.abs(d_min).max():.2f}|")
        print(f"  vs t_mean (TakeEnergyCentroid) mean {d_mean.mean():+8.2f}  "
              f"rms {d_mean.std():7.2f}  max |{np.abs(d_mean).max():.2f}|")
        print("\nThese offsets are picoseconds against a 10 ns coincidence "
              "window,\nso the choice does not change which pairs are "
              "accepted -- the\nsensitivity results are unaffected. It does "
              "matter once a realistic\nCTR is imposed.")

    if args.ctr:
        # Per-arm sigma so that the *difference* of two arms has the target FWHM.
        sigma_ps = args.ctr / (2.3548 * np.sqrt(2.0))
        rng = np.random.default_rng(0)
        t_ctr = m["t_eweighted"] + rng.normal(0.0, sigma_ps * 1e-12, n)
        print(f"\nresampled to CTR = {args.ctr:.0f} ps FWHM "
              f"(per-arm sigma {sigma_ps:.1f} ps); "
              f"example spread {np.std((t_ctr-m['t_eweighted'])*1e12):.1f} ps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
