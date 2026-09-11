"""Figure 10 -- topography and material heterogeneity on a single mesh
(vertical strike-slip fault, soft cylindrical inclusion, Gaussian hill).

The fields come from the moss2 mollified boundary element solver, vendored
unchanged in topo_inclusion/ (see README.md).  Two stages:

  1. SOLVE (optional, ~20 min and ~16 GB RAM: four dense 31k-unknown solves)
         python scripts/fig10_topography_inclusion.py --solve
     runs topo_inclusion/make_topo_inclusion.py with the paper's settings
     (--mu-inc 3, i.e. mu_inc = mu/10) and writes
     cache/topo_inclusion_fields_mu10.npz.  The cached file shipped with the
     package is the one the paper figure was rendered from.

  2. RENDER (seconds)
         python scripts/fig10_topography_inclusion.py
     runs topo_inclusion/render_topo_inclusion_contour.py on the cache with
     the paper's options (empty suffix, --smooth) and moves the resulting PDF
     to manuscript/figures/fig_topography_inclusion.pdf.

Both moss2 scripts are executed as subprocesses with the same interpreter, so
their own repo-relative path handling keeps working inside topo_inclusion/.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
TOPO = os.path.join(ROOT, "topo_inclusion")
CACHE = os.path.join(ROOT, "cache", "topo_inclusion_fields_mu10.npz")
OUT = os.path.join(ROOT, "manuscript", "figures", "fig_topography_inclusion.pdf")


def main(argv=None):
    p = argparse.ArgumentParser(description="Figure 10: topography + inclusion")
    p.add_argument("--solve", action="store_true",
                   help="recompute the BEM fields (slow) before rendering")
    args = p.parse_args(argv)

    if args.solve or not os.path.exists(CACHE):
        print("solving the four BEM systems (this takes ~20 minutes) ...",
              flush=True)
        subprocess.run([sys.executable,
                        os.path.join(TOPO, "make_topo_inclusion.py"),
                        "--mu-inc", "3", "--out", CACHE], check=True)

    # the renderer writes fig_topo_inclusion_contour<suffix>.{png,pdf} to the
    # PARENT of its own directory, i.e. the package root
    subprocess.run([sys.executable,
                    os.path.join(TOPO, "render_topo_inclusion_contour.py"),
                    CACHE, "", "--smooth"], check=True)
    produced = os.path.join(ROOT, "fig_topo_inclusion_contour_smooth")
    shutil.move(produced + ".pdf", OUT)
    if os.path.exists(produced + ".png"):
        os.remove(produced + ".png")
    print("wrote", os.path.relpath(OUT))


if __name__ == "__main__":
    main()
