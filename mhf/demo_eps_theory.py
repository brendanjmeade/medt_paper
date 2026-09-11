"""THEORETICAL CALIBRATION of the smearing length from scaling laws
alone -- no seismicity data.

Chain (Section 4 of the manuscript):

  Cowie & Scholz (1992):  d_cum = gamma_CS * L, linear displacement--
      length scaling of fault structures, gamma_CS ~ 1e-3..1e-1;
  Savage & Brodsky (2011): damage-zone width grows with cumulative
      displacement, W ~ beta * d_cum below the ~150 m transition
      displacement (beta parameterized), saturating at
      W_sat ~ 150--400 m above it;
  dictionary:  eps = (3/4) * W  (equivalent uniform-width match).

So  eps_theory = (3/4) * min(beta * gamma_CS * L, W_sat):  every host
structure with cumulative displacement above ~150 m sits on the
saturation plateau eps ~ 110--300 m, independent of beta.

Hirata (1989) closes the exponent side: for a fault population with
map-view fractal dimension D, the standard mass-scaling argument gives
the areal density of strands at fault-normal distance y from a primary
strand as ~ y^(D-2), i.e., decay exponent q = 2 - D.  D = 1.05--1.60
predicts q = 0.40--0.95, bracketing Savage & Brodsky's measured 0.8
(single strand) and 0.4 (mature superposition) -- the fractal skirt
that the blob marginal (exponent 5) deliberately omits.

Writes manuscript/figures/fig_mhf_eps_theory.{png,pdf}.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

try:
    from . import figures  # noqa: F401  (house rcParams)
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from mhf import figures  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "manuscript", "figures")

W_SAT_KM = (0.150, 0.400)         # Savage-Brodsky saturation width band
D_TRANS_KM = 0.150                # S&B transition displacement
BETA = (0.1, 1.0)                 # growth-regime slope (parameterized)
GAMMA_CS = 1e-2                   # Cowie-Scholz central value (top axis)

HIRATA_D = (1.05, 1.60)           # map-view fractal dimension range
SB_EXPONENTS = (0.8, 0.4)         # single strand / mature superposition
PJ_GAMMA = (1.16, 2.30)           # Powers-Jordan background range
THIS_STUDY = (2.04, 3.31)         # fitted aftershock exponents
BLOB = 5.0


def eps_theory(d_cum_km, beta, w_sat_km):
    return 0.75 * np.minimum(beta * d_cum_km, w_sat_km)


def main():
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.4, 4.4),
                                   gridspec_kw=dict(wspace=0.35))

    # (a) eps_theory vs cumulative displacement -------------------------
    d = np.logspace(-3, 2, 400)                 # km
    lo = eps_theory(d, BETA[0], W_SAT_KM[0])
    hi = eps_theory(d, BETA[1], W_SAT_KM[1])
    ax0.fill_between(d, lo, hi, color="#1f5fa6", alpha=0.25, lw=0)
    for beta, w in ((BETA[0], W_SAT_KM[0]), (BETA[1], W_SAT_KM[1])):
        ax0.plot(d, eps_theory(d, beta, w), "-", color="#1f5fa6",
                 lw=1.0)
    # saturation plateau band, beta-independent
    ax0.fill_between([D_TRANS_KM / BETA[0], 100.0],
                     0.75 * W_SAT_KM[0], 0.75 * W_SAT_KM[1],
                     color="#1f5fa6", alpha=0.10, lw=0)
    ax0.axvline(D_TRANS_KM, color="#c1272d", lw=0.9, ls=":")
    ax0.text(D_TRANS_KM * 1.25, 2.2e-4,
             r"S&B transition $d_{\mathrm{cum}} \approx 150$ m",
             color="#c1272d", fontsize=7.5, rotation=90, va="bottom")
    ax0.annotate(r"plateau $\varepsilon = (3/4)W_{\mathrm{sat}}"
                 r" \approx$ 110$-$300 m",
                 xy=(3.0, 0.75 * W_SAT_KM[0]),
                 xytext=(0.44, 0.56), textcoords="axes fraction",
                 fontsize=8, ha="left",
                 arrowprops=dict(arrowstyle="-", lw=0.7, color="0.4"))
    ax0.text(0.0035, 0.0012,
             r"growth $\varepsilon = (3/4)\beta\,"
             r"d_{\mathrm{cum}}$" "\n"
             r"($\beta$ parameterized 0.1$-$1)",
             fontsize=8, color="#1f5fa6")
    # case-study hosts: km-scale cumulative displacements -> plateau
    ax0.text(0.97, 0.04,
             r"case-study hosts: $d_{\mathrm{cum}} \gtrsim 1$ km "
             r"$\rightarrow$ plateau",
             transform=ax0.transAxes, ha="right", va="bottom",
             fontsize=8, color="0.25")
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim(1e-3, 1e2)
    ax0.set_ylim(1e-4, 1e0)
    ax0.set_xticks([1e-3, 1e-1, 1e1])
    ax0.set_yticks([1e-4, 1e-2, 1e0])
    ax0.minorticks_off()
    ax0.set_xlabel(r"cumulative displacement $d_{\mathrm{cum}}$ (km)")
    ax0.set_ylabel(r"$\varepsilon_{\mathrm{theory}}$ (km)")
    # top axis: fault length via Cowie-Scholz at gamma_CS = 1e-2
    axt = ax0.secondary_xaxis(
        "top", functions=(lambda x: x / GAMMA_CS,
                          lambda x: x * GAMMA_CS))
    axt.set_xticks([1e-1, 1e1, 1e3])
    axt.set_xlabel(r"fault length $L$ (km) at "
                   r"$\gamma_{\mathrm{CS}}=10^{-2}$", fontsize=8.5)
    axt.tick_params(direction="out", length=3, width=0.8)
    axt.minorticks_off()

    # (b) fault-normal decay exponents ---------------------------------
    rows = [
        (r"blob marginal", (BLOB, BLOB), "#1f5fa6", "line"),
        ("this study (aftershocks)", THIS_STUDY, "0.2", "bar"),
        ("Powers–Jordan background", PJ_GAMMA, "0.55", "bar"),
        ("Savage–Brodsky fractures", (SB_EXPONENTS[1],
                                           SB_EXPONENTS[0]),
         "#c1272d", "bar"),
        (r"Hirata: $q = 2 - D$", (2.0 - HIRATA_D[1], 2.0 - HIRATA_D[0]),
         "#2a7f4f", "bar"),
    ]
    for i, (label, (qlo, qhi), color, kind) in enumerate(rows):
        y = len(rows) - 1 - i
        if kind == "line" or qlo == qhi:
            ax1.plot([qlo], [y], "o", ms=7, color=color)
            ax1.text(qlo - 0.18, y, label, ha="right", va="center",
                     fontsize=8.5, color=color)
        else:
            ax1.plot([qlo, qhi], [y, y], "-", color=color, lw=5,
                     solid_capstyle="butt", alpha=0.85)
            ax1.text(qhi + 0.18, y, label, ha="left", va="center",
                     fontsize=8.5, color=color)
    # S&B endpoint markers on their bar
    y_sb = 1
    ax1.plot([SB_EXPONENTS[0]], [y_sb], "o", ms=4, color="#c1272d")
    ax1.plot([SB_EXPONENTS[1]], [y_sb], "s", ms=4, color="#c1272d")
    ax1.set_xlim(0, 5.5)
    ax1.set_xticks([0, 1, 2, 3, 4, 5])
    ax1.set_ylim(-0.7, len(rows) - 0.3)
    ax1.set_yticks([])
    ax1.set_xlabel(r"fault-normal decay exponent")
    for k, ax in enumerate((ax0, ax1)):
        ax.tick_params(direction="out", length=3, width=0.8)
        ax.text(0.96, 0.95, "ab"[k], transform=ax.transAxes, ha="right",
                va="top", fontsize=9)
    ax0.set_box_aspect(1)
    ax1.set_box_aspect(1)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(FIG_DIR, f"fig_mhf_eps_theory.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
