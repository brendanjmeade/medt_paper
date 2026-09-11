"""Fault-normal profile of the mollified slip distribution.

Four panels:
  (a) eps = 0     : slip-gradient density, a Dirac delta at y = 0
  (b) eps = eps_0 : the Cortez blob's fault-normal marginal
                    gamma(y) = (3/4) eps^4 / (y^2 + eps^2)^(5/2)
                    (integrates to 1)
  (c) eps = 0     : cumulative slip profile, a Heaviside step
  (d) eps = eps_0 : cumulative slip profile, the closed-form CDF
                    S(y) = 1/2 + y (3 eps^2 + 2 y^2)
                               / (4 (y^2 + eps^2)^(3/2))

Run:  python plot_mollified_profile.py
"""
# medt_paper: copied from moss/plot_mollified_profile.py; only the output path
# was changed (manuscript/figures/fig_blob_slip_cartoon.pdf).

import os

import matplotlib.pyplot as plt
import numpy as np

EPS0 = 0.25  # mollification scale epsilon_0
SLIP = 1.0  # total slip (profiles normalized by this)
Y_MAX = 5.0  # plot half-width in units of eps_0
LW = 1.0  # line width


def gamma_marginal(y, eps):
    """Fault-normal marginal of the Cortez blob (integrates to 1)."""
    return 0.75 * eps**4 / (y**2 + eps**2) ** 2.5


def cumulative(y, eps):
    """Closed-form integral of gamma_marginal from -infinity to y."""
    return 0.5 + y * (3 * eps**2 + 2 * y**2) / (4 * (y**2 + eps**2) ** 1.5)


def main():
    y = np.linspace(-Y_MAX, Y_MAX, 801)

    fig, axes = plt.subplots(
        2, 2, figsize=(5.0, 5.0), gridspec_kw=dict(wspace=0.35, hspace=0.35)
    )

    # (a) eps = 0: delta-function slip-gradient density
    ax = axes[0, 0]
    ax.plot([-Y_MAX, 0, 0, Y_MAX], [0, 0, 0, 0], "-", color="tab:blue", lw=LW)
    ax.plot([0, 0], [0, 1], "-", color="tab:blue", lw=LW)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([])
    ax.set_xlim(-Y_MAX, Y_MAX)
    ax.set_yticks([])
    ax.set_ylabel(r"$s(x)$")
    ax.set_title(r"CEDT ($\epsilon =0$)", fontsize=10)

    # (b) eps = eps_0: Cortez marginal
    ax = axes[0, 1]
    ax.plot(y, SLIP * gamma_marginal(y, EPS0), "-", color="tab:orange", lw=LW)
    ax.set_ylim(-0.1 * 0.75 * SLIP / EPS0, 1.1 * 0.75 * SLIP / EPS0)
    ax.set_xlim(-Y_MAX, Y_MAX)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(r"$s(x)$")
    ax.set_title(r"MEDT ($\epsilon > 0$)", fontsize=10)

    # (c) eps = 0: Heaviside step (cumulative slip)
    ax = axes[1, 0]
    ax.plot([-Y_MAX, 0], [0, 0], "-", color="tab:blue", lw=LW)
    ax.plot([0, Y_MAX], [SLIP, SLIP], "-", color="tab:blue", lw=LW)
    ax.plot([0, 0], [0, SLIP], "-", color="tab:blue", lw=LW)
    ax.set_ylim(-0.1 * SLIP, 1.1 * SLIP)
    ax.set_xlim(-Y_MAX, Y_MAX)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(r"$\int s(x)\;dx$")
    ax.set_title(r"CEDT ($\epsilon = 0$)", fontsize=10)

    # (d) eps = eps_0: smooth cumulative slip
    ax = axes[1, 1]
    ax.plot(y, SLIP * cumulative(y, EPS0), "-", color="tab:orange", lw=LW)
    ax.set_ylim(-0.1 * SLIP, 1.1 * SLIP)
    ax.set_xlim(-Y_MAX, Y_MAX)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(r"$\int s(x)\;dx$")
    ax.set_title(r"MEDT ($\epsilon > 0$)", fontsize=10)

    for _, ax in enumerate(axes.flat):
        ax.set_xlabel(r"$x$")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "manuscript", "figures", "fig_blob_slip_cartoon.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", os.path.relpath(out))


if __name__ == "__main__":
    main()
