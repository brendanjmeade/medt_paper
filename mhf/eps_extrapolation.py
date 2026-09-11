"""Richardson / Romberg extrapolation in the mollification parameter eps.

WHY THIS EXISTS
---------------
The exact-BC mollified Mindlin kernel converges to the classical (cutde)
half-space solution at O(eps^2) in the bulk, BUT only at O(eps^1) near a
SURFACE-BREAKING trace (where source rows have c=src_z -> 0, so the effective
image depth a=sqrt(c^2+eps^2) deviates from the classical |c| by O(eps) instead
of O(eps^2)).  This is an intrinsic property of the mollified kernel, NOT a
quadrature artifact -- diagnosed in mh/proto_decompose.py (converged-quadrature
error vs cutde has rate exactly 2.00 per eps-halving at the surface trace).

The near-trace error is a CLEAN analytic power series in eps,
    field(eps) = field_exact + A1 eps + A2 eps^2 + A3 eps^3 + ...,
so Richardson / Romberg extrapolation over a geometric eps-ladder cancels the
leading terms:
    levels=2  ->  2 field(eps/2) - field(eps)            = field_exact + O(eps^2)
    levels=k  ->  Romberg(eps, eps/2, ..., eps/2^{k-1})  = field_exact + O(eps^k).

CRUCIAL: every field(eps) satisfies the free-surface traction BC T_3k(z=0)=0 to
machine precision, so ANY linear combination does too -- extrapolation preserves
the exact BC.  Validated in mh/proto_romberg.py: near-trace stress error driven
from 3.7e-3 MPa (single eps) to 3.9e-10 MPa (6-level Romberg).

REQUIREMENT (the quadrature coupling): each field(eps) on the ladder must be
QUADRATURE-converged.  At the smallest eps the correction integrand is sharpest,
so n_quad must be large enough there (tensor n_quad~48 suffices to eps~0.025;
use the graded correction quadrature for smaller eps / higher Romberg orders).
"""

import numpy as np


def romberg_weights(levels, ratio=2.0):
    """Weights w[i] (i=0..levels-1) such that

        sum_i w[i] * field(eps / ratio**i)

    is the Romberg extrapolant of order O(eps**levels), for a field whose error
    is an integer power series field_exact + sum_{p>=1} A_p eps**p sampled on the
    geometric ladder eps_i = eps / ratio**i.

    levels=1 -> [1.0] (no extrapolation).  Weights sum to 1.
    """
    if levels < 1 or int(levels) != levels:
        raise ValueError(f"levels must be an integer >= 1; got {levels!r}")
    k = int(levels)
    # Carry the Romberg tableau one column at a time, each entry a weight vector
    # over the k samples.  col[i] = T[i][j]; the recursion
    #   T[i][j] = (f T[i][j-1] - T[i-1][j-1]) / (f-1),  f = ratio**j
    # reads only the previous column, so write into a fresh list.
    col = [np.eye(k)[i] for i in range(k)]                 # column j=0 (identity)
    for j in range(1, k):
        f = float(ratio) ** j
        newcol = list(col)
        for i in range(j, k):
            newcol[i] = (f * col[i] - col[i - 1]) / (f - 1.0)
        col = newcol
    return col[k - 1]


def eps_ladder(eps, levels, ratio=2.0):
    """The geometric eps-ladder [eps, eps/ratio, ..., eps/ratio**(levels-1)]."""
    return [eps / float(ratio) ** i for i in range(int(levels))]


def extrapolate_scalarfn(evalf, eps, levels=2, ratio=2.0):
    """Romberg-extrapolate a field-eval callable evalf(eps) -> ndarray (or any
    object supporting w*x and x+y, e.g. numpy/jax arrays).  levels=1 just returns
    evalf(eps)."""
    w = romberg_weights(levels, ratio)
    ladder = eps_ladder(eps, levels, ratio)
    acc = w[0] * evalf(ladder[0])
    for wi, ei in zip(w[1:], ladder[1:]):
        acc = acc + wi * evalf(ei)
    return acc


def extrapolate_dictfn(evalf, eps, levels=2, ratio=2.0, keys=None):
    """Romberg-extrapolate a callable evalf(eps) -> dict[str, ndarray], combining
    each key independently (valid because the field is linear in the per-eps
    influence kernels)."""
    w = romberg_weights(levels, ratio)
    ladder = eps_ladder(eps, levels, ratio)
    r0 = evalf(ladder[0])
    ks = list(r0.keys()) if keys is None else keys
    acc = {k: w[0] * r0[k] for k in ks}
    for wi, ei in zip(w[1:], ladder[1:]):
        r = evalf(ei)
        for k in ks:
            acc[k] = acc[k] + wi * r[k]
    return acc


def combine_samples(samples, ratio=2.0):
    """Romberg-combine an explicit list `samples` already evaluated on the ladder
    eps_i = eps0/ratio**i (samples[i] for i=0..len-1).  Returns sum_i w[i]*samples[i].
    Useful when the per-eps evaluations are cached/precomputed (e.g. JAX matvecs)."""
    w = romberg_weights(len(samples), ratio)
    acc = w[0] * samples[0]
    for wi, s in zip(w[1:], samples[1:]):
        acc = acc + wi * s
    return acc


# ---------------------------------------------------------------------------
def _selftest():
    # synthetic field with a clean power series: f(eps) = 7 + 3 eps + 5 eps^2
    #   - 2 eps^3 + 0.4 eps^4 ;  extrapolant order k should kill eps^1..eps^{k-1}.
    coeffs = [7.0, 3.0, 5.0, -2.0, 0.4, 0.1]   # [exact, A1, A2, A3, A4, A5]
    exact = coeffs[0]

    def f(eps):
        return np.array([sum(c * eps ** p for p, c in enumerate(coeffs))])

    eps0 = 0.1
    print("Romberg extrapolation self-test (synthetic clean eps-series):")
    ok = True
    for levels in range(1, 6):
        val = extrapolate_scalarfn(f, eps0, levels=levels)[0]
        err = abs(val - exact)
        # expected order: leading uncancelled term ~ eps0**levels
        expect = abs(coeffs[levels]) * eps0 ** levels if levels < len(coeffs) else 0.0
        good = err <= max(5 * expect, 1e-13) + 1e-12
        ok &= good
        print(f"  levels={levels}: val={val:.12f} err={err:.2e} "
              f"(expect ~{expect:.2e}) {'OK' if good else 'FAIL'}")
    w2 = romberg_weights(2)
    print(f"  weights(2) = {w2} (expect [-1, 2])  sum={w2.sum():.3f}")
    ok &= np.allclose(w2, [-1.0, 2.0]) and abs(w2.sum() - 1.0) < 1e-14
    print("  %s" % ("ALL PASS" if ok else "FAILURES"))
    return ok


if __name__ == "__main__":
    _selftest()
