"""mhf validation suite.

Run:  JAX_PLATFORMS=cpu python -m mhf.validate     (from the repo root)

Checks (external oracle = cutde.halfspace where available):
  1. eps->0 convergence to the unmollified half-space solution at O(eps^2),
     for nu in {0.25, 0.30, 0.35} (the nu-sweep guards the lam/mu pairing),
     at INTERIOR observation depths.
  2. Machine-precision free-surface BC (analytic stress at z=0), tensor and
     graded rules, buried and surface-breaking triangles.
  3. stress == Hooke(sym(grad_u)) internal consistency.
  4. Graded rule == converged tensor rule on buried AND horizontal triangles
     (the horizontal case guards the zero-weight-rule fix).
  5. Displacement mu-invariance.
  6. Richardson levels=2 + graded beats levels=1 near a surface-breaking
     trace (rate ~O(eps^2) vs ~O(eps)).
  7. JAX twin parity (skipped if jax unavailable).
  8. Anelastic eigenstress subtraction: off-fault no-op, on-fault raw ~1/eps
     vs corrected bounded, exact wiring identity through eval_fields AND
     eval_fields_fault_aware, on-fault marginal magnitude (3/4) mu s / eps.
"""

import numpy as np

try:                                   # python -m mhf.validate
    from .mindlin import disp_hs_mindlin, field_hs_mindlin, _strike_dip_basis
    from .exact_bc_triangle import (disp_dd_kernel, stress_dd_kernel,
                                    grad_dd_kernel)
    from .exact_bc_graded_quad import graded_quad
    from .fields import (eval_fields, eval_fields_fault_aware, make_rect_fault,
                         _slip_cart_per_tri, MU)
    from .anelastic import eigenstress_at_points
except ImportError:                    # python mhf/validate.py
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mhf.mindlin import (disp_hs_mindlin, field_hs_mindlin,
                             _strike_dip_basis)
    from mhf.exact_bc_triangle import (disp_dd_kernel, stress_dd_kernel,
                                       grad_dd_kernel)
    from mhf.exact_bc_graded_quad import graded_quad
    from mhf.fields import (eval_fields, eval_fields_fault_aware,
                            make_rect_fault, _slip_cart_per_tri, MU)
    from mhf.anelastic import eigenstress_at_points

TRI_BURIED = np.array([[0.0, -1.0, -2.5], [0.0, 1.2, -2.2], [0.0, 0.1, -0.9]])
TRI_SURF = np.array([[0.0, -1.0, -2.0], [0.0, 1.0, -2.0], [0.0, 0.0, 0.0]])
TRI_HORIZ = np.array([[-1.0, -1.0, -2.0], [1.0, -1.0, -2.0], [0.0, 1.0, -2.0]])


def _report(name, val, tol, extra=""):
    ok = bool(val < tol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {val:.3e} (tol {tol:.0e}) {extra}")
    return ok


def _cutde():
    try:
        import cutde.halfspace as hs
        return hs
    except Exception:
        return None


def check_eps_convergence():
    hs = _cutde()
    if hs is None:
        print("  [SKIP] eps->0 vs cutde (cutde not installed)")
        return True
    ok = True
    obs = np.array([0.8, 0.4, -1.2])           # interior depth
    slip = np.array([1.0, 0.0, 0.0])
    for nu in (0.25, 0.30, 0.35):
        uref = hs.disp(obs[None, :], TRI_BURIED[None, :, :], slip[None, :], nu)[0]
        errs = [np.abs(disp_hs_mindlin(obs, TRI_BURIED, [1, 0, 0], nu, e) - uref).max()
                for e in (0.2, 0.1, 0.05)]
        rate = float(np.mean([errs[i] / errs[i + 1] for i in range(2)]))
        ok &= _report(f"eps->0 nu={nu}", errs[-1], 1e-3, f"rate/halving={rate:.2f}")
        ok &= rate > 3.0
    return ok


def check_bc():
    ok = True
    for label, tri, kw in [
        ("buried/tensor", TRI_BURIED, {}),
        ("buried/graded", TRI_BURIED, dict(graded=True, n_c=20, n_t=16)),
        ("horizontal/graded", TRI_HORIZ, dict(graded=True, n_c=20, n_t=16)),
        # surface-breaking trace: the graded BC residual is quadrature-limited
        # and converges spectrally in (n_c, n_t): 1.3e-10 at 20x16,
        # 7.8e-13 at 24x20, 7.4e-16 at 32x24.
        ("surface-breaking/graded", TRI_SURF, dict(graded=True, n_c=24, n_t=20)),
    ]:
        n = _strike_dip_basis(tri)[0]
        H = stress_dd_kernel(np.array([0.4, 0.3, 0.0]), tri[0], tri[1], tri[2],
                             n, 1.0, 0.30, 0.1, 16, **kw)
        ok &= _report(f"BC |T_3k(z=0)| {label}", np.abs(H[:, 2, :]).max(), 1e-10)
    return ok


def check_stress_consistency():
    ok = True
    obs = np.array([0.8, 0.4, -1.2])
    for nu in (0.25, 0.35):
        n = _strike_dip_basis(TRI_BURIED)[0]
        H = stress_dd_kernel(obs, *TRI_BURIED, n, 1.0, nu, 0.1, 8)
        G = grad_dd_kernel(obs, *TRI_BURIED, n, 1.0, nu, 0.1, 8)
        lam = 2 * nu / (1 - 2 * nu)
        worst = 0.0
        for k in range(3):
            ee = 0.5 * (G[:, :, k] + G[:, :, k].T)
            worst = max(worst, np.abs(
                H[:, :, k] - lam * np.trace(ee) * np.eye(3) - 2 * ee).max())
        ok &= _report(f"stress==Hooke(grad) nu={nu}", worst, 1e-12)
    return ok


def check_graded_vs_tensor():
    ok = True
    obs = np.array([0.4, 0.3, -1.2])
    for label, tri in [("buried", TRI_BURIED), ("horizontal", TRI_HORIZ)]:
        n = _strike_dip_basis(tri)[0]
        Ug = disp_dd_kernel(obs, *tri, n, 1.0, 0.25, 0.1, 16,
                            graded=True, n_c=20, n_t=16)
        Ut = disp_dd_kernel(obs, *tri, n, 1.0, 0.25, 0.1, 24)
        ok &= _report(f"graded==tensor disp ({label})", np.abs(Ug - Ut).max(), 1e-10)
    area = 0.5 * np.linalg.norm(np.cross(TRI_HORIZ[1] - TRI_HORIZ[0],
                                         TRI_HORIZ[2] - TRI_HORIZ[0]))
    _, wq = graded_quad(obs, *TRI_HORIZ, 0.1, 20, 16)
    ok &= _report("horizontal weight-sum contract", abs(wq.sum() - area) / area, 1e-10)
    return ok


def check_mu_invariance():
    obs = np.array([[1.5, 0.8, -1.0]])
    u1 = np.atleast_2d(disp_hs_mindlin(obs, TRI_BURIED, [1, 0, 0], 0.25, 0.2, mu=1.0))
    u30 = np.atleast_2d(disp_hs_mindlin(obs, TRI_BURIED, [1, 0, 0], 0.25, 0.2, mu=30.0))
    return _report("displacement mu-invariance", np.abs(u1 - u30).max(), 1e-13)


def check_richardson():
    """Near-trace stress for a 10x10 km SURFACE-BREAKING strike-slip fault
    (top edge on z=0, fault plane y=0), obs 2 km from the trace on the free
    surface — the documented regime where a single eval is intrinsically
    O(eps) and Richardson levels=2 restores >= O(eps^2)."""
    hs = _cutde()
    if hs is None:
        print("  [SKIP] Richardson near-trace rates (cutde not installed)")
        return True
    T1 = np.array([[-5.0, 0.0, -10.0], [5.0, 0.0, -10.0], [5.0, 0.0, 0.0]])
    T2 = np.array([[-5.0, 0.0, -10.0], [5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    obs = np.array([0.0, 2.0, 0.0])
    slip = np.array([1.0, 0.0, 0.0])
    nu, mu = 0.25, 1.0
    e6 = np.zeros(6)
    for T in (T1, T2):
        e6 += np.asarray(hs.strain(obs[None].astype(float),
                                   T[None].astype(float), slip[None], nu))[0]
    sref_xy = float(np.asarray(hs.strain_to_stress(e6[None], mu, nu))[0][3])
    rates, last = {}, {}
    for levels in (1, 2):
        errs = []
        for eps in (1.0, 0.5, 0.25):
            sig = np.zeros((3, 3))
            for T in (T1, T2):
                f = field_hs_mindlin(obs, T, slip, nu, eps, mu=mu, n_quad=16,
                                     graded=True, n_c=20, n_t=16,
                                     richardson_levels=levels)
                sig += f["stress"]
            errs.append(abs(sig[0, 1] - sref_xy))
        rates[levels] = float(np.mean([errs[i] / errs[i + 1] for i in range(2)]))
        last[levels] = errs[-1]
        print(f"        levels={levels}: errs="
              f"{['%.2e' % x for x in errs]} rate/halving={rates[levels]:.2f}")
    ok = (1.5 < rates[1] < 3.2 and rates[2] > 3.5
          and last[2] < last[1] / 5.0)         # O(eps) -> >= O(eps^2)
    print(f"  [{'PASS' if ok else 'FAIL'}] Richardson L2 restores O(eps^2) near trace")
    return ok


def check_jax_parity():
    try:
        from .exact_bc_jax import make_kernels
    except Exception as exc:  # noqa: BLE001
        print(f"  [SKIP] JAX parity (jax unavailable: {exc})")
        return True
    ok = True
    obs = np.array([0.8, 0.4, -1.2])
    n = _strike_dip_basis(TRI_BURIED)[0]
    for label, kw in [("tensor", dict(n_quad=8)),
                      ("graded", dict(n_quad=16, graded=True, n_c=20, n_t=16))]:
        K = make_kernels(**kw)
        nq = kw.get("n_quad", 8)
        Unp = disp_dd_kernel(obs, *TRI_BURIED, n, 1.0, 0.30, 0.1, nq,
                             graded=kw.get("graded", False),
                             n_c=kw.get("n_c"), n_t=kw.get("n_t"))
        Ujx = np.asarray(K["disp_U"](obs, *TRI_BURIED, 1.0, 0.30, 0.1))
        ok &= _report(f"JAX disp parity ({label}, nu=0.30)",
                      np.abs(Unp - Ujx).max(), 1e-12)
        Hnp = stress_dd_kernel(obs, *TRI_BURIED, n, 1.0, 0.30, 0.1, nq,
                               graded=kw.get("graded", False),
                               n_c=kw.get("n_c"), n_t=kw.get("n_t"))
        Hjx = np.asarray(K["stress_H"](obs, *TRI_BURIED, 1.0, 0.30, 0.1))
        ok &= _report(f"JAX stress parity ({label}, nu=0.30)",
                      np.abs(Hnp - Hjx).max(), 1e-12)
    return ok


def check_eigenstress_subtraction():
    """Anelastic eigenstress subtraction (mhf.anelastic wired into the batch
    evaluators): the raw kernel stress is TOTAL (C:eps_star included; on-fault
    slip-sense shear (3/4) mu s / eps, diverging as eps->0); the default
    subtract_anelastic=True must return the bounded ELASTIC stress."""
    tris = make_rect_fault(half_len=5.0, depth=10.0)   # surface-breaking, y=0
    slip = (1.0e-3, 0.0, 0.0)                          # 1 m strike-slip
    nu = 0.25
    slip_cart = _slip_cart_per_tri(tris, slip)
    kw = dict(slip=slip, nu=nu, n_quad=8, graded=True, n_c=20, n_t=16)
    obs_on = np.array([[0.0, 0.0, -5.0]])              # fault-interior center
    obs_off = np.array([[0.0, 5.0, -5.0]])             # 5 km off the plane
    ok = True

    # (a) off-fault no-op: eigenstress negligible vs total
    e = 0.5
    _, s_tot = eval_fields(obs_off, tris, eps=e, subtract_anelastic=False, **kw)
    star = eigenstress_at_points(obs_off, tris, slip_cart, MU, nu, e)
    ok &= _report("off-fault |sig*|/|sig_total|",
                  float(np.abs(star).max() / np.abs(s_tot).max()), 1e-2)

    # (b) on-fault boundedness: raw grows ~1/eps, corrected stays flat
    raw, cor = [], []
    for e in (1.0, 0.25):
        _, st = eval_fields(obs_on, tris, eps=e, subtract_anelastic=False, **kw)
        _, se = eval_fields(obs_on, tris, eps=e, subtract_anelastic=True, **kw)
        raw.append(abs(float(st[0, 0, 1])))
        cor.append(abs(float(se[0, 0, 1])))
    raw_growth, cor_growth = raw[1] / raw[0], cor[1] / cor[0]
    print(f"        on-fault |sxy| raw: {raw[0]:.3e} -> {raw[1]:.3e} "
          f"(x{raw_growth:.2f});  corrected: {cor[0]:.3e} -> {cor[1]:.3e} "
          f"(x{cor_growth:.2f})")
    bounded = (raw_growth > 2.5) and (cor_growth < 1.5) and (cor[1] < raw[1])
    print(f"  [{'PASS' if bounded else 'FAIL'}] on-fault corrected bounded, "
          f"raw ~ 1/eps")
    ok &= bounded

    # (c) wiring identity: total - elastic == eigenstress, exactly, through
    #     BOTH evaluators (fault-aware guards the blend-then-subtract order)
    e = 1.0
    star_on = eigenstress_at_points(obs_on, tris, slip_cart, MU, nu, e)
    _, st = eval_fields(obs_on, tris, eps=e, subtract_anelastic=False, **kw)
    _, se = eval_fields(obs_on, tris, eps=e, subtract_anelastic=True, **kw)
    rel = np.abs((st - se) - star_on).max() / np.abs(star_on).max()
    ok &= _report("wiring identity (eval_fields)", float(rel), 1e-12)
    _, ft = eval_fields_fault_aware(obs_on, tris, eps=e, richardson_levels=2,
                                    subtract_anelastic=False, **kw)
    _, fe = eval_fields_fault_aware(obs_on, tris, eps=e, richardson_levels=2,
                                    subtract_anelastic=True, **kw)
    rel = np.abs((ft - fe) - star_on).max() / np.abs(star_on).max()
    ok &= _report("wiring identity (fault-aware, L2)", float(rel), 1e-12)

    # (d) on-fault marginal magnitude |sig*_xy(0)| = (3/4) mu s / eps, and the
    #     eigenstress carries the same slip sense as the total on-fault shear
    #     (make_rect_fault normals are +y, so V_strike = z_hat x y_hat = -x_hat
    #     and both are NEGATIVE for positive strike slip here)
    s_mag = float(np.linalg.norm(slip_cart[0]))
    for e in (1.0, 0.5):
        star_xy = float(eigenstress_at_points(obs_on, tris, slip_cart,
                                              MU, nu, e)[0, 0, 1])
        expect = 0.75 * MU * s_mag / e
        ok &= _report(f"on-fault |sig*_xy| vs (3/4) mu s/eps (eps={e})",
                      abs(abs(star_xy) - expect) / expect, 2e-2)
    _, st = eval_fields(obs_on, tris, eps=1.0, subtract_anelastic=False, **kw)
    star1 = float(eigenstress_at_points(obs_on, tris, slip_cart,
                                        MU, nu, 1.0)[0, 0, 1])
    sign_ok = star1 * float(st[0, 0, 1]) > 0.0
    print(f"  [{'PASS' if sign_ok else 'FAIL'}] eigenstress sign matches "
          f"on-fault total slip-sense shear")
    return ok and sign_ok


def main():
    print("=" * 74)
    print(" mhf validation: exact-BC mollified Mindlin (clean build)")
    print("=" * 74)
    groups = [
        ("eps->0 convergence vs cutde (nu sweep)", check_eps_convergence),
        ("free-surface BC", check_bc),
        ("stress/grad consistency", check_stress_consistency),
        ("graded vs tensor (incl. horizontal)", check_graded_vs_tensor),
        ("mu-invariance", check_mu_invariance),
        ("Richardson near-trace", check_richardson),
        ("JAX parity", check_jax_parity),
        ("anelastic eigenstress subtraction", check_eigenstress_subtraction),
    ]
    results = []
    for name, fn in groups:
        print(f"({len(results) + 1}) {name}")
        results.append(bool(fn()))
    print("=" * 74)
    print(f"  RESULT: {sum(results)}/{len(results)} groups passed")
    print("=" * 74)
    return all(results)


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
