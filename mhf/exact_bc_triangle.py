"""Exact-BC mollified Mindlin half-space displacement from a triangular
dislocation (DD), with MACHINE-PRECISION free-surface BC.

  u_i(obs) = INT_T C_jklm n_l (dG_ij/d src_m) du_k dA(src),
  G = G^{F,eps} (mollified Kelvin direct) + G^{C,eps} (exact-BC complementary,
      exact_bc_pointkernel / exact_bc_coeffs).

Split (same hybrid structure as tdhs_triangle, but with the EXACT-BC correction):
  * direct part  -> integrated ANALYTICALLY (analytical_kernels; on/near-fault OK);
  * complementary part -> SMOOTH (image point at depth +a>0, distance never small
    for obs in z<=0), here by Gauss-Legendre quadrature.

KEY PROPERTY: because T_3k(G^{eps,HS})(obs in z=0; src) = 0 for EVERY src, the
free-surface traction of the integrated triangle field is zero to machine
precision for ANY quadrature order -- the quadrature only affects bulk accuracy,
never the BC.  (Exactness of the analytic direct part's BC contribution + the
smooth correction's BC contribution both hold pointwise in src.)

The complementary point kernel is built symbolically once (G^{C,0}(x,y,z,c) with
x,y the obs-src horizontal offset, c=src_z, image depth a=sqrt(c^2+eps^2)); its
source derivatives are
    dG/d src_x = -dG/dx,  dG/d src_y = -dG/dy,  dG/d src_z = +dG/dc
(the last carries the chain rule through a(c) and the c-dependent coefficients).
"""

import numpy as np
import sympy as sp

from . import exact_bc_coeffs as _EC
from .analytical_kernels import (
    analytical_dd_displacement, analytical_stress_kernel, integrate_D2G,
)


def _strike_dip_basis(tri):
    """(V_normal, V_strike, V_dip) for a triangle, cutde convention."""
    Vnorm = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    Vnorm = Vnorm / np.linalg.norm(Vnorm)
    eY = np.array([0.0, 1.0, 0.0]); eZ = np.array([0.0, 0.0, 1.0])
    Vstrike = np.cross(eZ, Vnorm)
    if np.linalg.norm(Vstrike) == 0:
        Vstrike = eY * Vnorm[2]
    Vstrike = Vstrike / np.linalg.norm(Vstrike)
    Vdip = np.cross(Vnorm, Vstrike)
    return Vnorm, Vstrike, Vdip


def _stiffness_contract(dG, normal, mu, nu):
    """T[i,k] = C_kjpq n_j dG[i,p,q]  (slip index k pairs with the normal in
    the FIRST index pair of C), C_kjpq = lam d_kj d_pq + mu(d_kp d_jq + d_kq d_jp):
    T[i,k] = mu n_l dG[i,k,l] + mu n_l dG[i,l,k] + lam n_k dG[i,m,m]."""
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    n = normal
    T = np.zeros((3, 3))
    for i in range(3):
        for k in range(3):
            T[i, k] = (mu * sum(n[l] * dG[i, k, l] for l in range(3))
                       + mu * sum(n[l] * dG[i, l, k] for l in range(3))
                       + lam * n[k] * sum(dG[i, m, m] for m in range(3)))
    return T


def _stiffness_contract_DDG(DDG, normal, mu, nu):
    """S[i,p,k] = C_kjab n_j DDG[i,a,p,b] (same first-pair slip/normal pairing
    as _stiffness_contract)."""
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    n = normal
    S = np.zeros((3, 3, 3))
    for i in range(3):
        for p in range(3):
            for k in range(3):
                S[i, p, k] = (mu * sum(n[l] * DDG[i, k, p, l] for l in range(3))
                              + mu * sum(n[l] * DDG[i, l, p, k] for l in range(3))
                              + lam * n[k] * sum(DDG[i, m, p, m] for m in range(3)))
    return S

# ---------------------------------------------------------------------------
# Symbolic complementary point kernel G^{C,0}(x,y,z,c) and its source derivatives
# ---------------------------------------------------------------------------
_x, _y, _z, _c, _eps, _mu, _nu = sp.symbols("x y z c eps mu nu", real=True)
_a = sp.sqrt(_c**2 + _eps**2)
_CO = (_x, _y, _z)
_sa = sp.sqrt(_x**2 + _y**2 + (_z - _a)**2)
_g = 1 / _sa
_B = sp.log(_sa - (_z - _a))
_Qa = _sa - (_z - _a)
_gz = sp.diff(_g, _z)
_gx = sp.diff(_g, _x); _gy = sp.diff(_g, _y)
_Bx = sp.diff(_B, _x); _By = sp.diff(_B, _y)
_Cx = _x / _Qa; _Cy = _y / _Qa


def _pn(phi, Psi):
    F = phi + _x * Psi[0] + _y * Psi[1] + _z * Psi[2]
    return [(4 * (1 - _nu) * Psi[i] - sp.diff(F, _CO[i])) / (2 * _mu) for i in range(3)]


# basis fields per force direction, in exact_bc_pointkernel._BASIS order
_FIELDS = {
    2: [_pn(_g, [0, 0, 0]), _pn(_B, [0, 0, 0]), _pn(0, [0, 0, _g]), _pn(0, [0, 0, _gz])],
    0: [_pn(_gx, [0, 0, 0]), _pn(_Bx, [0, 0, 0]), _pn(_Cx, [0, 0, 0]),
        _pn(0, [_g, 0, 0]), _pn(0, [_gz, 0, 0]),
        _pn(0, [0, 0, _gx]), _pn(0, [0, 0, _Bx]), _pn(0, [0, 0, _Cx])],
    1: [_pn(_gy, [0, 0, 0]), _pn(_By, [0, 0, 0]), _pn(_Cy, [0, 0, 0]),
        _pn(0, [0, _g, 0]), _pn(0, [0, _gz, 0]),
        _pn(0, [0, 0, _gy]), _pn(0, [0, 0, _By]), _pn(0, [0, 0, _Cy])],
}
_CF = {2: _EC.coeffs_vertical_sym(_c, _eps, _nu),
       0: _EC.coeffs_horizontal_sym(_c, _eps, _nu),
       1: _EC.coeffs_horizontal_sym(_c, _eps, _nu)}

_Gc = sp.zeros(3, 3)
for _j in range(3):
    for _k, _fld in enumerate(_FIELDS[_j]):
        for _i in range(3):
            _Gc[_i, _j] += _CF[_j][_k] * _fld[_i]

# source-derivative tensors DG[i,j,m] = d G^{C}_{ij} / d src_m
_dGc_dsx = sp.Matrix(3, 3, lambda i, j: -sp.diff(_Gc[i, j], _x))
_dGc_dsy = sp.Matrix(3, 3, lambda i, j: -sp.diff(_Gc[i, j], _y))
_dGc_dsz = sp.Matrix(3, 3, lambda i, j: sp.diff(_Gc[i, j], _c))   # src_z = c
_args = (_x, _y, _z, _c, _eps, _mu, _nu)
_lam_dGc = [sp.lambdify(_args, m, "numpy") for m in (_dGc_dsx, _dGc_dsy, _dGc_dsz)]


def Gcorr_DG_source(obs, src, mu, nu, eps):
    """d G^{C}_{ij} / d src_m at (obs, src) -> (3,3,3) DG[i,j,m]."""
    obs = np.asarray(obs, float); src = np.asarray(src, float)
    X, Y, Z, C = obs[0] - src[0], obs[1] - src[1], obs[2], src[2]
    DG = np.zeros((3, 3, 3))
    for m in range(3):
        DG[:, :, m] = np.asarray(_lam_dGc[m](X, Y, Z, C, eps, mu, nu), float)
    return DG


# mixed 2nd derivatives d^2 G^{C}_{ij}/(d obs_p d src_m) for strain/stress.
# obs_p: d/dx,d/dy,d/dz ; src_m: -d/dx,-d/dy,+d/dc.  8 distinct sympy quantities.
_d2 = {k: sp.Matrix(3, 3, lambda i, j, kk=k: sp.diff(_Gc[i, j], *kk))
       for k in [(_x, _x), (_y, _y), (_x, _y), (_x, _z), (_y, _z),
                 (_x, _c), (_y, _c), (_z, _c)]}
_lam_d2 = {k: sp.lambdify(_args, m, "numpy") for k, m in _d2.items()}


def Gcorr_DDG_obs_source(obs, src, mu, nu, eps):
    """d^2 G^{C}_{ij}/(d obs_p d src_m) -> (3,3,3,3) DDG[i,j,p,m]."""
    obs = np.asarray(obs, float); src = np.asarray(src, float)
    X, Y, Z, C = obs[0] - src[0], obs[1] - src[1], obs[2], src[2]
    a = (X, Y, Z, C, eps, mu, nu)
    d = {k: np.asarray(f(*a), float) for k, f in _lam_d2.items()}
    DDG = np.zeros((3, 3, 3, 3))
    # obs d/dx,d/dy,d/dz ; src d/d src = (-d/dx, -d/dy, +d/dc)
    DDG[:, :, 0, 0] = -d[(_x, _x)]; DDG[:, :, 0, 1] = -d[(_x, _y)]; DDG[:, :, 0, 2] = d[(_x, _c)]
    DDG[:, :, 1, 0] = -d[(_x, _y)]; DDG[:, :, 1, 1] = -d[(_y, _y)]; DDG[:, :, 1, 2] = d[(_y, _c)]
    DDG[:, :, 2, 0] = -d[(_x, _z)]; DDG[:, :, 2, 1] = -d[(_y, _z)]; DDG[:, :, 2, 2] = d[(_z, _c)]
    return DDG


# ---------------------------------------------------------------------------
# Triangle DD displacement: analytic direct + quadrature on the smooth correction
# ---------------------------------------------------------------------------
def _tri_quad(n):
    """Symmetric-ish Gauss rule on the unit triangle via collapsed tensor product."""
    pts, wts = np.polynomial.legendre.leggauss(n)
    pts = 0.5 * (pts + 1.0); wts = 0.5 * wts
    xi1, xi2, w = [], [], []
    for i in range(n):
        for jx in range(n):
            xi1.append(pts[i]); xi2.append(pts[jx] * (1.0 - pts[i]))
            w.append(wts[i] * wts[jx] * (1.0 - pts[i]))
    return np.array(xi1), np.array(xi2), np.array(w)


def _corr_quad(obs, v1, v2, v3, eps, n_quad, graded=False, n_c=None, n_t=None):
    """Physical correction-quadrature points ys[Q,3] and weights wq[Q] (sum=area).
    graded=True clusters toward the obs near-singularity for surface-breaking
    accuracy at small eps (exact_bc_graded_quad); graded=False is the fixed
    collapsed tensor rule (current default behaviour)."""
    if graded:
        from .exact_bc_graded_quad import graded_quad
        return graded_quad(obs, v1, v2, v3, eps,
                           n_c if n_c is not None else n_quad,
                           n_t if n_t is not None else n_quad)
    xi1, xi2, wts = _tri_quad(n_quad)
    area2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    ys = ((1.0 - xi1 - xi2)[:, None] * v1 + xi1[:, None] * v2 + xi2[:, None] * v3)
    return ys, wts * area2


def disp_dd_kernel(obs, v1, v2, v3, normal, mu, nu, eps, n_quad=6,
                   graded=False, n_c=None, n_t=None):
    """U[i,k] s.t. u_i(obs) = U[i,k]*slip_cart_k for uniform Cartesian slip."""
    U = analytical_dd_displacement(obs, v1, v2, v3, normal, mu, nu, eps)   # direct
    ys, wq = _corr_quad(obs, v1, v2, v3, eps, n_quad, graded, n_c, n_t)
    for y, w in zip(ys, wq):
        DG = Gcorr_DG_source(obs, y, mu, nu, eps)
        U += w * _stiffness_contract(DG, normal, mu, nu)
    return U


def disp_dd(obs, v1, v2, v3, slip_cart, mu, nu, eps, n_quad=6):
    """Displacement at obs from uniform Cartesian slip on triangle (v1,v2,v3).
    Free-surface BC is exact up to the correction-quadrature error (-> 0 with
    n_quad), and exact at the point-kernel level."""
    normal = _strike_dip_basis(np.array([v1, v2, v3]))[0]
    U = disp_dd_kernel(obs, v1, v2, v3, normal, mu, nu, eps, n_quad)
    return U @ np.asarray(slip_cart, float)


# ---------------------------------------------------------------------------
# Strain / stress: analytic direct stress + quadrature on the smooth correction.
# ---------------------------------------------------------------------------
def stress_dd_kernel(obs, v1, v2, v3, normal, mu, nu, eps, n_quad=6,
                     graded=False, n_c=None, n_t=None):
    """H[m,n,k] s.t. sigma_mn(obs) = H[m,n,k]*slip_cart_k."""
    H = analytical_stress_kernel(obs, v1, v2, v3, normal, mu, nu, eps)   # direct
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    ys, wq = _corr_quad(obs, v1, v2, v3, eps, n_quad, graded, n_c, n_t)
    Gdu = np.zeros((3, 3, 3))                       # grad_u_corr[i, p, k]
    for y, w in zip(ys, wq):
        DDG = Gcorr_DDG_obs_source(obs, y, mu, nu, eps)
        Gdu += w * _stiffness_contract_DDG(DDG, normal, mu, nu)
    for k in range(3):
        gu = Gdu[:, :, k]
        e = 0.5 * (gu + gu.T)
        H[:, :, k] += lam * np.trace(e) * np.eye(3) + 2.0 * mu * e
    return H


def grad_dd_kernel(obs, v1, v2, v3, normal, mu, nu, eps, n_quad=6,
                   graded=False, n_c=None, n_t=None):
    """Full displacement-gradient kernel Gd[i,p,k] = d u_i/d obs_p per slip_k.
    Direct part via -integrate_D2G (obs.src mixed deriv = -d^2/dd^2 of Kelvin);
    correction by quadrature.  (Strain = sym, stress = Hooke; consistent with
    stress_dd_kernel.)"""
    D2G = integrate_D2G(v1, v2, v3, obs, mu, nu, eps)           # [i,j,p,m]=d2G/dd_p dd_m
    Gd = _stiffness_contract_DDG(-D2G, normal, mu, nu)          # direct  -> [i,p,k]
    ys, wq = _corr_quad(obs, v1, v2, v3, eps, n_quad, graded, n_c, n_t)
    for y, w in zip(ys, wq):
        DDG = Gcorr_DDG_obs_source(obs, y, mu, nu, eps)
        Gd += w * _stiffness_contract_DDG(DDG, normal, mu, nu)
    return Gd


def stress_dd(obs, v1, v2, v3, slip_cart, mu, nu, eps, n_quad=6):
    """Stress tensor (3,3) at obs from uniform Cartesian slip."""
    normal = _strike_dip_basis(np.array([v1, v2, v3]))[0]
    H = stress_dd_kernel(obs, v1, v2, v3, normal, mu, nu, eps, n_quad)
    return np.einsum("mnk,k->mn", H, np.asarray(slip_cart, float))


def strain_dd(obs, v1, v2, v3, slip_cart, mu, nu, eps, n_quad=6):
    """Strain tensor (3,3) via Hooke inverse of stress_dd."""
    sig = stress_dd(obs, v1, v2, v3, slip_cart, mu, nu, eps, n_quad)
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    tr_e = np.trace(sig) / (3.0 * lam + 2.0 * mu)
    return (sig - lam * tr_e * np.eye(3)) / (2.0 * mu)


def surface_traction(obs_xy, v1, v2, v3, slip_cart, mu, nu, eps, n_quad=6):
    """Free-surface traction T_3k = sigma_{k3} at a surface point (z=0), via the
    ANALYTIC stress kernel -- the true measure of the BC residual."""
    obs = np.array([obs_xy[0], obs_xy[1], 0.0])
    sig = stress_dd(obs, v1, v2, v3, slip_cart, mu, nu, eps, n_quad)
    return sig[:, 2]

