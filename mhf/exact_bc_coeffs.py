"""Closed-form Papkovich-Neuber coefficients for the exact-BC mollified Mindlin
point kernel, as functions of (c, eps, nu) with a = sqrt(c^2 + eps^2).

Derived by exact polynomial-coefficient matching of the total surface traction
(see _closedform_j2.py / _closedform_j0.py), cross-checked to machine precision
against the per-source numeric solve.  mu-independent (the complementary field
scales with the direct field).  Returned in the order of exact_bc_pointkernel._BASIS.

Vertical (j=2):  [A0(g), AB(B), P0(g in Psi_z), P1(gz in Psi_z)]
Horizontal (j=0):[A0(gx), AB(Bx), AC(Cx), PX0(g), PX1(gz), PZ0(gx), PZB(Bx), PZC(Cx)]
Horizontal (j=1): same values (x<->y symmetry).

The core expressions are backend-agnostic (pure arithmetic + sqrt + pi), so the
same formulas serve both numeric (numpy) evaluation and symbolic (sympy) use in
the triangle integrator -- single source of truth.
"""

import numpy as np


def _vertical(c, eps, nu, sqrt, pi):
    a = sqrt(c * c + eps * eps)
    e2 = eps * eps
    den_n1 = 8 * pi * (nu - 1)
    den_q = 8 * pi * (2 * nu**2 - 3 * nu + 1)        # 8 pi (2nu-1)(nu-1)
    A0 = (2*a**3*nu - 2*a**3 - a**2*c*nu - 2*a*e2*nu**2 + 2*a*e2*nu
          - c**3*nu + c**3 + 2*c*e2*nu**2 - 4*c*e2*nu + 2*c*e2) / (a**2 * den_n1)
    AB = (4*a**3*nu - 2*a**3 - 4*a**2*c*nu + a**2*c + c**3
          - 2*c*e2*nu + 2*c*e2) / (8 * pi * a**3)
    P0 = (4*a**3*nu**2 - 4*a**3*nu + a**3 - 4*a**2*c*nu**2 + 5*a**2*c*nu - a**2*c
          + c**3*nu - c**3 - 2*c*e2*nu**2 + 4*c*e2*nu - 2*c*e2) / (a**3 * den_q)
    P1 = (-2*a**3*nu + a**3 + a**2*c*nu + 2*a*e2*nu**2 - a*e2*nu
          + c**3*nu - c**3 - 2*c*e2*nu**2 + 4*c*e2*nu - 2*c*e2) / (a**2 * den_q)
    return [A0, AB, P0, P1]


def _horizontal(c, eps, nu, sqrt, pi):
    a = sqrt(c * c + eps * eps)
    e2 = eps * eps
    den_n1 = 16 * pi * (nu - 1)
    A0 = c * e2 / (a * den_n1)
    AB = (4*a**3*nu - 2*a**3 - 4*a**2*c*nu + 2*a**2*c + 4*a*e2*nu**2 - 6*a*e2*nu
          + 2*a*e2 - 4*c*e2*nu**2 + 4*c*e2*nu - c*e2) / (a**2 * den_n1)
    AC = (8*a**3*nu**2 - 8*a**3*nu + 2*a**3 - 8*a**2*c*nu**2 + 8*a**2*c*nu
          - 2*a**2*c - 4*c*e2*nu**2 + 4*c*e2*nu - c*e2) / (a**3 * den_n1)
    PX0 = c * (2*a**2 + e2) / (a**3 * den_n1)
    PX1 = c * e2 / (a**2 * den_n1)
    PZ0 = (2*a**3 - 2*a**2*c + 2*a*e2*nu - 2*a*e2 - 2*c*e2*nu + c*e2) / (a**2 * den_n1)
    PZB = (-4*a**3*nu + 2*a**3 + 4*a**2*c*nu - 2*a**2*c + 2*c*e2*nu - c*e2) / (a**3 * den_n1)
    PZC = c * 0          # identically zero (kept for slot alignment, backend-safe)
    return [A0, AB, AC, PX0, PX1, PZ0, PZB, PZC]


# ----- numeric (numpy) -----
def coeffs_vertical(c, eps, nu):
    return np.array(_vertical(c, eps, nu, np.sqrt, np.pi))


def coeffs_horizontal(c, eps, nu):
    return np.array(_horizontal(c, eps, nu, np.sqrt, np.pi))


def coeffs(j, c, eps, nu):
    """PN coefficients for force direction j, in exact_bc_pointkernel._BASIS order."""
    if j == 2:
        return coeffs_vertical(c, eps, nu)
    return coeffs_horizontal(c, eps, nu)      # j=0 and j=1 share values


# ----- symbolic (sympy): pass sympy symbols c, eps, nu -----
def coeffs_vertical_sym(c, eps, nu):
    import sympy as sp
    return _vertical(c, eps, nu, sp.sqrt, sp.pi)


def coeffs_horizontal_sym(c, eps, nu):
    import sympy as sp
    return _horizontal(c, eps, nu, sp.sqrt, sp.pi)
