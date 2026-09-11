"""Single source of truth for mbem tolerances and thresholds.

Every numeric default a solver, compressor, or pipeline uses must live
here, so the whole stack can be audited (and a test can assert nothing
drifts). Values follow the approved plan: solution accuracy target ~1e-6,
linear-algebra layers run with ~100x margin under it.
"""

# --- Accuracy targets -------------------------------------------------
SOLUTION_RTOL = 1e-6          # end-to-end solver accuracy target
GMRES_RTOL = 1e-8             # true-residual stop (100x margin)
GMRES_RESTART = 200
GMRES_MAXITER = 600

# --- Compression ------------------------------------------------------
BLOCK_COMPRESSION_TOL = 1e-8  # rel-Frobenius per admissible block
CLUSTER_MIN_LEAF = 32         # elements per leaf cluster
ADMISSIBILITY_ETA = 2.0
# Admissible blocks smaller than this (elements per side) are stored
# dense: at small sizes the epsilon-rank is a large fraction of the
# block and cross approximation cannot be certified by sampling.
ACA_MIN_BLOCK = 64
# ACA must converge within this fraction of full element rank, else the
# block is declared not-low-rank and falls back to dense evaluation.
ACA_MAX_RANK_FRACTION = 1.0 / 3.0

# --- Dense fallbacks --------------------------------------------------
MAX_DENSE_PRECOND_DOF = 9000  # exact dense LU below this, per block

# --- HODLR ladder rung ------------------------------------------------
HODLR_PRECOND_TOL = 1e-2      # loose tol when used as a preconditioner
HODLR_LEAF_ELEMS = 96         # dense leaf size (elements)

# --- Basis-recombination parity gates (Phase 1) -----------------------
# Direct vs basis assembly differ only in floating-point evaluation
# order. At moderate nu the agreement is machine precision; near the
# fluid limit (nu -> 1/2) the lam*C1 coefficient amplifies cancellation
# between the L[P1]/L[P2] basis terms by ~|lam*C1|/|c_M2|, so the gate
# loosens accordingly (still far below any physical tolerance).
BASIS_PARITY_RTOL = 1e-13
BASIS_PARITY_RTOL_NEAR_FLUID = 1e-10

# --- Mollification ----------------------------------------------------
EPS_OVER_H = 1.25             # per-element eps_j = EPS_OVER_H * h_j (opt-in)
