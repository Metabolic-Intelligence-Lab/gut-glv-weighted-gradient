"""
Irreversibility (FDT-violation) index directly from the inferred human matrix
=============================================================================

The two-time / FDT appendix defines the non-equilibrium order parameter
    I = ||Omega||_F / (T sqrt S),   Omega = A C + T I  (antisymmetric),
with C the stationary covariance solving the Lyapunov equation
    A C + C A^T = -2T I     (needs A Hurwitz / feasible).
I(gamma)=0 exactly at the symmetric/equilibrium point gamma=1, and grows as gamma
leaves 1 (dmft_fdt.py computes the synthetic I(gamma) curve).

This script reads I DIRECTLY from the inferred human QMP community matrix
(Vandeputte 2021, absolute flow-cytometry load), the same matrix whose symmetry is
gamma_hat ~ 0.10 in Fig. fig:data(d). Two robustness points:

  1. SCALE INVARIANCE. Under A -> alpha A the Lyapunov C -> C/alpha, so
     Omega = A C + T I is invariant and I is unchanged. Hence I does NOT depend on
     the absolute disorder strength sigma -- the one coordinate the inference cannot
     identify. The irreversibility placement is therefore sigma-free.
  2. DIRECT, FEASIBLE. The inferred human matrix is stable (max Re lambda < 0), so
     the Lyapunov C exists and I_data is well defined without any stabilisation.

Caveat (quantified here): like gamma_hat, the ABSOLUTE value of I_data inherits the
inference bias and the assumed ensemble. A calibration that pushes synthetic GLV
series of known gamma through the identical fit shows the inferred I sits ABOVE the
true I, so I_data is read qualitatively (a finite distance from equilibrium), not as
a calibrated coordinate.

Run:  python3 irreversibility_from_data.py
"""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov, eigvals
import networkx as nx
import warnings; warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

from infer_and_place import fit_matrix, disorder_stats, S_TAXA
from infer_vandeputte import load_vandeputte, design

T = 1.0


def irreversibility(A, T=T):
    """I = ||Omega||_F/(T sqrt S), Omega = A C + T I; requires A Hurwitz."""
    C = solve_continuous_lyapunov(A, -2 * T * np.eye(A.shape[0]))
    C = 0.5 * (C + C.T)
    Om = A @ C + T * np.eye(A.shape[0])
    Om = 0.5 * (Om - Om.T)                       # exact Omega is antisymmetric
    return np.linalg.norm(Om, "fro") / (T * np.sqrt(A.shape[0]))


def synth_dense_A(gamma, g=0.35, S=400, seed=1):
    """Dense mean-field A=a-I (the ensemble of dmft_fdt.py's I(gamma) curve)."""
    rng = np.random.default_rng(seed); a = np.zeros((S, S)); sd = g / np.sqrt(S)
    iu = np.triu_indices(S, 1)
    z1 = rng.standard_normal(len(iu[0])); z2 = rng.standard_normal(len(iu[0]))
    a[iu] = sd * z1; a.T[iu] = sd * (gamma * z1 + np.sqrt(max(1 - gamma**2, 0.0)) * z2)
    return a - np.eye(S)


def synth_glv_series(gamma_true, S=30, nsub=20, ndays=40, seed=7):
    """Scale-free GLV time series with known gamma, matched to the data design."""
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(S, 2, seed=seed); Adj = nx.to_numpy_array(G)
    iu = np.where(np.triu(Adj, 1) > 0); ne = len(iu[0])
    sd = 0.9 / np.sqrt(max(Adj.sum(1).mean(), 1)); J = np.zeros((S, S))
    z1 = rng.normal(size=ne); z2 = rng.normal(size=ne)
    J[iu] = sd * z1; J[(iu[1], iu[0])] = sd * (gamma_true * z1 + np.sqrt(max(1 - gamma_true**2, 0)) * z2)
    np.fill_diagonal(J, -1.0); r = rng.uniform(0.5, 1.5, S)
    Xall, pall, dall = [], [], []
    for sb in range(nsub):
        x = rng.uniform(0.1, 1.0, S); rec = []
        for t in range(ndays):
            for _ in range(5):
                x = np.clip(x + 0.02 * x * (r + J @ x) + 1e-3, 0, 50)
            rec.append(x.copy())
        Xall.append(np.array(rec)); pall.append(np.full(ndays, sb)); dall.append(np.arange(ndays))
    return np.vstack(Xall), np.concatenate(dall).astype(float), np.concatenate(pall).astype(str), J


def main():
    # ---- (1) inferred human QMP matrix -> I_data --------------------------
    X, day, pid = load_vandeputte()
    prop = X / X.sum(1, keepdims=True)
    top = np.argsort(prop.mean(0))[::-1][:S_TAXA]
    Y, Z = design(X, day, pid, use_load=True, top=top)
    A = fit_matrix(Y, Z)                                # diagonal normalised to ~ -1
    _, sig, gam = disorder_stats(A)
    maxRe = float(np.max(eigvals(A).real))
    feasible = maxRe < 0
    print(f"[human QMP]  S={A.shape[0]}  gamma_hat={gam:+.3f}  sigma_hat={sig:.2f}  "
          f"max Re lambda={maxRe:+.3f}  ({'feasible' if feasible else 'NON-FEASIBLE'})")
    if feasible:
        I_data = irreversibility(A)
        chk = irreversibility(3.7 * A)                  # scale-invariance
        print(f"[human QMP]  I_data = {I_data:.3f}   "
              f"(scale-invariance A vs 3.7A: {I_data:.4f} = {chk:.4f})")

    # ---- (2) synthetic dense reference I(gamma) ---------------------------
    print("\n[dense mean-field reference I(gamma)]")
    for gm in (1.0, 0.5, 0.2, 0.1, 0.0):
        print(f"   gamma={gm:+.2f}  I={irreversibility(synth_dense_A(gm)):.3f}")

    # ---- (3) calibration: the inference inflates I ------------------------
    print("\n[calibration through the identical pipeline: I_hat vs I_true]")
    for gt in (0.0, 0.2):
        Xs, ds, ps, Jt = synth_glv_series(gt)
        tp = np.argsort((Xs / Xs.sum(1, keepdims=True)).mean(0))[::-1][:30]
        Ys, Zs = design(Xs, ds, ps, use_load=True, top=tp)
        Jh = fit_matrix(Ys, Zs); _, _, gh = disorder_stats(Jh)
        It = irreversibility(Jt) if np.max(eigvals(Jt).real) < 0 else float("nan")
        Ih = irreversibility(Jh) if np.max(eigvals(Jh).real) < 0 else float("nan")
        print(f"   gamma_true={gt:+.2f} -> gamma_hat={gh:+.3f} | I_true={It:.3f}  I_hat={Ih:.3f}")
    print("\n=> I_data is nonzero and far from the equilibrium value I=0 (gamma=1),")
    print("   scale-invariant (sigma-free); its absolute value is inflated by the fit,")
    print("   so it is read qualitatively as a finite distance from equilibrium.")


if __name__ == "__main__":
    main()
