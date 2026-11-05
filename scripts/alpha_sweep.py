r"""
The real trade-off: settling time against cross-response.
=========================================================
Strengthening self-limitation by a factor alpha on the diagonal leaves every off-diagonal
coupling, and therefore f_sign and r_cycle, exactly unchanged, so the hypothesis under test is
untouched. It changes two things that matter for the experiment: the slowest relaxation rate,
which sets how long the community must be held before a steady state can be read, and the
size of the susceptibility cross-responses, which sets the abundance precision needed.
"""
import os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
repo = sys.argv[1] if len(sys.argv) > 1 else "."
sys.path.insert(0, os.path.join(repo, "scripts")); os.chdir(repo)
from constrained_vs_free_glv import load_communities, transitions
from descent_specificity import fit_constrained_fast_d
from symmetrizability_invariants import invariants
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
V = np.loadtxt("data/venturelli2018/interaction_matrix_averaged.csv", delimiter=",", skiprows=1)
comms = load_communities(); n = len(V)
xstar = np.maximum(comms[0][1].mean(0), 0.03); xstar = xstar / xstar.sum()
def scaled(J, a):
    K = J.copy(); np.fill_diagonal(K, np.diag(J)*a); return K
def full_symmetrizable(J):
    iu = np.triu_indices(len(J),1); a,b = J[iu].copy(), J.T[iu].copy()
    inter=(a!=0)|(b!=0); ii,jj=iu[0][inter],iu[1][inter]; aa,bb=a[inter],b[inter]
    s=np.where(np.abs(aa)>=np.abs(bb),np.sign(aa),np.sign(bb)); s[s==0]=1.0
    am=np.where(aa!=0,np.abs(aa),np.abs(bb)); bm=np.where(bb!=0,np.abs(bb),np.abs(aa))
    gm=np.sqrt(am*bm); E=len(ii)
    B=np.zeros((E,len(J))); B[np.arange(E),jj]=1; B[np.arange(E),ii]=-1
    u,*_=np.linalg.lstsq(B,np.log(am/bm),rcond=None); fit=B@u
    A=np.zeros_like(J); np.fill_diagonal(A,np.diag(J))
    A[ii,jj]=s*gm*np.exp(fit/2); A[jj,ii]=s*gm*np.exp(-fit/2); return A
print(f"{'alpha':>7}{'slowest rate (null)':>21}{'settling time':>15}"
      f"{'chi_off/chi_diag':>18}{'precision':>11}{'r_cycle of J':>14}")
for a in (1.06, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0):
    Js = scaled(V, a); A0 = full_symmetrizable(Js)
    ev = np.linalg.eigvals(np.diag(xstar) @ A0)
    lam = float(np.max(ev.real))
    if lam >= 0:
        print(f"{a:>7.2f}   unstable"); continue
    chi = -np.linalg.inv(A0); off = ~np.eye(n, dtype=bool)
    rel = float(np.median(np.abs(chi[off]))/np.median(np.abs(np.diag(chi))))
    _,_,rc,_ = invariants(Js)
    print(f"{a:>7.2f}{lam:>21.4f}{3/abs(lam):>15.0f}{rel:>18.4f}{100*rel/2:>10.2f}%{rc:>14.4f}")
print("\nsettling time is 3 / |slowest rate|, in units of one generation of the slowest mode.")
tr = transitions(comms); r0, J0, d0 = fit_constrained_fast_d(tr)
x = comms[0][1][0].copy(); pres = comms[0][1].mean(0) > 1e-3; x[~pres]=0.0
for _ in range(400):
    x = np.clip(np.exp(np.log(np.maximum(x,1e-12)) + r0 + J0@x), 0, 8.0); x[~pres]=0.0
sv = np.where(x>1e-2)[0]; Jsub=J0[np.ix_(sv,sv)]
ev = np.linalg.eigvals(np.diag(x[sv])@Jsub); chi=-np.linalg.inv(Jsub)
off=~np.eye(len(sv),dtype=bool)
rel=float(np.median(np.abs(chi[off]))/np.median(np.abs(np.diag(chi))))
print(f"project ridge fit: slowest rate {np.max(ev.real):.4f}, settling time "
      f"{3/abs(np.max(ev.real)):.0f}, chi_off/chi_diag {rel:.4f}, precision {100*rel/2:.2f}%")
