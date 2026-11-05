"""
Size calibration of the predictive comparison (BIB revision, Reviewer 4 point 3)
=================================================================================

detection_power.py tests "the free and the diagonally symmetrizable gLV predict equally well"
with an exact sign-flip permutation test on the 13 leave-one-community-out advantages, and
reports its rejection rate at alpha = 0.05 as power. At delta = 0 the ground truth IS
symmetrizable, so the rejection rate there is the test's SIZE, and it was 0.125, not 0.05: the
13 advantages share 11 of 12 training communities each, so they are not independent and the
sign-flip null is too narrow.

This script reruns the published protocol with the published seeds (and asserts that the
published rates are reproduced), draws an INDEPENDENT calibration set at delta = 0, sets the
critical value so that the calibrated size is 0.05 there, and recomputes the power curve at
that critical value. Two calibrations are reported, on the permutation p value (keeps the
pre-specified statistic, moves its threshold) and on the mean advantage itself.

Run:  OMP_NUM_THREADS=1 python3 scripts/detection_power_calibrated.py [--workers 2]
      python3 scripts/detection_power_calibrated.py --replot
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys                                                                  # noqa: E402
import time                                                                 # noqa: E402
from concurrent.futures import ProcessPoolExecutor                         # noqa: E402
from pathlib import Path                                                    # noqa: E402

import numpy as np                                                          # noqa: E402
import matplotlib                                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE.parent)

from constrained_vs_free_glv import load_communities, transitions, S        # noqa: E402
from inference_aware_null import resid_std                                 # noqa: E402
from power_analysis import antisym, fit_constrained_fast                   # noqa: E402
from detection_power import one_replicate, DELTAS, N_REP, ALPHA            # noqa: E402

import warnings                                                             # noqa: E402
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

OUT = "figures"
NPZ = os.path.join(OUT, "detection_power_calibrated.npz")
FIG = os.path.join(OUT, "fig_detection_power_calibrated.png")
N_CAL = 200
PUBLISHED_POWER = [0.125, 0.0833, 0.10, 0.1583, 0.275]      # figures/power.npz, as published


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def run(workers):
    t0 = time.time()
    comms = load_communities()
    tr = transitions(comms)
    r0, J0 = fit_constrained_fast(tr)
    sproc = resid_std(comms, r0, J0)
    Pdir = antisym(J0, np.random.default_rng(0))
    off = ~np.eye(S, dtype=bool)
    A = np.array([a for a, g, m in tr])
    Mk = np.array([m for a, g, m in tr])
    scale = np.sqrt(np.mean((A @ J0.T)[Mk] ** 2))
    dvf = [float(np.sqrt(np.mean((A @ (dl * Pdir).T)[Mk] ** 2)) / scale) for dl in DELTAS]
    dF = [float(np.linalg.norm(dl * Pdir[off]) / np.linalg.norm(J0[off])) for dl in DELTAS]
    print(f"process noise (published) = {sproc:.3f}", flush=True)

    adv = np.zeros((len(DELTAS), N_REP))
    pv = np.zeros((len(DELTAS), N_REP))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, dl in enumerate(DELTAS):
            Jd = J0 + dl * Pdir
            jobs = [(comms, r0, Jd, sproc, 3000 + int(dl * 10) + i) for i in range(N_REP)]
            res = list(pool.map(one_replicate, jobs, chunksize=2))
            adv[k] = [a for a, p in res]
            pv[k] = [p for a, p in res]
            print(f"  delta={dl}: nominal reject {np.mean(pv[k] <= ALPHA):.4f} "
                  f"(published {PUBLISHED_POWER[k]:.4f})  {time.time()-t0:.0f} s", flush=True)
        jobs = [(comms, r0, J0, sproc, 700_000 + i) for i in range(N_CAL)]
        res = list(pool.map(one_replicate, jobs, chunksize=2))
        adv_cal = np.array([a for a, p in res])
        pv_cal = np.array([p for a, p in res])
    print(f"calibration set done, {time.time()-t0:.0f} s", flush=True)
    np.savez(NPZ, deltas=np.array(DELTAS), dvf=np.array(dvf), dF=np.array(dF), adv=adv, pv=pv,
             adv_cal=adv_cal, pv_cal=pv_cal, sproc=sproc, alpha=ALPHA)
    report()


def report():
    D = np.load(NPZ)
    deltas, dvf, adv, pv = D["deltas"], D["dvf"], D["adv"], D["pv"]
    adv_cal, pv_cal = D["adv_cal"], D["pv_cal"]
    n = pv.shape[1]
    # p-value calibration: the largest threshold whose size on the calibration set is <= alpha
    cand = np.sort(np.unique(pv_cal))
    ok = [c for c in cand if np.mean(pv_cal <= c) <= ALPHA]
    p_star = max(ok) if ok else 0.0
    # mean-advantage calibration
    c_star = float(np.quantile(adv_cal, 1 - ALPHA, method="higher"))
    size_cal_nom = np.mean(pv_cal <= ALPHA)
    print(f"\ncalibration set (delta = 0, n = {len(pv_cal)}): nominal size {size_cal_nom:.3f} "
          f"{tuple(round(x, 3) for x in wilson(int(np.sum(pv_cal <= ALPHA)), len(pv_cal)))}")
    print(f"calibrated critical values: p* = {p_star:.4f}   mean advantage c* = {c_star:+.5f}")
    print(f"\n{'delta':>5} {'dVF':>6} {'nominal':>20} {'calibrated (p*)':>22} "
          f"{'calibrated (adv)':>22}")
    out = []
    for k, dl in enumerate(deltas):
        kn = int(np.sum(pv[k] <= ALPHA))
        kp = int(np.sum(pv[k] <= p_star))
        ka = int(np.sum(adv[k] > c_star))
        rows = [(kn / n, *wilson(kn, n)), (kp / n, *wilson(kp, n)), (ka / n, *wilson(ka, n))]
        out.append(rows)
        print(f"{dl:>5.0f} {dvf[k]:>6.3f} " + " ".join(
            f"{a:>8.3f} [{b:.3f},{c:.3f}]" for a, b, c in rows))
    # new values override any cached copy, so --replot can re-save without duplicate keys
    np.savez(NPZ, **{**{k: D[k] for k in D.files}, "p_star": p_star, "c_star": c_star,
                     "table": np.array(out)})
    make_figure(deltas, dvf, np.array(out), p_star)


def make_figure(deltas, dvf, tab, p_star):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    lab = ["nominal ($\\alpha=0.05$)", f"size-calibrated ($p\\leq{p_star:.3f}$)",
           "size-calibrated (mean advantage)"]
    col = ["#999999", "#0072B2", "#56B4E9"]
    for j in range(3):
        y, lo, hi = tab[:, j, 0], tab[:, j, 1], tab[:, j, 2]
        ax.errorbar(dvf + (j - 1) * 0.004, y, yerr=[y - lo, hi - y], fmt="o-", color=col[j],
                    lw=2, ms=6, capsize=3, label=lab[j])
    ax.axhline(ALPHA, color="black", ls=":", lw=1)
    ax.text(dvf[-1], ALPHA + 0.01, "nominal level", ha="right", va="bottom", fontsize=10)
    ax.set_xlabel("vector-field displacement $\\delta_{\\rm VF}$ of the ground truth", fontsize=12)
    ax.set_ylabel("rejection rate, 95% Wilson interval", fontsize=12)
    ax.tick_params(labelsize=10.5)
    ax.set_ylim(0, 0.6)
    ax.set_title("Predictive comparison: size at $\\delta_{\\rm VF}=0$, power beyond", fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG, dpi=200)
    print(f"[fig] {FIG}")


if __name__ == "__main__":
    if "--replot" in sys.argv:
        report()
    else:
        w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
        run(w)
