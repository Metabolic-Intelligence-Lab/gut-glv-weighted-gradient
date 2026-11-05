r"""
Design sweep for the decisive experiment of BIB-26-2181.
========================================================
The positive control showed that only ONE instrument of the protocol has useful power:
the held-out predictive comparison between the symmetrizable fit and the free fit.
This script keeps that instrument alone (so each replicate is cheap) and sweeps the three
knobs an experimenter actually controls:

    pairs   0 / 1    add 12 monocultures and 66 pairwise cultures (1:1 and 95:5)
    pulses  0 / 2    add a knockdown of 4 species and a 1:20 dilution per community,
                     6 relaxation points after each
    reps    1/2/4    independent biological replicates of EVERY culture in the design

Ground truths and noise settings are the ones of the specificity ensemble, so the numbers
are comparable with the manuscript. Size is calibrated on the J0 replicates of the same cell.
Output: design_sweep<tag>.json plus a printed table of calibrated power.

Run:  OMP_NUM_THREADS=1 python3 design_sweep.py --repo <repo> --workers 16
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, argparse, json, itertools, collections
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=".")
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--out", default="figures")
ap.add_argument("--tag", default="")
ap.add_argument("--rep-base", type=int, default=40)
ap.add_argument("--rep-dep", type=int, default=24)
ap.add_argument("--grid", default=None, help='override, e.g. "0,0,1;1,2,4"  (pairs,pulses,reps)')
ap.add_argument("--settings", default=None, help='e.g. "0.20,0;0.10,0.12"')
ap.add_argument("--time-one", default=None, help='time a single replicate: "pairs,pulses,reps"')
ap.add_argument("--analyse", default=None, help="path to an existing json")
ARGS = ap.parse_args()
sys.path.insert(0, os.path.join(ARGS.repo, "scripts"))
os.chdir(ARGS.repo)

from constrained_vs_free_glv import (load_communities, transitions, fit_free, pred_mse,  # noqa
                                     S, EPS)
from descent_specificity import fit_constrained_fast_d                          # noqa
from symmetrizability_invariants import invariants as inv3                      # noqa
from detection_power import perm_p                                              # noqa
import descent_specificity_ensemble as ENS                                      # noqa
from power_analysis import antisym                                              # noqa
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")

CAP = 8.0
ALPHA = 0.05
SETTINGS = [(0.20, 0.00), (0.10, 0.12)]
if ARGS.settings:
    SETTINGS = [tuple(float(v) for v in s.split(",")) for s in ARGS.settings.split(";")]
GRID = [(p, u, r) for p in (0, 1) for u in (0, 2) for r in (1, 2, 4)]
if ARGS.grid:
    GRID = [tuple(int(v) for v in g.split(",")) for g in ARGS.grid.split(";")]

# --------------------------------------------------------------------------- truths
# five departures at the levels that matter: the predictively tied free fit (t=1), a weak
# antisymmetric departure, and the three families the positive control found invisible.
WANT = [("base", -1), ("free", 0), ("antisym", 1), ("oneway", 3), ("sign", 3), ("cycle", 3)]

def build_truths():
    comms = load_communities()
    tr = transitions(comms)
    r0, J0, d0 = fit_constrained_fast_d(tr)
    rf, Jf = fit_free(tr)
    geo = ENS.Geometry(comms, r0, J0, d0)
    out = [dict(fam="base", lev=-1, r=r0, J=J0.copy(), dVF=0.0)]
    P = antisym(J0, np.random.default_rng(0))
    unit = geo.d_vf(r0, J0 + P)
    J = J0 + (ENS.TARGETS[1] / unit) * P
    out.append(dict(fam="antisym", lev=1, r=r0, J=J, dVF=geo.d_vf(r0, J)))
    for fam, one, off in (("oneway", ENS._oneway_one, 1000), ("sign", ENS._sign_one, 2000)):
        J, v = ENS.greedy_subset(geo, r0, J0, one, off + 3, ENS.TARGETS[3])
        out.append(dict(fam=fam, lev=3, r=r0, J=J, dVF=v))
    kap, v = ENS.bisect_param(lambda k: geo.d_vf(r0, ENS.gt_cycle(J0, 0, k)),
                              ENS.TARGETS[3], 0.0, 6.0)
    out.append(dict(fam="cycle", lev=3, r=r0, J=ENS.gt_cycle(J0, 0, kap), dVF=v))
    r, J = ENS.gt_free(r0, J0, rf, Jf, 1.0)
    out.append(dict(fam="free", lev=0, r=r, J=J, dVF=geo.d_vf(r, J)))
    out = [t for t in out if (t["fam"], t["lev"]) in WANT]
    for k, t in enumerate(out):
        f1, fs, rc, _ = inv3(t["J"]); t.update(idx=k, f_one=f1, f_sign=fs, r_cycle=rc)
    return comms, out

# --------------------------------------------------------------------------- designs
def make_design(spec, comms):
    """spec = (pairs, pulses, reps). group = fold id (community), -1 = culture, always training.
    Every block is instantiated `reps` times with independent noise."""
    pairs, pulses, reps = spec
    proto = []
    for ci, (cname, X) in enumerate(comms):
        x0 = X[0].copy(); x0[X.mean(0) <= 1e-3] = 0.0
        pul, T = [], 7
        if pulses >= 1:
            pul.append((6, "knockdown", ci)); T = 13
        if pulses >= 2:
            pul.append((12, "dilute", None)); T = 19
        proto.append(dict(group=ci, name=cname, x0=x0, T=T, pulses=pul))
    if pairs:
        for i in range(S):
            x0 = np.zeros(S); x0[i] = 0.05
            proto.append(dict(group=-1, name=f"mono{i}", x0=x0, T=7, pulses=[]))
        for i in range(S):
            for j in range(i + 1, S):
                for frac in (0.5, 0.95):
                    x0 = np.zeros(S); x0[i] = 0.1 * frac; x0[j] = 0.1 * (1 - frac)
                    proto.append(dict(group=-1, name=f"pair{i}_{j}_{frac}", x0=x0, T=7, pulses=[]))
    blocks = []
    for k in range(reps):
        for b in proto:
            c = dict(b); c["name"] = f"{b['name']}_r{k}"; c["rep"] = k
            blocks.append(c)
    return blocks

def simulate_design(blocks, r, J, sp, so, seed):
    rng = np.random.default_rng(seed)
    rngo = np.random.default_rng(seed + 10_000_019)
    data = []
    for b in blocks:
        x = b["x0"].copy(); present = x > 0
        segs, cur = [], [x.copy()]
        pulse_at = {p[0]: p for p in b["pulses"]}
        for t in range(1, b["T"]):
            if t in pulse_at:
                _, kind, par = pulse_at[t]
                segs.append(np.array(cur))
                if kind == "knockdown":
                    k = np.random.default_rng(500 + par).choice(np.where(present)[0], 4, replace=False)
                    x = x.copy(); x[k] *= 0.05
                else:
                    x = x / 20.0
                cur = [x.copy()]
                continue
            x = np.clip(np.exp(np.log(x + EPS) + r + J @ x + sp * rng.standard_normal(S)), 0, CAP)
            x[~present] = 0.0
            cur.append(x.copy())
        segs.append(np.array(cur))
        if so > 0:
            segs = [Y * np.exp(so * rngo.standard_normal(Y.shape)) * (Y > 0) for Y in segs]
        data.append((b["group"], b["name"], segs))
    return data

def all_trans(data, exclude_group=None):
    cs = [(f"{n}_{k}", seg) for g, n, segs in data if g != exclude_group
          for k, seg in enumerate(segs) if len(seg) >= 2]
    return transitions(cs)

# --------------------------------------------------------------------------- instrument
def predictive(data):
    """Leave-one-community-out (all its replicates together). Returns the per-fold advantage
    of the symmetrizable fit over the free fit on held-out one-step prediction, and the
    sign-flip permutation p of its mean."""
    groups = sorted({g for g, _, _ in data if g >= 0})
    adv = []
    for h in groups:
        train = all_trans(data, exclude_group=h)
        r, J, d = fit_constrained_fast_d(train)
        rf, Jf = fit_free(train)
        te = all_trans([x for x in data if x[0] == h])
        adv.append(pred_mse(r, J, te) - pred_mse(rf, Jf, te))
    adv = np.asarray(adv)
    return float(adv.mean()), float(perm_p(adv)), len(all_trans(data))

# --------------------------------------------------------------------------- workers
_CTX = None
def _init(ctx):
    global _CTX; _CTX = ctx

def _job(args):
    gi, si, ti, rep = args
    comms, truths, blocks_by = _CTX
    sp, so = SETTINGS[si]
    t = truths[ti]
    spec = GRID[gi]
    seed = 100_000 * gi + 10_000 * si + 100 * ti + rep
    t0 = time.time()
    data = simulate_design(blocks_by[spec], t["r"], t["J"], sp, so, seed)
    a, p, nt = predictive(data)
    return dict(pairs=spec[0], pulses=spec[1], reps=spec[2], gi=gi, setting=si, truth=ti,
                fam=t["fam"], lev=t["lev"], dVF=t["dVF"], rep=rep, adv=a, p_pred=p,
                ntrans=nt, secs=time.time() - t0)

def main():
    comms, truths = build_truths()
    print(f"{len(truths)} truths", flush=True)
    for t in truths:
        print(f"  {t['idx']} {t['fam']:<8} lev {t['lev']:>2} dVF {t['dVF']:.3f} "
              f"f_sign {t['f_sign']:.2f} r_cycle {t['r_cycle']:.2f}", flush=True)
    blocks_by = {g: make_design(g, comms) for g in GRID}
    for g, b in blocks_by.items():
        print(f"design pairs={g[0]} pulses={g[1]} reps={g[2]}: {len(b)} cultures, "
              f"{sum(x['T'] for x in b)} measurements", flush=True)
    if ARGS.time_one:
        spec = tuple(int(v) for v in ARGS.time_one.split(","))
        _init((comms, truths, {spec: make_design(spec, comms)}))
        g0 = GRID.index(spec) if spec in GRID else 0
        globals()["GRID"] = [spec]
        print(_job((0, 0, 0, 0)), flush=True)
        return
    jobs = []
    for gi in range(len(GRID)):
        for si in range(len(SETTINGS)):
            for t in truths:
                R = ARGS.rep_base if t["fam"] == "base" else ARGS.rep_dep
                for rep in range(R):
                    jobs.append((gi, si, t["idx"], rep))
    print(f"{len(jobs)} replicates, {ARGS.workers} workers", flush=True)
    from multiprocessing import Pool
    t0 = time.time(); results = []
    with Pool(ARGS.workers, initializer=_init, initargs=((comms, truths, blocks_by),)) as pool:
        for k, res in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            results.append(res)
            if (k + 1) % 50 == 0 or k + 1 == len(jobs):
                print(f"  {k+1}/{len(jobs)} done, {time.time()-t0:.0f} s, last {res['secs']:.1f} s/rep "
                      f"(pairs{res['pairs']} pulses{res['pulses']} reps{res['reps']} s{res['setting']} "
                      f"{res['fam']})", flush=True)
    out = os.path.join(ARGS.out, f"design_sweep{ARGS.tag}.json")
    with open(out, "w") as f:
        json.dump(dict(results=results,
                       truths=[{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                for k, v in t.items()} for t in truths]), f)
    print("saved", out, flush=True)
    analyse(results)

def analyse(results):
    print("\n==== calibrated power of the predictive comparison (alpha 0.05, critical value "
          "from the J0 replicates of the same cell) ====")
    fams = []
    for r in results:
        if (r["fam"], r["lev"]) not in fams:
            fams.append((r["fam"], r["lev"]))
    for si in sorted({r["setting"] for r in results}):
        sp, so = SETTINGS[si]
        print(f"\n-- sigma_p={sp:.2f} sigma_o={so:.2f}")
        hdr = "  " + "pairs pulses reps".ljust(20) + "meas".rjust(6) + "  " + "".join(
            f"{f}{'' if l < 0 else l}".rjust(12) for f, l in fams if f != "base")
        print(hdr)
        for gi, spec in enumerate(GRID):
            cell = [r for r in results if r["gi"] == gi and r["setting"] == si]
            if not cell: continue
            base = [r for r in cell if r["fam"] == "base"]
            pv = np.array([r["p_pred"] for r in base])
            pstar = float(np.quantile(pv, ALPHA)) if len(pv) else np.nan
            row = f"  {spec[0]:<6}{spec[1]:<7}{spec[2]:<7}" + f"{np.mean([r['ntrans'] for r in cell]):>6.0f}  "
            for f, l in fams:
                if f == "base": continue
                rs = [r for r in cell if r["fam"] == f and r["lev"] == l]
                row += f"{np.mean([r['p_pred'] <= pstar for r in rs]):>12.2f}" if rs else f"{'-':>12}"
            print(row + f"   [p*={pstar:.4f}, n_base={len(base)}]")

if __name__ == "__main__":
    if ARGS.analyse:
        with open(ARGS.analyse) as f:
            analyse(json.load(f)["results"])
    else:
        main()
