r"""
Monte Carlo error of every cell of Table 5, and the source of its descent-test column
=====================================================================================
Table 5 prints eighteen size-0.05 powers to two decimals and names two scripts, but neither
writes a cache: `plateau_specificity.py` only prints, and `design_sweep.py` writes the raw
replicates and not the table. A referee cannot check a single cell, and the sentence that
abstract, Key Points, cover letter and the table's own caption all carry, "no instrument that
reads trajectories exceeds 0.25 at the published design", rests on those cells. This script
recomputes all eighteen from the two caches, attaches an interval to each, and writes the table
as a file.

MANDATORY CONTROL, checked before anything else is reported: every published cell must be
reproduced to the two decimals printed. A cell that does not reproduce means this script has
the wrong statistic, the wrong noise setting or the wrong design cell, and nothing of its own
output is usable. An earlier run of this script declared one exception, the antisymmetric cell of
the proposed design published at 0.70, because the sweep had been regenerated with the script's
own default of 24 dependent replicates. None of the twelve published predictive cells is
reachable with 24 replicates and all twelve are exact with 40, so the published table was run
with 40 and the mismatch was in the rerun, not in the table. Regenerated at 40, every cell
reproduces, and no exception is declared.

Statistic, per cell: the size-0.05 power, with the critical value taken from the symmetrizable
replicates of that same cell, exactly as the two source scripts define it. Descent test: the
margin, fitted minus scrambled, pooled over the series, critical value the 95th percentile on
the symmetrizable truth. Predictive comparison: the permutation p of the held-out advantage,
critical value the 5th percentile on the symmetrizable truth.

Interval: 95 percent, from a bootstrap that resamples every stage that was random in the
original run, so it is the honest width and not the binomial one. The critical value is
re-estimated inside each draw, since it is itself an estimate from a finite number of
symmetrizable replicates. For the descent column the resampling is hierarchical, truths first
and then replicates within each truth, because that column averages over a sample of ground
truths drawn from each family and a referee reading "0.17" is being told about the family, not
about the one matrix that was drawn. Seed fixed at 20261105.

CRITERION, fixed before execution, three branches:
  1. The table stands as printed, if every published cell lies inside its own interval and
     every half-width is at most 0.05, which is what two decimals assert.
  2. The point values stand but the precision does not, if every published cell lies inside
     its interval and some half-width exceeds 0.05. Then the two decimals claim a resolution
     the replicate count cannot deliver, and the fix is to print intervals or to state the
     replicate count in the caption.
  3. A cell is a verified defect, if the published value lies outside its interval. Then it is
     corrected to the measured value.
Each of the three sentences the table supports is tested separately against the upper bounds,
since a bound asserted over several cells fails if any one of them can exceed it.

Run:  python3 scripts/table5_audit.py
Writes: figures/table5_audit.json
"""
import json
import numpy as np

NPZ = "figures/descent_specificity_ensemble.npz"
SWEEP = "figures/design_sweep.json"
OUT = "figures/table5_audit.json"
F_ALL, N_ALL = 2, 3          # pooled fitted and scrambled margin columns of the ensemble
ALPHA, NBOOT, SEED = 0.05, 4000, 20261105
HALFWIDTH_OK = 0.05          # what two printed decimals assert
FAMS = ["base", "free", "antisym", "cycle", "oneway", "sign"]

# Table 5 as printed in manuscript_finale.tex, rows in the order of the table.
PUBLISHED = {
    "base":    dict(descent=0.05, pred_published=0.05, pred_proposed=0.05),
    "free":    dict(descent=0.01, pred_published=0.25, pred_proposed=1.00),
    "antisym": dict(descent=0.08, pred_published=0.10, pred_proposed=0.70),
    "cycle":   dict(descent=0.17, pred_published=0.10, pred_proposed=0.12),
    "oneway":  dict(descent=0.04, pred_published=0.05, pred_proposed=0.10),
    "sign":    dict(descent=0.04, pred_published=0.03, pred_proposed=0.05),
}
KNOWN_MISMATCH = []        # see the docstring: the one that was declared was the rerun's


def descent_column(rng):
    """Published-design descent test: point estimate and hierarchical bootstrap interval."""
    z = np.load(NPZ, allow_pickle=True)
    R, meta, fams, settings = z["rows"], z["meta"], z["fams"], z["settings"]
    name = {i: f for i, f in enumerate(fams)}
    fam = lambda ti: name.get(int(meta[ti][0]), "base") if meta[ti][0] >= 0 else "base"
    si = [i for i, (sp, so) in enumerate(settings)
          if abs(sp - 0.20) < 1e-9 and abs(so) < 1e-9]
    assert len(si) == 1, f"the published noise setting (0.20, 0) is not unique in {NPZ}"
    si = si[0]
    sel = R[:, 0] == si
    # margins per truth, and the symmetrizable truth that sets the critical value
    by_fam, base_margins = {}, None
    for ti in range(len(meta)):
        X = R[sel & (R[:, 1] == ti)]
        if len(X) < 5:
            continue
        m = X[:, F_ALL] - X[:, N_ALL]
        if fam(ti) == "base":
            base_margins = m
        by_fam.setdefault(fam(ti), []).append(m)
    assert base_margins is not None, "no symmetrizable truth in the published setting"
    c = np.quantile(base_margins, 1 - ALPHA)
    point = {f: float(np.mean([np.mean(m > c) for m in ms])) for f, ms in by_fam.items()}
    point["base"] = float(np.mean(base_margins > c))
    draws = {f: [] for f in by_fam}
    for _ in range(NBOOT):
        cb = np.quantile(rng.choice(base_margins, len(base_margins), replace=True), 1 - ALPHA)
        for f, ms in by_fam.items():
            pick = rng.integers(0, len(ms), len(ms))          # truths, with replacement
            draws[f].append(np.mean([np.mean(rng.choice(ms[i], len(ms[i]), replace=True) > cb)
                                     for i in pick]))
    out = {}
    for f in by_fam:
        lo, hi = np.quantile(draws[f], [0.025, 0.975])
        out[f] = dict(power=point[f], lo=float(lo), hi=float(hi),
                      n_truths=len(by_fam[f]), n_rep=int(sum(len(m) for m in by_fam[f])))
    return out, float(c), len(base_margins)


def pred_column(res, gi, si, rng):
    """One cell column of the sweep: power of the predictive comparison with its interval."""
    cell = [r for r in res if r["gi"] == gi and r["setting"] == si]
    base = np.array([r["p_pred"] for r in cell if r["fam"] == "base"])
    assert len(base) > 0, f"no symmetrizable replicates at gi={gi}, setting={si}"
    pstar = float(np.quantile(base, ALPHA))
    out = {}
    for f in FAMS:
        if f == "base":
            pv = base
        else:
            pv = np.array([r["p_pred"] for r in cell if r["fam"] == f])
            if not len(pv):
                continue
        point = float(np.mean(pv <= pstar))
        draws = np.empty(NBOOT)
        for b in range(NBOOT):
            pb = float(np.quantile(rng.choice(base, len(base), replace=True), ALPHA))
            draws[b] = np.mean(rng.choice(pv, len(pv), replace=True) <= pb)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        out[f] = dict(power=point, lo=float(lo), hi=float(hi), n_truths=1, n_rep=int(len(pv)))
    return out, pstar, len(base)


def main():
    rng = np.random.default_rng(SEED)
    desc, c_desc, n_base_desc = descent_column(rng)
    res = json.load(open(SWEEP))["results"]
    grid = {(r["pairs"], r["pulses"], r["reps"]): r["gi"] for r in res}
    gi_pub, gi_prop = grid[(0, 0, 1)], grid[(0, 2, 4)]
    pub, p_pub, n_pub = pred_column(res, gi_pub, 0, rng)
    prop, p_prop, n_prop = pred_column(res, gi_prop, 0, rng)
    cols = dict(descent=desc, pred_published=pub, pred_proposed=prop)

    print("Table 5 recomputed from the two caches, with 95% bootstrap intervals\n")
    print(f"critical values: descent margin {c_desc:+.3f} (n_base={n_base_desc}), "
          f"predictive p* {p_pub:.4f} published / {p_prop:.4f} proposed (n_base={n_pub})\n")
    hdr = f"{'ground truth':<14}" + "".join(f"{k:>34}" for k in cols)
    print(hdr + "\n" + "-" * len(hdr))
    control_fail, wide, outside = [], [], []
    for f in FAMS:
        row = f"{f:<14}"
        for k, col in cols.items():
            if f not in col:
                row += f"{'-':>34}"
                continue
            d, p = col[f], PUBLISHED[f][k]
            half = (d["hi"] - d["lo"]) / 2
            inside = d["lo"] - 1e-12 <= p <= d["hi"] + 1e-12
            # formatted comparison, not a tolerance: 5/40 is exactly 0.125 and prints
            # as 0.12, so any symmetric tolerance of 0.005 falls on the boundary and
            # reports a cell that reproduces exactly as a failure.
            repro = f"{d['power']:.2f}" == f"{p:.2f}"
            flag = "" if repro else ("  [declared]" if (f, k) in KNOWN_MISMATCH else "  REPRO-FAIL")
            if not repro and (f, k) not in KNOWN_MISMATCH:
                control_fail.append((f, k, d["power"], p))
            if half > HALFWIDTH_OK:
                wide.append((f, k, half))
            if not inside:
                outside.append((f, k, d["power"], p, d["lo"], d["hi"]))
            row += f"{d['power']:>9.2f} [{d['lo']:.2f},{d['hi']:.2f}] pub {p:.2f}{flag}"[:34].rjust(34)
        print(row)
    print(f"\nreplicates per cell: descent {desc['cycle']['n_rep']} over "
          f"{desc['cycle']['n_truths']} truths (cycle row), predictive "
          f"{pub['cycle']['n_rep']} over 1 truth")

    print("\n--- mandatory control ---")
    if control_fail:
        print("FAILED, the statistic or the cell is wrong; nothing below is usable:")
        for f, k, got, p in control_fail:
            print(f"   {f} {k}: recomputed {got:.3f}, published {p:.2f}")
        return
    print("passed: every published cell reproduces to two decimals"
          + (f", except the {len(KNOWN_MISMATCH)} declared in advance" if KNOWN_MISMATCH
             else ", with no exception declared"))

    print("\n--- criterion ---")
    if outside:
        print("BRANCH 3, verified defect: a published value lies outside its own interval")
        for f, k, got, p, lo, hi in outside:
            print(f"   {f} {k}: published {p:.2f}, measured {got:.2f} [{lo:.2f},{hi:.2f}]")
    elif wide:
        print("BRANCH 2, the point values stand but two decimals overstate the precision")
        w = max(wide, key=lambda t: t[2])
        print(f"   {len(wide)} of 18 cells have a half-width above {HALFWIDTH_OK:.2f}, "
              f"widest {w[0]} {w[1]} at {w[2]:.2f}")
    else:
        print("BRANCH 1, the table stands as printed")

    print("\n--- the three sentences the table carries ---")
    trj = {**{("descent", f): d for f, d in desc.items()},
           **{("pred_published", f): d for f, d in pub.items()}}
    worst = max(trj.items(), key=lambda kv: kv[1]["hi"])
    print(f"1. 'no instrument that reads trajectories exceeds 0.25 at the published design': "
          f"largest point {max(d['power'] for d in trj.values()):.2f}, "
          f"largest upper bound {worst[1]['hi']:.2f} ({worst[0][1]}, {worst[0][0]}) "
          f"-> {'HOLDS' if worst[1]['hi'] <= 0.25 else 'the bound is not safe at 95%'}")
    dmax = max(desc.items(), key=lambda kv: kv[1]["hi"])
    print(f"2. 'the descent test does not exceed 0.25 at any design': at the one design the "
          f"cache covers, largest point {max(d['power'] for d in desc.values()):.2f}, "
          f"largest upper bound {dmax[1]['hi']:.2f} ({dmax[0]}) "
          f"-> {'HOLDS' if dmax[1]['hi'] <= 0.25 else 'the bound is not safe at 95%'}")
    over, pts = [], []
    for si in sorted({r["setting"] for r in res}):
        for g in sorted({r["gi"] for r in res}):
            col, _, _ = pred_column(res, g, si, rng)
            for f in ("oneway", "cycle"):
                if f in col:
                    pts.append(col[f]["power"])
                    if col[f]["hi"] > 0.29:
                        over.append((f, g, si, col[f]["power"], col[f]["hi"]))
    print(f"3. 'one-way and cyclic departures never exceed 0.29 at any design simulated': "
          f"largest point {max(pts):.2f} over {len(pts)} cells; "
          f"{len(over)} cells have an upper bound above 0.29 "
          f"-> {'HOLDS' if not over else 'the bound is not safe at 95%'}")
    for f, g, si, p, hi in over:
        print(f"     {f} gi={g} setting={si}: {p:.2f} [upper {hi:.2f}]")

    payload = dict(
        generated_by="scripts/table5_audit.py", seed=SEED, nboot=NBOOT, alpha=ALPHA,
        sources=dict(descent=NPZ, predictive=SWEEP),
        noise_setting=dict(sigma_p=0.20, sigma_o=0.00),
        designs=dict(published=[0, 0, 1], proposed=[0, 2, 4]),
        critical_values=dict(descent_margin=c_desc, pstar_published=p_pub,
                             pstar_proposed=p_prop, n_base_descent=n_base_desc,
                             n_base_predictive=n_pub),
        published_table=PUBLISHED, declared_mismatch=[list(k) for k in KNOWN_MISMATCH],
        cells=cols,
        sentence_checks=dict(
            trajectories_025=dict(max_point=max(d["power"] for d in trj.values()),
                                  max_upper=worst[1]["hi"], cell=list(worst[0])),
            descent_025=dict(max_point=max(d["power"] for d in desc.values()),
                             max_upper=dmax[1]["hi"], family=dmax[0]),
            oneway_cyclic_029=dict(max_point=max(pts), n_cells=len(pts),
                                   cells_over=[list(o) for o in over])),
    )
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
