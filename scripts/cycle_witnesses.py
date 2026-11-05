r"""
The cycle obstructions of Table 4, and their robustness to the compositional gauge
=================================================================================
Generates every number in Table 4 of manuscript_BIB_v2.tex.

1. Shared support. Keep the species pairs that are nonzero in both directions in each of
   Venturelli's three fitted parameter sets T2, T3, T4 (T1 has no interactions) and
   sign-concordant in the published matrix, the edge set of averaging_audit.py
   (14 pairs, 4 independent cycles). On it no support mismatch between fits can act.
2. Cycle basis. A spanning forest of the shared graph gives one fundamental cycle per
   non-tree edge. Each cycle is written as an ordered node list, so the residual
   sum_{(a,b) around the cycle} log|J_ab / J_ba| is unambiguous; it is zero for every
   diagonally symmetrizable matrix. Each cycle is oriented so that its mean residual is
   negative, which fixes the sign convention of the table.
3. Gauge. If the data used for a fit sum to a constant, the rows of J are defined only up
   to an additive shift c_i. The residual of each cycle is recomputed on the published
   matrix (the mean of T2..T4) under random row shifts whose root-mean-square size is a
   given fraction of the median |off-diagonal| coefficient, 4000 draws per size. A draw in
   which an edge of the cycle loses sign concordance is counted as "cycle broken".

Runs from the repository root with no arguments; writes figures/cycle_witnesses.json.
"""
import os, re, json, itertools, argparse
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--repo", default=".")
ap.add_argument("--out", default="figures"); ap.add_argument("--draws", type=int, default=4000)
ARGS = ap.parse_args(); os.chdir(ARGS.repo)

SP = ["BH","CA","BU","PC","BO","BV","BT","EL","FP","CH","DP","ER"]; S = 12
IDX = {s: k for k, s in enumerate(SP)}

def parse(p):
    t = open(p).read(); M = np.zeros((S, S))
    for m in re.finditer(r'<parameter[^>]*id="a([A-Z]{2})([A-Z]{2})"[^>]*value="([^"]+)"', t):
        a, b, v = m.group(1), m.group(2), float(m.group(3))
        if a in IDX and b in IDX: M[IDX[a], IDX[b]] = v
    return M

T = {t: parse(f"data/venturelli2018/twelveSpecies{t}.xml") for t in ("T2", "T3", "T4")}
V = np.mean(list(T.values()), axis=0)
Vpub = np.loadtxt("data/venturelli2018/interaction_matrix_averaged.csv", delimiter=",", skiprows=1)
assert np.abs(V - Vpub).max() < 1e-4, "published matrix is not the mean of T2..T4"

# 1. shared, sign-concordant support
# same edge set as averaging_audit.py: nonzero in both directions in every fit and
# sign-concordant in the published (mean) matrix
edges = [(i, j) for i, j in itertools.combinations(range(S), 2)
         if all(M[i, j] != 0 and M[j, i] != 0 for M in T.values())
         and np.sign(V[i, j]) == np.sign(V[j, i])]
print(f"shared pairs: {len(edges)}")

# 2. fundamental cycles from a spanning forest (BFS)
adj = {k: [] for k in range(S)}
for i, j in edges: adj[i].append(j); adj[j].append(i)
parent, tree = {}, set()
for root in range(S):
    if root in parent or not adj[root]: continue
    parent[root] = None; queue = [root]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in parent:
                parent[v] = u; tree.add(frozenset((u, v))); queue.append(v)

def path_to_root(v):
    out = [v]
    while parent[v] is not None: v = parent[v]; out.append(v)
    return out

cycles = []
for i, j in edges:
    if frozenset((i, j)) in tree: continue
    pi, pj = path_to_root(i), path_to_root(j)
    common = next(x for x in pi if x in pj)
    nodes = pi[:pi.index(common) + 1] + pj[:pj.index(common)][::-1]   # i .. common .. j
    cycles.append(nodes)
print(f"independent cycles on the shared support: {len(cycles)}")

def resid(M, cyc):
    tot = 0.0
    for a, b in zip(cyc, cyc[1:] + cyc[:1]):
        u, v = M[a, b], M[b, a]
        if u * v <= 0: return np.nan          # an edge of the cycle is not sign-concordant
        tot += np.log(abs(u / v))
    return tot

off = ~np.eye(S, dtype=bool)
scale = float(np.median(np.abs(V[off][V[off] != 0])))
def shift(M, c):
    A = M.copy(); A[off] = (M - c[:, None])[off]; return A

rows = []
for cyc in cycles:
    per = np.array([resid(M, cyc) for M in T.values()])
    if np.nanmean(per) > 0: cyc = cyc[::-1]; per = -per          # orient: mean negative
    name = "-".join(SP[k] for k in cyc)
    mags = [min(abs(V[a, b]), abs(V[b, a])) for a, b in zip(cyc, cyc[1:] + cyc[:1])]
    weakest = min(range(len(cyc)), key=lambda k: mags[k])
    wa, wb = cyc[weakest], cyc[(weakest + 1) % len(cyc)]
    gauge = {}
    for bound in (0.10, 0.25):
        rng = np.random.default_rng(11); vs = []; broken = 0
        for _ in range(ARGS.draws):
            c = rng.normal(0, 1, S); c = c / np.linalg.norm(c) * bound * scale * np.sqrt(S)
            r_ = resid(shift(V, c), cyc)
            if np.isfinite(r_): vs.append(r_)
            else: broken += 1
        vs = np.array(vs)
        gauge[f"{bound:.2f}"] = dict(
            lo=float(vs.min()), hi=float(vs.max()),
            sign_kept=float(np.mean(vs < 0)) if len(vs) else float("nan"),
            broken=broken / ARGS.draws)
    rows.append(dict(cycle=name, T2=float(per[0]), T3=float(per[1]), T4=float(per[2]),
                     mean_of_fits=float(per.mean()), published=float(resid(V, cyc)),
                     same_sign_all_fits=bool(np.all(per < 0)),
                     min_coefficient=float(min(mags)),
                     weakest_edge=f"{SP[wa]}-{SP[wb]}", gauge=gauge))

rows.sort(key=lambda r: -r["min_coefficient"])     # best-determined cycles first
print(f"\nmedian |off-diagonal| of the published matrix: {scale:.4f}\n")
for r in rows:
    g1, g2 = r["gauge"]["0.10"], r["gauge"]["0.25"]
    print(f"{r['cycle']:<16} T2 {r['T2']:+.3f}  T3 {r['T3']:+.3f}  T4 {r['T4']:+.3f}  "
          f"mean {r['mean_of_fits']:+.3f}  published {r['published']:+.3f}  "
          f"same sign {r['same_sign_all_fits']}")
    print(f"{'':<16} smallest coefficient {r['min_coefficient']:.3f} (edge {r['weakest_edge']})")
    for k, g in (("0.10", g1), ("0.25", g2)):
        print(f"{'':<16} shifts {k}x scale: range [{g['lo']:+.3f}, {g['hi']:+.3f}]  "
              f"sign kept {g['sign_kept']:.3f} of valid draws  cycle broken in {g['broken']:.3f}")

os.makedirs(ARGS.out, exist_ok=True)
json.dump(dict(shared_edges=len(edges), cycles=len(cycles), scale=scale, rows=rows),
          open(os.path.join(ARGS.out, "cycle_witnesses.json"), "w"), indent=1)
print("\nsaved cycle_witnesses.json")
