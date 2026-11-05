r"""
Is the published cycle inconsistency an averaging artefact?
==========================================================
The matrix the manuscript analyses is exactly the mean of Venturelli's three non-trivial
fitted parameter sets, T2, T3 and T4 (T1 has no interactions at all and is excluded; the mean
of T2..T4 reproduces the published file to 4 decimals). The three fits have different support:
89, 100 and 101 nonzero off-diagonal entries out of 132, and their mean has 109.

That creates a mechanism that inflates r_cycle. If J_ij is nonzero in all three fits while
J_ji is nonzero in only one, the mean divides the second by three and leaves the first intact,
so log|J_ij / J_ji| gains log 3 = 1.10 from the averaging alone, which is of the same order as
the whole reported statistic, 2.169. The reported value is in fact LARGER than that of any
individual fit: 1.12 for T2, 1.55 for T3, 0.58 for T4.

The test is to restrict the statistic to the pairs that are present in both directions in
every fit, where no support mismatch can act, and see what survives.
"""
import os, sys, re, argparse, itertools, json
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--repo", default=".")
ap.add_argument("--out", default="figures")
ARGS = ap.parse_args()
sys.path.insert(0, os.path.join(ARGS.repo, "scripts")); os.chdir(ARGS.repo)
from symmetrizability_invariants import invariants
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")

SP = ["BH","CA","BU","PC","BO","BV","BT","EL","FP","CH","DP","ER"]; S = 12
IDX = {s: k for k, s in enumerate(SP)}
def parse(p):
    t = open(p).read(); M = np.zeros((S, S))
    for m in re.finditer(r'<parameter[^>]*id="a([A-Z]{2})([A-Z]{2})"[^>]*value="([^"]+)"', t):
        a, b, v = m.group(1), m.group(2), float(m.group(3))
        if a in IDX and b in IDX: M[IDX[a], IDX[b]] = v
    return M
T = [parse(f"data/venturelli2018/twelveSpecies{t}.xml") for t in ("T2","T3","T4")]
V = np.mean(T, axis=0)
iu = np.triu_indices(S, 1)

def rcycle_on(edges, Aij, Aji):
    """r_cycle restricted to a given edge set (boolean over pairs)."""
    ii, jj = iu[0][edges], iu[1][edges]; E = int(edges.sum())
    if E < 3: return np.nan, E, 0
    b = np.log(np.abs(Aij[edges] / Aji[edges]))
    B = np.zeros((E, S)); B[np.arange(E), jj] = 1; B[np.arange(E), ii] = -1
    u, *_ = np.linalg.lstsq(B, b, rcond=None)
    return float(np.sqrt(np.mean((b - B @ u) ** 2))), E, E - np.linalg.matrix_rank(B)

aij_V, aji_V = V[iu[0], iu[1]], V[iu[1], iu[0]]
nz = np.array([[(M[iu[0], iu[1]] != 0), (M[iu[1], iu[0]] != 0)] for M in T])  # (3,2,npair)
both_all = nz[:, 0, :].all(0) & nz[:, 1, :].all(0)
conc_V = (aij_V != 0) & (aji_V != 0) & (np.sign(aij_V) == np.sign(aji_V))

print("the averaging mechanism, measured on the concordant pairs of the published matrix")
cnt_ij = nz[:, 0, :].sum(0); cnt_ji = nz[:, 1, :].sum(0)
mism = conc_V & (cnt_ij != cnt_ji)
print(f"  concordant pairs in the published matrix              : {int(conc_V.sum())}")
print(f"  of those, the two directions appear in a DIFFERENT     ")
print(f"  number of the three fits (support mismatch)           : {int(mism.sum())}")
print(f"  log-ratio contributed by the mismatch alone, median    : "
      f"{np.median(np.abs(np.log(cnt_ij[mism]/cnt_ji[mism]))):.3f}")
print(f"  observed |log-ratio| on those pairs, median            : "
      f"{np.median(np.abs(np.log(np.abs(aij_V[mism]/aji_V[mism])))):.3f}")

print("\nr_cycle, computed on different edge sets")
rc_all, E_all, d_all = rcycle_on(conc_V, aij_V, aji_V)
print(f"  published matrix, all its concordant pairs   : r_cycle {rc_all:.4f}  "
      f"edges {E_all}  cycles {d_all}")
sub = conc_V & both_all & (cnt_ij == cnt_ji)
rc_s, E_s, d_s = rcycle_on(sub, aij_V, aji_V)
print(f"  restricted to pairs present in both directions")
print(f"  in all three fits, with matched support      : r_cycle {rc_s:.4f}  "
      f"edges {E_s}  cycles {d_s}")

print("\nthe same restricted statistic computed on each individual fit")
for k, M in enumerate(T):
    a, b = M[iu[0], iu[1]], M[iu[1], iu[0]]
    rc, E, d = rcycle_on(sub, a, b)
    print(f"  T{k+2}: r_cycle on the shared edge set {rc:.4f}  (edges {E}, cycles {d})")

print("\nand the unrestricted statistic per fit, for reference")
for k, M in enumerate(T):
    f1, fs, rc, E = invariants(M)
    print(f"  T{k+2}: f_oneway {f1:.3f}  f_sign {fs:.3f}  r_cycle {rc:.4f}  concordant {E}")

print("\n==== replicated cycle witnesses on the shared edge set ====")
ii, jj = iu[0][sub], iu[1][sub]; E = int(sub.sum())
if E >= 3:
    bd = np.array([np.log(np.abs(M[iu[0], iu[1]][sub] / M[iu[1], iu[0]][sub])) for M in T])
    B = np.zeros((E, S)); B[np.arange(E), jj] = 1; B[np.arange(E), ii] = -1
    import collections
    adj = collections.defaultdict(list)
    for e in range(E):
        adj[ii[e]].append((jj[e], e)); adj[jj[e]].append((ii[e], e))
    parent, seen, tree = {}, set(), set()
    for s in range(S):
        if s in seen or s not in adj: continue
        parent[s] = (None, None); seen.add(s); st = [s]
        while st:
            u = st.pop()
            for v, e in adj[u]:
                if v not in seen:
                    seen.add(v); parent[v] = (u, e); tree.add(e); st.append(v)
    def rp(v):
        o = []
        while parent.get(v, (None, None))[0] is not None:
            u, e = parent[v]; o.append((v, u, e)); v = u
        return o
    out = []
    for e in [x for x in range(E) if x not in tree]:
        a, b = ii[e], jj[e]
        ra = [a] + [u for _, u, _ in rp(a)]; rb = [b] + [u for _, u, _ in rp(b)]
        common = next((x for x in ra if x in rb), None)
        if common is None: continue
        edges, signs, nodes = [e], [+1], {SP[a], SP[b]}
        v = a
        while v != common:
            u, ee = parent[v]; edges.append(ee); signs.append(-1 if ii[ee] == v else +1)
            nodes.add(SP[u]); v = u
        v = b
        while v != common:
            u, ee = parent[v]; edges.append(ee); signs.append(+1 if ii[ee] == v else -1)
            nodes.add(SP[u]); v = u
        rr = bd[:, edges] @ np.array(signs, float)
        out.append((sorted(nodes), rr))
    agree = [(n, r) for n, r in out if np.all(r > 0) or np.all(r < 0)]
    print(f"  fundamental cycles on the shared edge set: {len(out)}")
    print(f"  with the same sign of the residual in all three fits: {len(agree)}")
    for n, r in sorted(agree, key=lambda t: -abs(np.mean(t[1])))[:12]:
        print(f"    {'-'.join(n):<22} per fit {np.array2string(r, precision=2, floatmode='fixed')}"
              f"  mean {np.mean(r):+.2f}")
    json.dump(dict(rc_published_all=float(rc_all), edges_all=int(E_all),
                   rc_shared=float(rc_s), edges_shared=int(E_s), cycles_shared=int(d_s),
                   cycles_tested=int(len(out)), cycles_sign_agree=int(len(agree))),
              open(os.path.join(ARGS.out, "averaging_audit.json"), "w"), indent=1)
    print("\nsaved averaging_audit.json")
