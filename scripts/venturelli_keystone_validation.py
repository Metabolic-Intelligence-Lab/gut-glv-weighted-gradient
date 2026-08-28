"""
Ground-truth keystone validation from Venturelli 2018 leave-one-out data (A5)
=============================================================================

The paper's keystones are in-silico single-species knockouts of the inferred gLV
(keystones_from_data.py, human_keystones.py). The main text is explicit that
these are hypotheses, "not externally validated". Venturelli et al. give the
external check: their Dataset EV3 contains the full 12-member community (NONE)
and all 12 single-species leave-one-out communities, i.e. real knockout
experiments with a measured outcome.

We compare, per species, the community-wide impact of removing it:
  - EMPIRICAL  impact_i = Bray-Curtis( full community , community with i removed )
                          on the shared 11 species (from Dataset EV3).
  - MODEL      impact_i = the same, but the "community with i removed" is obtained
                          by knocking i out of the MEASURED gLV (SBML, T4) and
                          re-equilibrating (paper's gLV with immigration floor
                          eps=1e-3, so competitive exclusion does not spuriously
                          drop species; the full model then coexists at 12).

Confound: removing an abundant species forces a large renormalisation of the
rest, inflating impact regardless of interactions. We therefore also report the
Spearman PARTIAL correlation controlling for the removed species' abundance.

Finding: model and ground-truth impact rankings agree at Spearman rho ~ 0.58
(p ~ 0.05), and the agreement SURVIVES the abundance control (partial rho ~ 0.51),
so the knockout captures interaction-driven keystone-ness, not mere abundance.
The top ground-truth keystone, Eggerthella lenta (EL), sits at ~2% abundance and
a 0.1%-abundance species (DP) is flagged high-impact by the model: network-pivotal
rather than abundant, as the keystone concept predicts. The moderate correlation
(one synthetic community, n=12) matches the method's own ~0.6 internal rank
stability and supports treating keystone calls as ranked hypotheses.

Requires scipy, openpyxl.  Run:  python3 venturelli_keystone_validation.py
"""

import re
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr, rankdata

NS = {"s": "http://www.sbml.org/sbml/level3/version1/core"}
SP = ["BH", "CA", "BU", "PC", "BO", "BV", "BT", "EL", "FP", "CH", "DP", "ER"]
IDX = {s: k for k, s in enumerate(SP)}
S = len(SP)
DATA = "data/venturelli2018"
OUT = "figures"
EPS = 1e-3                                   # immigration floor, as in the paper's gLV


def load_glv(fn):
    P = {p.get("id"): float(p.get("value"))
         for p in ET.parse(fn).getroot().findall(".//s:parameter", NS)}
    A = np.zeros((S, S)); u = np.zeros(S)
    for a in SP:
        u[IDX[a]] = P["u" + a]
        for b in SP:
            A[IDX[a], IDX[b]] = P["a" + a + b]
    return u, A


def load_leaveoneout():
    import openpyxl
    ws = openpyxl.load_workbook(f"{DATA}/timeseries_multispecies_EV3.xlsx",
                               read_only=True, data_only=True)["Dataset EV3"]
    comms, cur = {}, None
    for r in ws.iter_rows(values_only=True):
        if not r or r[0] is None:
            continue
        s = str(r[0]).strip().strip("'").strip()
        if re.search(r"[A-Za-z]", s):
            cur = s; comms[cur] = []
        else:
            v = [float(x) for x in s.split()]
            if len(v) == S:
                comms[cur].append(v)
    return {k: (np.array(v)[-2:].mean(0) / np.array(v)[-2:].mean(0).sum())
            for k, v in comms.items()}       # steady composition = mean of last 2 pts


def steady(u, A, mask, tmax=800):
    idx = np.where(mask)[0]
    def rhs(t, x):
        xi = np.clip(x, 0, None); xi[~mask] = 0
        dx = np.zeros(S); dx[idx] = xi[idx] * (u[idx] + A[idx] @ xi) + EPS
        return dx
    sol = solve_ivp(rhs, [0, tmax], np.where(mask, 0.1, 0.0),
                    method="LSODA", rtol=1e-8, atol=1e-10)
    xf = np.clip(sol.y[:, -1], 0, None); xf[~mask] = 0
    return xf


def bc(p, q):
    p = p / p.sum(); q = q / q.sum()
    return float(np.abs(p - q).sum() / 2)


def partial_spearman(x, y, z):
    """Spearman(x, y | z): rank, residualise on rank(z), correlate residuals."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.c_[np.ones(len(rz)), rz]
    res = lambda a: a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def main():
    u, A = load_glv(f"{DATA}/twelveSpeciesT4.xml")
    comm = load_leaveoneout()
    full_emp = comm["NONE"]
    xfull = steady(u, A, np.ones(S, bool))

    imp_mod = np.zeros(S); imp_emp = np.zeros(S)
    for i in range(S):
        keep = [j for j in range(S) if j != i]
        m = np.ones(S, bool); m[i] = False
        imp_mod[i] = bc(xfull[keep], steady(u, A, m)[keep])
        imp_emp[i] = bc(full_emp[keep], comm[f"{SP[i]}-"][keep])

    rho, p = spearmanr(imp_mod, imp_emp)
    pr = partial_spearman(imp_mod, imp_emp, full_emp)
    print(f"Spearman(model, empirical)            = {rho:.3f}  (p={p:.3f})")
    print(f"partial Spearman(model, emp | abund)  = {pr:.3f}")
    print(f"\n{'sp':>4} {'abund%':>7} {'model':>7} {'empir':>7}")
    for i in np.argsort(-imp_emp):
        print(f"{SP[i]:>4} {100*full_emp[i]:>7.2f} {imp_mod[i]:>7.3f} {imp_emp[i]:>7.3f}")

    # ---- figure: model vs ground-truth impact, point size = abundance ----
    from matplotlib.lines import Line2D
    INK, MUTED = "#1a1a1a", "#5a5a5a"
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": MUTED, "axes.linewidth": 0.9,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    sz = 70 + 1100 * full_emp
    lowab = full_emp < 0.03           # low-abundance keystones: the interaction-driven story
    ax.scatter(imp_mod[~lowab], imp_emp[~lowab], s=sz[~lowab], c="#9aa6b2", alpha=.8,
               edgecolor="w", linewidth=1.0, zorder=3)
    ax.scatter(imp_mod[lowab], imp_emp[lowab], s=sz[lowab], c="#0072B2", alpha=.9,
               edgecolor="w", linewidth=1.0, zorder=4)
    for i in range(S):
        ax.annotate(SP[i], (imp_mod[i], imp_emp[i]), fontsize=9,
                    fontweight=("bold" if lowab[i] else "normal"),
                    color=("#0072B2" if lowab[i] else MUTED),
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("model in-silico knockout impact")
    ax.set_ylabel("measured leave-one-out impact")
    ax.set_title("Keystone ranking agrees with real knockouts\n"
                 rf"Spearman $\rho={rho:.2f}$ ($p={p:.2f}$); abundance-controlled $\rho={pr:.2f}$",
                 fontsize=12, color=INK)
    ax.text(0.60, 0.90, "point size $\\propto$ abundance", transform=ax.transAxes,
            fontsize=9, va="top", ha="left", color=MUTED)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2",
                              markersize=10, label="low-abundance keystone ($<$3%)"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor="#9aa6b2",
                              markersize=10, label="typical species")],
              loc="lower right", frameon=False, fontsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = f"{OUT}/fig24_keystone_validation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
