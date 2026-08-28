"""
Hero figure (Fig. 1): where the gut microbiome lands on the reciprocity axis.
=============================================================================

(a) Corrected cavity phase plane: the resilient-phase boundary sigma_c=sqrt2/(1+gamma)
    separates the unique-fixed-point region from the unstable one; the microbiome is
    constrained only in gamma (horizontal band, gamma<=~0.1) -- sigma is not identifiable.
(b) The measured reciprocity gamma across seven cohorts and two experimentally
    constrained matrices, coloured by data type. Absolute-abundance and measured-matrix
    values cluster at gamma<=~0.1, far below the spin-glass value gamma=1 and below the
    competitive-fold threshold gamma~0.3; relative-abundance values are inflated upward.

Palette: Okabe-Ito colourblind-safe categorical set (identity by data type).
Run:  python3 fig_hero.py     (writes figures/fig_hero.png)
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "figures"
# Okabe-Ito colourblind-safe categorical palette (identity = data type)
C_ABS = "#0072B2"   # absolute abundance (qPCR / QMP)
C_MAT = "#009E73"   # experimentally constrained matrix
C_REL = "#E69F00"   # relative abundance (compositional)
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d9d9d9"

plt.rcParams.update({
    "font.size": 12, "axes.edgecolor": MUTED, "axes.linewidth": 0.9,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "font.family": "DejaVu Sans",
})

# measured reciprocity  (name, gamma, type)
DATA = [
    ("Gibson healthy",  0.00, "abs"), ("Gibson colitis", 0.01, "abs"),
    ("Vandeputte QMP",  0.10, "abs"),
    ("Venturelli",      0.06, "mat"), ("Clark",         -0.06, "mat"),
    ("David",           0.31, "rel"), ("DiGiulio",       0.26, "rel"),
    ("Bokulich",        0.15, "rel"), ("DIABIMMUNE",     0.22, "rel"),
    ("Vandeputte rel.", 0.38, "rel"),
]
COL = {"abs": C_ABS, "mat": C_MAT, "rel": C_REL}
ROW = {"abs": 2, "mat": 1, "rel": 0}
ROWLAB = {2: "absolute\n(qPCR/QMP)", 1: "measured\nmatrix", 0: "relative\n(compositional)"}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.7), gridspec_kw={"width_ratios": [1, 1.15]})

# ---- (a) phase plane -------------------------------------------------------
g = np.linspace(-0.25, 1.0, 300)
sc = np.sqrt(2) / (1 + g)
smax = 3.0
axA.fill_betweenx(g, 0, sc, color="#1b9e77", alpha=0.10)          # resilient
axA.fill_betweenx(g, sc, smax, color="#d95f02", alpha=0.10)       # unstable
axA.plot(sc, g, color=INK, lw=2.3)
axA.text(0.40, 0.78, "resilient\n(unique fixed point)", fontsize=10, color="#0a6b4e", ha="center")
axA.text(2.02, 0.76, "unstable\n(multiplicity / chaos)", fontsize=10, color="#a34400", ha="center")
axA.text(1.02, 0.60, r"$\sigma_c=\sqrt{2}/(1+\gamma)$", fontsize=11, color=INK, rotation=-52)
# empirical band: only gamma constrained, sigma spans the axis
axA.axhspan(-0.05, 0.13, color=C_ABS, alpha=0.16)
axA.axhline(1.0, color=MUTED, ls=":", lw=1.1)
axA.text(2.98, 0.965, "spin glass $\\gamma=1$", fontsize=9.5, color=MUTED, va="center", ha="right")
axA.annotate("microbiome (measured):\n$\\gamma\\lesssim0.1$, $\\sigma$ not identified",
             xy=(2.15, 0.045), fontsize=10, color=C_ABS, ha="center", va="center", fontweight="bold")
axA.set_xlim(0, smax); axA.set_ylim(-0.25, 1.05)
axA.set_xlabel(r"disorder strength  $\sigma$")
axA.set_ylabel(r"interaction reciprocity  $\gamma=\mathrm{corr}(J_{ij},J_{ji})$")
axA.set_title("a  Corrected phase plane: only $\\gamma$ is constrained", loc="left", fontsize=12, color=INK)
for s in ("top", "right"): axA.spines[s].set_visible(False)

# ---- (b) measured reciprocity on the axis ----------------------------------
axB.axvspan(-0.05, 0.13, color=C_ABS, alpha=0.12)
axB.axvline(1.0, color=MUTED, ls=":", lw=1.1)
axB.axvline(0.30, color="#a34400", ls="--", lw=1.1, alpha=0.7)
axB.text(0.30, 2.62, "fold needs\n$\\gamma\\gtrsim0.3$", fontsize=9, color="#a34400", ha="center")
axB.text(1.0, 2.62, "spin\nglass", fontsize=9, color=MUTED, ha="center")
axB.text(0.04, 2.62, "measured\nband", fontsize=9, color=C_ABS, ha="center")
rng = np.random.default_rng(0)
# explicit label offsets (dx, dy, ha) for the four anchor points; others unlabelled
LAB = {"Vandeputte QMP": (0.03, 0.24, "left"), "Venturelli": (0.03, 0.22, "left"),
       "Clark": (-0.03, 0.22, "right"), "Vandeputte rel.": (0.0, 0.26, "center")}
for name, gm, tp in DATA:
    y = ROW[tp] + rng.uniform(-0.10, 0.10)
    axB.scatter(gm, y, s=95, color=COL[tp], edgecolor="white", linewidth=1.1, zorder=3)
    if name in LAB:
        dx, dy, ha = LAB[name]
        axB.annotate(name, (gm, y), (gm + dx, y + dy), fontsize=8.5, color=MUTED, ha=ha)
axB.set_yticks([0, 1, 2]); axB.set_yticklabels([ROWLAB[i] for i in (0, 1, 2)], fontsize=9.5)
axB.set_ylim(-0.6, 2.9); axB.set_xlim(-0.25, 1.05)
axB.set_xlabel(r"measured interaction reciprocity  $\gamma$")
axB.set_title("b  Seven cohorts + two matrices: the microbiome is weakly reciprocal",
              loc="left", fontsize=12, color=INK)
for s in ("top", "right"): axB.spines[s].set_visible(False)
axB.legend(handles=[Patch(fc=C_ABS, label="absolute abundance"),
                    Patch(fc=C_MAT, label="measured matrix"),
                    Patch(fc=C_REL, label="relative abundance")],
           loc="lower right", frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig(f"{OUT}/fig_hero.png", dpi=200, bbox_inches="tight")
print(f"[fig] {OUT}/fig_hero.png written")
