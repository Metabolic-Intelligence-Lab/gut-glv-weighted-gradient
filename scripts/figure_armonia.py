"""
ArMonIA data figure for the PRX manuscript (weak-perturbation / resistance pole).
Builds figC7_armonia.png from data/armonia/armonia_metrics.npz (run
analyze_armonia.py first). Model-free readouts, mapping-agnostic:
  (a) consecutive Bray-Curtis displacement decays monotonically (relaxation, not
      a random walk); baseline spread sets the intrinsic-variability scale.
  (b) displacement to the early-sample centroid stays within that baseline
      envelope for every subject (no basin-crossing).
  (c) Shannon equitability rises monotonically while richness settles: the
      community relaxes toward a more even, single-basin configuration.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = np.load(os.path.join(ROOT, "data", "armonia", "armonia_metrics.npz"),
            allow_pickle=True)
BC = D["BCpair"]                     # 4x4 mean Bray-Curtis
base_drift = D["base_drift"]         # per subject T1<->T2
rich, shan, equit = D["rich"], D["shan"], D["equit"]   # subjects x 4
ids = D["ids"]
T = ["T1", "T2", "T3", "T4"]
x = np.arange(4)

# per-subject consecutive displacements and displacement-to-baseline centroid
# (recomputed here from BCpair is not per-subject; reload tidy for per-subject)
# We stored base_drift, disp3, disp4 per subject:
disp3, disp4 = D["disp3"], D["disp4"]
pair3, pair4 = D["pair3"], D["pair4"]   # pairwise BC to the two baselines (apples-to-apples)

NAVY = "#1f3a5f"; TEAL = "#2a9d8f"; ORANGE = "#e76f51"; GREY = "#8a8a8a"
plt.rcParams.update({"font.size": 10, "axes.splines.top" if False else "axes.grid": False})

fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.7))

# ---- (a) consecutive displacement decays -------------------------------
cons = [BC[0, 1], BC[1, 2], BC[2, 3]]
labs = ["T1→T2\n(pre-diet)", "T2→T3\n(diet)", "T3→T4\n(post)"]
ax[0].bar(range(3), cons, color=[GREY, TEAL, TEAL], width=0.6, edgecolor="k", lw=0.6)
ax[0].axhline(base_drift.mean(), color=NAVY, ls="--", lw=1.2,
              label=f"pre-diet drift {base_drift.mean():.2f}")
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(labs, fontsize=8.5)
ax[0].set_ylabel("Bray-Curtis displacement")
ax[0].set_title("(a) Monotonic relaxation", fontsize=10, loc="left")
ax[0].legend(frameon=False, fontsize=8, loc="upper right")
ax[0].set_ylim(0, 0.75)

# ---- (b) displacement to baseline centroid stays within envelope -------
env = base_drift.mean()
for i in range(len(ids)):
    ax[1].plot([1, 2], [pair3[i], pair4[i]], "-", color=TEAL, alpha=0.35, lw=1)
ax[1].plot([1, 2], [pair3.mean(), pair4.mean()], "o-", color=NAVY, lw=2.2,
           ms=7, label="cohort mean")
ax[1].axhspan(0, env, color=GREY, alpha=0.15)
ax[1].axhline(env, color=NAVY, ls="--", lw=1.2)
ax[1].text(1.5, env + 0.01, "pre-diet drift (pairwise)", ha="center", fontsize=8, color=NAVY)
ax[1].set_xticks([1, 2]); ax[1].set_xticklabels(["T3", "T4"])
ax[1].set_xlim(0.7, 2.3); ax[1].set_ylim(0, 0.75)
ax[1].set_ylabel("pairwise BC to the two baselines")
ax[1].set_title("(b) Below pre-diet drift", fontsize=10, loc="left")
ax[1].legend(frameon=False, fontsize=8, loc="upper right")

# ---- (c) alpha diversity trajectories ----------------------------------
me = np.nanmean(equit, 0); se = np.nanstd(equit, 0) / np.sqrt((~np.isnan(equit)).sum(0))
mr = np.nanmean(rich, 0);  sr = np.nanstd(rich, 0) / np.sqrt((~np.isnan(rich)).sum(0))
ax2 = ax[2]
l1 = ax2.errorbar(x, me, yerr=se, color=ORANGE, marker="o", lw=2, capsize=3,
                  label="Shannon equitability")
ax2.set_ylabel("Shannon equitability", color=ORANGE)
ax2.tick_params(axis="y", labelcolor=ORANGE)
ax2.set_xticks(x); ax2.set_xticklabels(["T1\nApr", "T2\nMay", "T3\nJul", "T4\nSep"],
                                        fontsize=8.5)
ax2.axvspan(1.5, 3.5, color=TEAL, alpha=0.06)
ax2.text(2.5, ax2.get_ylim()[0], "diet (late Jun)", ha="center", va="bottom",
         fontsize=7.5, color=GREY)
axb = ax2.twinx()
l2 = axb.errorbar(x, mr, yerr=sr, color=NAVY, marker="s", lw=2, capsize=3,
                  ls="--", label="richness")
axb.set_ylabel("richness (# taxa)", color=NAVY)
axb.tick_params(axis="y", labelcolor=NAVY)
ax2.set_title("(c) Rising evenness", fontsize=10, loc="left")
ax2.legend(handles=[l1, l2], frameon=False, fontsize=8, loc="lower right")

for a in ax:
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
axb.spines["top"].set_visible(False)

fig.suptitle("ArMonIA human diet cohort (Nutrients 2023): weak-perturbation, "
             "within-basin relaxation", fontsize=11, y=1.02)
fig.tight_layout()
out = os.path.join(ROOT, "figures", "figC7_armonia.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
