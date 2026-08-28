"""
Place the real Gibson community on the phase diagram (Fig. 9).

Loads the inferred (mu, sigma, gamma) from infer_and_place.py, recomputes the
phase grid at the *inferred* competition level mu over an extended disorder
range, and overlays the data point with bootstrap error bars.

Run AFTER infer_and_place.py:  python3 place_on_phase_diagram.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from asymmetric_and_phases import classify

import warnings
warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

OUT = "figures"


def main():
    d = np.load(f"{OUT}/inferred_stats.npz")
    mu, sigma, gamma = float(d["mu"]), float(d["sigma"]), float(d["gamma"])
    bs, bg = d["boot_sigma"], d["boot_gamma"]

    S = 100
    r = np.ones(S)
    n_sigma, n_gamma = 40, 34
    sigmas = np.linspace(0.3, 5.0, n_sigma)
    gammas = np.linspace(-1.0, 1.0, n_gamma)
    import os
    cache = f"{OUT}/fig9_grid.npz"
    if (os.path.exists(cache) and abs(float(np.load(cache)["mu"]) - mu) < 1e-9
            and tuple(np.load(cache)["grid"].shape) == (len(gammas), len(sigmas))):
        grid = np.load(cache)["grid"]
        print("  [grid] loaded from cache")
    else:
        grid = np.zeros((len(gammas), len(sigmas)), dtype=int)
        for i, g in enumerate(gammas):
            for j, s in enumerate(sigmas):
                grid[i, j] = classify(S, r, s, g, seed=200 + j, mu=mu)
            if i % 5 == 0:
                print(f"  row {i+1}/{n_gamma} (gamma={g:+.2f}) done")
        np.savez(cache, grid=grid, mu=mu)          # cache -> instant re-plot on re-run

    from matplotlib.patches import Patch
    cmap = ListedColormap(["#1b9e77", "#7570b3", "#d95f02"])
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ds = (sigmas[1] - sigmas[0]) / 2; dg = (gammas[1] - gammas[0]) / 2
    se = np.linspace(sigmas[0] - ds, sigmas[-1] + ds, len(sigmas) + 1)
    ge = np.linspace(gammas[0] - dg, gammas[-1] + dg, len(gammas) + 1)
    ax.pcolormesh(se, ge, grid, cmap=cmap, vmin=0, vmax=2, shading="flat", alpha=0.9)

    # data point with bootstrap error bars. The absolute sigma from the ill-posed
    # fit is inflated (see recovery test), so the placement is qualitative and the
    # sigma coordinate is shown clamped to the axis.
    sx = min(sigma, sigmas[-1])
    xlo, xhi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    ax.errorbar(sx, gamma,
                xerr=[[sx - min(xlo, sigmas[-1])], [min(xhi, sigmas[-1]) - sx]],
                yerr=[[gamma - np.percentile(bg, 2.5)], [np.percentile(bg, 97.5) - gamma]],
                fmt="*", ms=20, color="black", ecolor="black", capsize=4,
                markerfacecolor="yellow", markeredgecolor="black", zorder=5,
                label=fr"Gibson data ($\gamma={gamma:.2f}$; $\sigma$ inflated, qualitative)")
    if sigma > sigmas[-1]:
        ax.annotate(fr"$\sigma_{{\rm fit}}\approx{sigma:.1f}$ (inflated)",
                    xy=(sigmas[-1], gamma), xytext=(sigmas[-1] - 1.6, gamma + 0.32),
                    fontsize=8, arrowprops=dict(arrowstyle="->"))

    ax.set_xlabel(r"disorder strength  $\sigma$")
    ax.set_ylabel(r"interaction symmetry  $\gamma$")
    ax.set_title("The real gut community on the phase diagram")
    handles = [Patch(fc="#1b9e77", label="unique fixed point (resilient)"),
               Patch(fc="#7570b3", label="multiple attractors (glassy)"),
               Patch(fc="#d95f02", label="persistent (chaotic)")]
    leg1 = ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.06),
                     ncol=1, fontsize=8, frameon=False)
    ax.add_artist(leg1)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig9_data_on_phase_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    counts = np.bincount(grid.ravel(), minlength=3)
    print(f"[fig9] written (UFP={counts[0]}, glassy={counts[1]}, chaos={counts[2]})")


if __name__ == "__main__":
    main()
