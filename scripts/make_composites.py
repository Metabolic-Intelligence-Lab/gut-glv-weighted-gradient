"""Build composite PRX figures (2x2) with (a)-(d) panel labels."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

OUT = "figures"
COMPOSITES = {
    "figC1_core":  ["fig1_three_regimes", "fig2_hysteresis_loop",
                    "fig3_overlap_Pq", "fig4_error_vs_attack"],
    "figC2_phase": ["fig5_glass_to_chaos", "fig6_phase_diagram",
                    "fig12_cavity_phase_boundary", "fig13_sparse_scalefree"],
    "figC3_data":  ["fig8_real_data_gibson", "fig9_data_on_phase_diagram",
                    "fig10_cohort_comparison", "fig18_vandeputte_human_absolute"],
    "figC4_appA":  ["fig14_distributional_cavity", "fig15_nonhermitian_sparse"],
    "figC5_appB":  ["fig7_proof_of_concept", "fig11_random_replicator",
                    "fig17_compositionality", "fig19_human_keystones"],
}
LBL = "abcd"


def build(name, panels):
    n = len(panels)
    ncol = 2
    nrow = (n + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.5, 5.1 * nrow))
    axes = axes.ravel() if n > 1 else [axes]
    for k in range(nrow * ncol):
        ax = axes[k]
        ax.axis("off")
        if k < n:
            ax.imshow(mpimg.imread(f"{OUT}/{panels[k]}.png"))
            ax.text(-0.015, 1.015, f"({LBL[k]})", transform=ax.transAxes,
                    fontsize=20, fontweight="bold", va="bottom", ha="right")
    fig.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.4)
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png  ({n} panels)")


if __name__ == "__main__":
    print("Composites:")
    for name, panels in COMPOSITES.items():
        build(name, panels)
