"""
Fig. 1 (model schematic): the network, the interaction matrix, and the landscape.
Answers 'show the network' and 'what is the model / how many species'.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from spin_glass_microbiome import enterotype_couplings

S, M = 120, 2
rng = np.random.default_rng(3)
G = nx.barabasi_albert_graph(S, M, seed=3)
deg = np.array([d for _, d in G.degree()])
guild = (np.arange(S) % 2)                      # two enterotypes A / B
hubs = np.argsort(deg)[::-1][:4]                # keystone hubs
A = nx.to_numpy_array(G)
J = enterotype_couplings(A, guild, w_in=0.4, w_out=1.2, sigma=0.12,
                         rng=np.random.default_rng(11))

fig = plt.figure(figsize=(13, 4.5))
gs = GridSpec(1, 3, width_ratios=[1.25, 1.0, 1.1], wspace=0.28)

# (a) network
axN = fig.add_subplot(gs[0, 0])
pos = nx.spring_layout(G, seed=1, k=0.25)
colA, colB = "#1b9e77", "#7570b3"
ncolor = [colA if g == 0 else colB for g in guild]
nx.draw_networkx_edges(G, pos, ax=axN, edge_color="#cccccc", width=0.5, alpha=0.7)
nx.draw_networkx_nodes(G, pos, ax=axN, node_size=8 + 3.2 * deg,
                       node_color=ncolor, linewidths=0, alpha=0.9)
nx.draw_networkx_nodes(G, pos, ax=axN, nodelist=list(hubs),
                       node_size=8 + 3.2 * deg[hubs], node_color="none",
                       edgecolors="#d95f02", linewidths=2.4)
axN.text(0.5, -0.06, f"$S={S}$ species, Barabási–Albert ($m={M}$, $\\langle k\\rangle={deg.mean():.0f}$)",
         transform=axN.transAxes, ha="center", fontsize=10)
axN.text(0.02, 0.98, "(a)", transform=axN.transAxes, fontsize=15, fontweight="bold", va="top")
axN.text(pos[hubs[0]][0], pos[hubs[0]][1] + 0.08, "keystone\nhubs", fontsize=8,
         color="#d95f02", ha="center")
axN.set_axis_off()
# legend
import matplotlib.lines as mlines
h1 = mlines.Line2D([], [], marker="o", color="w", markerfacecolor=colA, markersize=9, label="enterotype A")
h2 = mlines.Line2D([], [], marker="o", color="w", markerfacecolor=colB, markersize=9, label="enterotype B")
h3 = mlines.Line2D([], [], marker="o", color="w", markeredgecolor="#d95f02", markerfacecolor="none", markeredgewidth=2, markersize=11, label="keystone hub")
axN.legend(handles=[h1, h2, h3], loc="upper right", fontsize=8, frameon=False)

# (b) interaction matrix J
axJ = fig.add_subplot(gs[0, 1])
order = np.argsort(guild)                        # block-order by guild
im = axJ.imshow(J[np.ix_(order, order)], cmap="RdBu_r", vmin=-0.4, vmax=0.4)
axJ.set_title("interaction matrix  $J_{ij}$", fontsize=10)
axJ.set_xlabel("species $j$"); axJ.set_ylabel("species $i$")
axJ.text(0.02, 0.98, "(b)", transform=axJ.transAxes, fontsize=15, fontweight="bold",
         va="top", color="black")
cb = fig.colorbar(im, ax=axJ, fraction=0.046, pad=0.04)
cb.set_label("+ mutualism   /   − competition", fontsize=8)

# (c) energy landscape schematic
axL = fig.add_subplot(gs[0, 2])
x = np.linspace(-1.7, 1.7, 400)
V = (x**2 - 1)**2 + 0.12 * x                      # asymmetric double well
axL.plot(x, V, "k-", lw=2)
axL.fill_between(x, V, V.max() + 0.2, color="white")
# basins
axL.scatter([-1.02], [(((-1.02)**2-1)**2 + 0.12*-1.02)], s=120, color="#1b9e77", zorder=5)
axL.scatter([1.0], [((1.0**2-1)**2 + 0.12*1.0)], s=120, color="#7570b3", zorder=5)
axL.annotate("", xy=(-0.7, 0.55), xytext=(-1.02, 0.12),
             arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=1.6))
axL.text(-1.45, 0.55, "resistance\n(deep well)", fontsize=8, color="#1b9e77")
axL.annotate("", xy=(0.0, 1.05), xytext=(-1.0, 0.15),
             arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.8))
axL.text(-0.15, 1.15, "perturbation $h$", fontsize=8, color="#d95f02")
axL.text(0.55, 0.18, "alternative\nstate (hysteresis)", fontsize=8, color="#7570b3")
axL.text(-1.25, 0.18, "baseline", fontsize=8, color="#1b9e77")
axL.set_xlabel("community state (order parameter)")
axL.set_ylabel("free energy  $H$")
axL.set_yticks([]); axL.set_xticks([])
axL.set_ylim(-0.1, 1.5)
axL.set_title("rugged landscape: basins, barrier, tipping", fontsize=10)
axL.text(0.02, 0.98, "(c)", transform=axL.transAxes, fontsize=15, fontweight="bold", va="top")

fig.savefig("figures/figC0_model.png", dpi=170, bbox_inches="tight")
print("figC0_model.png written | S=%d, <k>=%.1f, hubs(deg)=%s" % (S, deg.mean(), deg[hubs].tolist()))
