"""
Regenerate all composite-relevant figures WITHOUT the internal 'Fig. N - ...'
titles (keeping descriptive A./B. sub-titles), then rebuild the composites.

We monkeypatch matplotlib so that:
  - any Axes.set_title whose text starts with 'Fig' is blanked,
  - Figure.suptitle is suppressed (these are always the 'Fig. N' banners here),
without editing the 15 plotting scripts.
"""
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.figure import Figure

_orig_set_title = Axes.set_title
def _patched_set_title(self, label="", *a, **k):
    if isinstance(label, str) and label.strip().lower().startswith("fig"):
        label = ""
    return _orig_set_title(self, label, *a, **k)
Axes.set_title = _patched_set_title
Figure.suptitle = lambda *a, **k: None

import numpy as np
np.seterr(all="ignore")


def run(desc, fn):
    try:
        fn(); print(f"  ok  {desc}")
    except Exception as e:
        print(f"  FAIL {desc}: {type(e).__name__}: {e}")


# fast core
import spin_glass_microbiome as M1
run("fig1-4 core", lambda: (M1.fig_three_regimes(), M1.fig_hysteresis_loop(),
                            M1.fig_overlap(), M1.fig_error_attack()))
import proof_of_concept as M7; run("fig7", M7.main)
import analyze_gibson as M8; run("fig8", M8.main)
import compositionality_test as M17; run("fig17", M17.main)
import infer_vandeputte as M18; run("fig18", M18.main)
import human_keystones as M19; run("fig19", M19.main)
import fig11_random_replicator as M11; run("fig11", M11.main)
import sparse_cavity as M13; run("fig13", M13.main)
import nonhermitian_sparse as M15; run("fig15", M15.main)
import pop_dynamics_cavity as M14; run("fig14", M14.main)
# slower (phase grids)
import asymmetric_and_phases as MA
run("fig5,6", lambda: (MA.fig_glass_to_chaos(), MA.fig_phase_diagram()))
import cavity_theory as M12; run("fig12 (cached grid)", M12.main)
import place_on_phase_diagram as M9; run("fig9", M9.main)
import compare_cohorts as M10; run("fig10", M10.main)

# rebuild composites
import make_composites as MC
print("Rebuilding composites:")
for name, panels in MC.COMPOSITES.items():
    MC.build(name, panels)
print("DONE_REGEN")
