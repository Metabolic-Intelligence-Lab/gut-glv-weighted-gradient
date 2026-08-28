# Data provenance

The analysis code in `scripts/` reads raw datasets from a `data/` directory that is
**not** included in this archive. Each dataset belongs to its original publication and
is obtained from the source below. The ArMonIA cohort is not included either: it is a
published study (Bianchetti G et al., *Nutrients* 15:3931, 2023) whose data are available
on request as stated in that publication. Only `figure_armonia.py` and `analyze_armonia.py`
need it, and only for one figure; every other script runs from the public sources below.

| `data/` subdirectory | Source publication | Where to obtain it |
|---|---|---|
| `gibson_healthy/`, `gibson_uc/` | Gibson TE et al., *Nat Microbiol* 10:2550 (2025) — MDSINE2 | MDSINE2 repository / the paper's supplementary data |
| `david2014/`, `david_daily/` | David LA et al., *Genome Biol* 15:R89 (2014) | The paper's supplementary data |
| `digiulio/` | DiGiulio DB et al., *PNAS* 112:11060 (2015) | The paper's supplementary data |
| `bokulich/` | Bokulich NA et al., *Sci Transl Med* 8:343ra82 (2016) | The paper's supplementary data |
| `karelia/` | Vatanen T et al., *Cell* 165:842 (2016) — DIABIMMUNE | DIABIMMUNE data portal |
| `vandeputte/` | Vandeputte D et al., *Nat Commun* 12:6740 (2021) | The paper's supplementary data (`temporalMicrobiome20.rda`; needs the R `rdata` Python package to read) |
| `venturelli2018/` | Venturelli OS et al., *Mol Syst Biol* 14:e8157 (2018) | Supplementary dataset EV3 |
| `clark2021/` | Clark RL et al., *Nat Commun* 12:3254 (2021) | github.com/RyanLincolnClark/DesignSyntheticGutMicrobiomeAssemblyFunction (branch `main`); `data/clark2021/PROVENANCE.txt` records exactly which files were taken |
| `armonia/` | Bianchetti G et al., *Nutrients* 15:3931 (2023) | Published study; data available on request as stated there |

# Reproducing the figures

Each script under `scripts/` is single-file and self-contained. Run them from the
repository root with `PYTHONPATH=scripts`. The manuscript's code-to-figure table maps
every figure to its generating script.

Two practical notes carried over from development:

* Set `OMP_NUM_THREADS=4` for the scripts that do many small eigendecompositions or
  matrix exponentials. Multi-threaded BLAS thrashes on those workloads and can make a
  one-minute job take ten.
* The `.npz` files under `figures/` are deterministic, fixed-seed caches of the
  expensive sweeps (the phase-diagram ODE grid, the γ-recovery distribution, the
  Clark posterior draws). Delete one to force its sweep to re-run.
