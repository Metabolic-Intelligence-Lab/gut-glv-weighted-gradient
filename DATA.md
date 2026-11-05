# Data provenance

> **Corrected and completed 5 November 2026.** This file previously attributed both
> `david2014/` and `david_daily/` to the *Genome Biology* daily-series paper. That is the
> misattribution referee 4 raises in point 8: the five-day diet swap is the *Nature* paper, and
> the two have different accessions, now given separately below.
>
> **Every accession below was looked up, not recalled.** Each was taken from the depositing
> study itself: from its data-availability statement where the full text is open, or from the
> registry record whose title matches the study. The route is recorded next to each one, so a
> reader can repeat the lookup. Where a paper's own statement was the source, its wording is
> what the entry reflects.
>
> **What the accessions are, and are not.** They identify the sequence data each cohort
> deposited. This work consumed none of it: every cohort enters as the abundance table its
> authors published, as the main text states and as the "obtained from" column below says.
> The accessions are given because referee 4 asks for them, and because they identify the
> deposit behind each published table, not the file the scripts read.

The analysis code in `scripts/` reads raw datasets from a `data/` directory that is
**not** included in this archive. Each dataset belongs to its original publication and
is obtained from the source below. The ArMonIA cohort is not included either: it is a
published study (Bianchetti G et al., *Nutrients* 15:3931, 2023) whose data are available
on request as stated in that publication. Only `figure_armonia.py` and `analyze_armonia.py`
need it, and only for one figure; every other script runs from the public sources below.

| `data/` subdirectory | Source publication | Where to obtain it |
|---|---|---|
| `gibson_healthy/`, `gibson_uc/` | Gibson TE et al., *Nat Microbiol* 10:2550 (2025) — MDSINE2 | Published abundance tables: the paper's source data and the MDSINE2 repository. Sequence deposit: **BioProject PRJNA784519**; intermediate files for the figures at **Zenodo 10.5281/zenodo.8208503**. Both from the paper's data-availability statement |
| `david2014/` | David LA et al., *Nature* 505:559 (2014) — the five-day controlled diet swap, 11 participants in a crossover design, 20 subject-diet series | The paper's supplementary data. Sequence deposit: **BioProject PRJNA202303**, whose registry title is the paper's own, "Diet rapidly and reproducibly alters the human gut microbiome". This is the accession referee 4 asks for in point 8 |
| `david_daily/` | David LA et al., *Genome Biol* 15:R89 (2014) — the daily series, 2 subjects, travel and enteric infection | The paper's supplementary data. Sequence deposit: **ENA ERP006059**, from this paper's own data-availability statement |
| `digiulio/` | DiGiulio DB et al., *PNAS* 112:11060 (2015) | Published abundance table: the paper's supplementary data. Sequence deposit: **BioProject PRJNA288562**, "Temporal and Spatial Variation of the Human Microbiota During Pregnancy", linked from the PubMed record |
| `bokulich/` | Bokulich NA et al., *Sci Transl Med* 8:343ra82 (2016) | Published abundance table: the paper's supplementary data. Sequence deposit: **BioProject PRJEB14529**, "Antibiotics, birth mode, and diet shape microbiome maturation during early life", linked from the PubMed record |
| `karelia/` | Vatanen T et al., *Cell* 165:842 (2016) — DIABIMMUNE | Published abundance table: the DIABIMMUNE data portal, which is the route used here. Sequence deposit: **PRJNA290380**, the ENA/BioProject study titled "A longitudinal analysis of the developing gut microbiome in infants from Finland, Estonia, and Russian Karelia". Identified by that title matching the cohort, not by a publication link: the paper is not open access, so its own statement could not be read |
| `vandeputte/` | Vandeputte D et al., *Nat Commun* 12:6740 (2021) | Published QMP profiles: the paper's Supplementary Data S1, read here as `temporalMicrobiome20.rda` (needs the R `rdata` Python package). Sequence deposit: **EGA EGAS00001005686**, with public access, from the paper's data-availability statement |
| `venturelli2018/` | Venturelli OS et al., *Mol Syst Biol* 14:e8157 (2018) | Trajectories: Dataset EV3. Fitted matrices: the SBML models T1-T4 in Code EV1 (`MSB-14-e8157-s006.zip`). Sequence deposit: **ENA PRJEB26607**; the original authors' scripts at **Zenodo 10.5281/zenodo.1248186** |
| `clark2021/` | Clark RL et al., *Nat Commun* 12:3254 (2021) | github.com/RyanLincolnClark/DesignSyntheticGutMicrobiomeAssemblyFunction (branch `main`); `data/clark2021/PROVENANCE.txt` records exactly which files were taken |
| `armonia/` | Bianchetti G et al., *Nutrients* 15:3931 (2023) | Published study; the genus-level abundance table is available on request as stated there. No sequence accession: the raw sequences are not redistributed, as participant consent does not cover it |

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
