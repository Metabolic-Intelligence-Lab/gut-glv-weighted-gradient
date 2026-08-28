# Code for "A weighted-gradient signature in designed gut-community dynamics, and why inferred interaction matrices hide it"

Giuseppe Maulucci (ORCID 0000-0002-2154-319X), Dipartimento di Neuroscienze, Sezione di Biofisica, Università Cattolica del Sacro Cuore, Rome, Italy; Fondazione Policlinico Universitario A. Gemelli IRCCS, Rome, Italy.
Manuscript submitted to *PLOS Computational Biology*, 2026.

This repository holds every analysis script behind the paper's figures and numerical claims,
the fixed-seed caches that let the slow sweeps regenerate in minutes instead of hours, and the
rendered figures. It holds **no data**: the eleven datasets analysed belong to their original
publications, and `DATA.md` says where each one comes from and where the scripts expect it.

    scripts/       one self-contained Python script per analysis; the table below maps figures to scripts
    figures/       rendered figures (.png) and deterministic caches (.npz) of the expensive sweeps
    DATA.md        provenance of every dataset, and how to lay out data/ for the scripts
    CITATION.cff   how to cite
    LICENSE        MIT

## Reproducing

    pip install numpy scipy matplotlib networkx pandas openpyxl rdata pillow
    export OMP_NUM_THREADS=4          # multi-threaded BLAS thrashes on the many small eigensolves
    PYTHONPATH=scripts python3 scripts/lyapunov_robustness.py    # Fig 1: the descent test, out of sample

Run scripts from the repository root. Each one regenerates its figure from `data/` and/or the
caches in `figures/`; delete a cache to force its sweep to re-run. The shared inference core
for the longitudinal cohorts is `scripts/infer_and_place.py`. Scripts marked deprecated in
their docstrings belong to analyses the paper retracts or does not cite; they are kept so the
retraction can be audited.

## Figures and the scripts that make them

| Figure | File | Script |
|---|---|---|
| Fig 1 | `figures/fig_lyapunov_robustness.png` | `lyapunov_robustness.py` |
| Fig 2 | `figures/fig_invariants.png` | `symmetrizability_invariants.py` |
| Fig 3 | `figures/fig_inference_null.png` | `inference_aware_null.py` |
| Fig 4 | `figures/fig_constrained_free.png` | `constrained_vs_free_glv.py` |
| Fig 5 | `figures/fig_power.png` | `detection_power.py, power_analysis.py` |
| Fig 6 | `figures/fig20_gamma_recovery.png` | `recovery_gamma.py` |
| Fig 7 | `figures/fig10_cohort_comparison.png` | `compare_cohorts.py, make_composites.py` |
| Fig 8 | `figures/fig18_vandeputte_human_absolute.png` | `infer_vandeputte.py, make_composites.py` |
| Fig 9 | `figures/fig22_venturelli_groundtruth.png` | `venturelli_ground_truth.py` |
| Fig 10 | `figures/fig8_real_data_gibson.png` | `analyze_gibson.py, make_composites.py` |
| Fig 11 | `figures/fig23_david_reversibility.png` | `david_daily_reversibility.py` |
| Fig 12 | `figures/figC7_armonia.png` | `figure_armonia.py` |
| Fig 13 | `figures/fig24_keystone_validation.png` | `venturelli_keystone_validation.py` |
| Fig 14 | `figures/fig_spectral_vs_knockout.png` | `spectral_vs_knockout.py` |
| Fig A (S1) | `figures/fig_infnull_compositional.png` | `inference_aware_null_compositional.py` |
| Fig B (S1) | `figures/fig_lyapunov_descent.png` | `lyapunov_descent.py` |
| Fig C (S1) | `figures/fig_gamma_confidence.png` | `gamma_confidence.py` |
| Fig D (S1) | `figures/fig_hierboot_cohorts.png` | `hierarchical_bootstrap_cohorts.py` |
| Fig E (S1) | `figures/fig_fold_reciprocity.png` | `fold_vs_reciprocity.py` |
| Fig F (S1) | `figures/fig17_compositionality.png` | `compositionality_test.py, make_composites.py, validate_keystones.py` |
| Fig G (S1) | `figures/fig19_human_keystones.png` | `human_keystones.py, make_composites.py` |
| Fig H (S1) | `figures/fig_keystone_robust.png` | `keystone_robustness.py` |
