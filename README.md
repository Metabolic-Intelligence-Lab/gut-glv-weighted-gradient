# Code for "An inference-aware protocol for testing weighted-gradient (energy-landscape) structure in gut microbial Lotka-Volterra models"

Giuseppe Maulucci (ORCID 0000-0002-2154-319X), Dipartimento di Neuroscienze, Sezione di Biofisica, Università Cattolica del Sacro Cuore, Rome, Italy; Fondazione Policlinico Universitario A. Gemelli IRCCS, Rome, Italy.
Manuscript under revision at *Briefings in Bioinformatics*, 2026 (Problem Solving
Protocol). The paper was peer-reviewed by four referees in October 2026 and this archive
is the state of the code at the revision: the figure table below follows the revised
numbering, eight main figures and nineteen in S1 Text. An earlier, longer version was
submitted to *PLOS Computational Biology* in August 2026.

The revision changed what the analyses claim, not only how they are presented. Several
claims were narrowed against their own calibration, and the scripts added here are the
ones that forced those narrowings: the calibration ensemble, the Lyapunov counterexample,
the replicated-support cycle null, the design sweep and the gauge analysis.

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
| Fig 1 | `figures/fig_lyapunov_robustness_v2.png` | `lyapunov_robustness.py` |
| Fig 2 | `figures/fig_descent_specificity_ensemble.png` | `descent_specificity_ensemble.py` |
| Fig 3 | `figures/fig_invariants_v2.png` | `symmetrizability_invariants.py` |
| Fig 4 | `figures/fig_inference_null_v2.png` | `inference_aware_null.py` |
| Fig 5 | `figures/fig_constrained_free_v2.png` | `constrained_vs_free_glv.py` |
| Fig 6 | `figures/fig_detection_power_calibrated.png` | `detection_power_calibrated.py` |
| Fig 7 | `figures/fig20_gamma_recovery_v2.png` | `recovery_gamma_v2.py` |
| Fig 8 | `figures/fig10_cohort_comparison_v2.png` | `compare_cohorts.py` |
| Fig A (S1) | `figures/fig_infnull_compositional.png` | `inference_aware_null_compositional.py` |
| Fig B (S1) | `figures/fig_lyapunov_descent.png` | `lyapunov_descent.py` |
| Fig C (S1) | `figures/fig_gamma_confidence.png` | `gamma_confidence.py` |
| Fig D (S1) | `figures/fig_hierboot_cohorts.png` | `hierarchical_bootstrap_cohorts.py` |
| Fig E (S1) | `figures/fig_fold_reciprocity.png` | `fold_vs_reciprocity.py` |
| Fig F (S1) | `figures/fig_descent_cohorts.png` | `descent_cohorts.py` |
| Fig G (S1) | `figures/fig_invariants_scaling.png` | `invariants_scaling.py` |
| Fig H (S1) | `figures/fig_descent_best_case_D_fast.png` | `descent_best_case_D_fast.py` |
| Fig I (S1) | `figures/fig_alternative_estimators.png` | `alternative_estimators.py` |
| Fig J (S1) | `figures/fig17_compositionality.png` | `compositionality_test.py` |
| Fig K (S1) | `figures/fig19_human_keystones.png` | `human_keystones.py` |
| Fig L (S1) | `figures/fig_knockout_robustness_rev.png` | `knockout_robustness_rev.py` |
| Fig M (S1) | `figures/fig8_real_data_gibson.png` | `analyze_gibson.py` |
| Fig N (S1) | `figures/fig23_david_reversibility.png` | `david_daily_reversibility.py` |
| Fig O (S1) | `figures/figC7_armonia.png` | `figure_armonia.py` |
| Fig P (S1) | `figures/fig18_vandeputte_human_absolute.png` | `infer_vandeputte.py` |
| Fig Q (S1) | `figures/fig22_venturelli_groundtruth.png` | `venturelli_ground_truth.py` |
| Fig R (S1) | `figures/fig24_keystone_validation_T3.png` | `venturelli_keystone_validation.py` |
| Fig S (S1) | `figures/fig_spectral_vs_knockout_T3.png` | `spectral_vs_knockout.py` |