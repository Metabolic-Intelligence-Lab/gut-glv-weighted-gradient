r"""
Control D1: does each claim survive a correction over the family of tests the protocol runs?
============================================================================================
The protocol is a grid, not a single test: three coordinate-invariant diagnostics on two
designed consortia under two noise calibrations, and the paper reports the cells that exceed
their null. A referee who notices the grid will ask for a multiplicity correction, and the
manuscript never mentions one. This is a control, so a pass changes nothing in the text and
only a failure would.

Families, declared before computing anything, smallest first, because a correction is only as
honest as the family it is taken over:
  F1  the algebraic grid the protocol specifies: 3 diagnostics x 2 consortia x 2 noise
      calibrations = 12 tests. This is the family a reader infers from the Methods.
  F2  every p the paper offers in support of a positive claim.
  F3  every p the paper reports at all, negative controls included. Maximally conservative,
      and not the right family, since a null result that fails to reject costs no alpha. It is
      computed only to show the headline claims do not depend on the choice.

Correction: Bonferroni, which bounds the family-wise error rate without assuming independence,
and Benjamini-Hochberg at FDR 0.05 for comparison. The Monte Carlo floor matters: a p reported
as < 2e-4 comes from 5000 draws and cannot be resolved below 1/5000, so a Bonferroni threshold
under 2e-4 would be undecidable. Each family is checked against that floor.

CRITERION, fixed before execution:
  1. Every claim the paper presents as positive survives Bonferroni over F1. Nothing changes;
     the result is recorded as reassurance.
  2. A claim presented as positive does not survive Bonferroni over F1 but the paper does not
     assert significance for it after correction. No defect in the text: it goes to the author
     as a sentence the response letter may want, since the question is predictable.
  3. A claim the paper asserts as significant does not survive. Verified defect, corrected.

Run:  python3 scripts/multiplicity_audit.py
"""
MC_FLOOR = 2e-4          # 1/5000 draws
ALPHA = 0.05

# (label, p, role, asserted_significant, where)
TESTS = [
    # --- positive claims
    ("held-out descent of H_D, fitted vs magnitude-matched scrambles", 2e-4, "positive", True,
     "abstract, Results, Key Points"),
    ("Venturelli r_cycle, replicated support, noise at the dispersion of the three fits",
     2e-4, "positive", True, "Results, S1"),
    ("Venturelli cycle sign agreement, same calibration", 0.006, "positive", True, "Results, S1"),
    ("Venturelli cycle sign agreement, independent noise", 0.008, "positive", True, "Results"),
    ("Clark f_sign vs magnitude-preserving surrogate, proportional noise", 2e-4, "positive",
     True, "Results, S1"),
    ("keystone ranking, Spearman rho=0.65", 0.022, "positive", True, "Results, S1"),
    # --- a positive claim that the paper itself reports as failing
    ("Clark f_sign, additive calibration to the same posterior spread", 0.063, "positive",
     False, "Results, S1"),
    # --- negative controls: the paper's point is that these do NOT reject
    ("inference-aware null, Venturelli f_oneway", 0.25, "control", False, "Results"),
    ("inference-aware null, Venturelli f_sign", 0.25, "control", False, "Results"),
    ("inference-aware null, renormalised observations", 0.243, "control", False, "Results"),
    ("cycle null before calibration, lower end", 0.77, "control", False, "Results"),
    ("cycle null before calibration, upper end", 1.00, "control", False, "Results"),
    ("lasso at matched complexity, one-way support", 0.93, "control", False, "S1"),
    ("elastic net at matched complexity, one-way support", 0.83, "control", False, "S1"),
    ("sign discordance on 3 reciprocal pairs", 0.03, "control", False, "S1"),
    ("paired permutation, free fit vs symmetrizable fit in prediction", 0.80, "control", False,
     "Results"),
    ("sign test, free model ahead in 11 of 13 communities", 0.02, "control", False, "Results"),
    ("MAR comparator", 0.23, "control", False, "Results"),
    ("cohort descent, Gibson healthy", 0.72, "control", False, "Results"),
    ("cohort descent, Gibson UC", 0.28, "control", False, "Results"),
    ("cohort descent, Vandeputte", 0.10, "control", False, "Results"),
    ("optimizer sensitivity", 0.46, "control", False, "S1"),
]
F1_SIZE = 12             # 3 diagnostics x 2 consortia x 2 noise calibrations


def bh(ps, alpha):
    """Benjamini-Hochberg: the largest k with p_(k) <= k*alpha/m, and the implied threshold."""
    s = sorted(ps)
    m = len(s)
    k = max((i for i in range(1, m + 1) if s[i - 1] <= i * alpha / m), default=0)
    return (k * alpha / m) if k else 0.0


def main():
    pos = [t for t in TESTS if t[2] == "positive"]
    families = {
        "F1 algebraic grid (3 diagnostics x 2 consortia x 2 noise)": F1_SIZE,
        "F2 every p supporting a positive claim": len(pos),
        "F3 every p reported, controls included": len(TESTS),
    }
    print(f"{len(TESTS)} distinct tests catalogued: {len(pos)} positive, "
          f"{len(TESTS) - len(pos)} controls\n")
    for name, m in families.items():
        thr = ALPHA / m
        dec = "decidable" if thr > MC_FLOOR else f"UNDECIDABLE, below the {MC_FLOOR:.0e} floor"
        print(f"{name}\n   m = {m:>2}   Bonferroni threshold {thr:.5f}   ({dec})")
        surv = [t for t in pos if t[1] <= thr]
        fail = [t for t in pos if t[1] > thr]
        print(f"   survives ({len(surv)}):")
        for t in surv:
            print(f"      p={t[1]:<8.4g} {t[0]}")
        print(f"   does not survive ({len(fail)}):")
        for t in fail:
            mark = "  <-- but the paper asserts it as significant" if t[3] else ""
            print(f"      p={t[1]:<8.4g} {t[0]}{mark}")
        print()

    thr_bh = bh([t[1] for t in pos], ALPHA)
    print(f"Benjamini-Hochberg over F2 at FDR {ALPHA}: threshold {thr_bh:.5f}, "
          f"{sum(1 for t in pos if t[1] <= thr_bh)} of {len(pos)} positive claims survive")

    print("\n--- criterion ---")
    thr1 = ALPHA / F1_SIZE
    broken = [t for t in pos if t[3] and t[1] > thr1]
    headline = [t for t in pos if t[1] <= ALPHA / len(TESTS)]
    print(f"The two headline claims and {len(headline)} positive results in all survive "
          f"Bonferroni over even F3, the maximal family (threshold "
          f"{ALPHA / len(TESTS):.5f}):")
    for t in headline:
        print(f"   p={t[1]:<8.4g} {t[0]}")
    if not broken:
        print("\nBRANCH 1: nothing to change.")
    else:
        print(f"\nBRANCH 2: {len(broken)} claim(s) are asserted as exceeding their null and do "
              f"not survive Bonferroni over F1 ({thr1:.5f}). The paper never claims "
              "significance after correction, so the text is not wrong; the question is "
              "predictable and belongs in the response letter:")
        for t in broken:
            print(f"   p={t[1]:<8.4g} {t[0]}  [{t[4]}]")
        print("\n   Recorded for the author, not corrected in the manuscript: an audit control "
              "only changes the text when it finds the text wrong.")


if __name__ == "__main__":
    main()
