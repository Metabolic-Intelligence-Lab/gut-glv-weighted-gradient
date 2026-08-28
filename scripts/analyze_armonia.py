"""
ArMonIA cohort (Bianchetti, De Maio, Abeltino, ... Maulucci, Nutrients 2023,
15(18):3931) - the authors' own human diet-intervention cohort.

Design (dates confirmed in 2021_ArMonIA/analisi_db_ottobre22/microbiome_date.xlsx,
sheet 'calcoli'): 10 subjects (WL010105-WL010114), 4 faecal 16S time points.
Two PRE-diet baselines (T1 ~April, T2 ~May 2022), a one-month personalized-nutrition
intervention beginning 28 June (Mediterranean for 9, ketogenic for 1), then two
post-diet samples (T3 ~July, T4 ~Aug-Sep). The largest Bray-Curtis step is the
pre-diet T1->T2 baseline drift, before any intervention.

This is a MILD, within-host perturbation with a built-in null: the T1->T2
baseline drift measures intrinsic variability with no intervention. We therefore
use ArMonIA as a MODEL-FREE resistance test (Bray-Curtis displacement, alpha
diversity, enterotype stability), NOT for gLV inference: 3 transitions/subject is
far too sparse for a reliable interaction matrix (same reason the infant cohorts
were left out). It fills the weak-perturbation end of the empirical spectrum, in
contrast with the antibiotic basin-crossing seen in Gibson et al.

Reads the longitudinal xlsx files straight from the ArMonIA OneDrive folder and
writes a tidy CSV under data/armonia/ plus figures.
"""
import os, glob
import numpy as np
import openpyxl

ARM = ("/Users/giuseppe.maulucci/Library/Group Containers/UBF8T346G9."
       "OneDriveStandaloneSuite/OneDrive - Università Cattolica del Sacro Cuore"
       ".noindex/OneDrive - Università Cattolica del Sacro Cuore/"
       "Metabolic Intelligence - Projects-MI/2021_ArMonIA/Microbiota/"
       "AnalisiMicrobiota")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "armonia")
os.makedirs(OUT, exist_ok=True)
TIMES = ["time1", "time2", "time3", "time4"]


def _num(v):
    if v is None:
        return np.nan
    if isinstance(v, str):
        s = v.strip()
        if s in ("", "#N/A", "NA", "N/A", "#DIV/0!"):
            return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan
    return float(v)


def load_subject(path, sheet="Genus_tot"):
    """Return {taxon: [t1,t2,t3,t4]} of relative abundances (%), NaN->0."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    data = {}
    for r in rows[1:]:
        name = r[0]
        if not name or str(name).strip() == "":
            continue
        vals = [_num(r[i]) for i in range(1, 5)]
        data[str(name).strip()] = np.nan_to_num(np.array(vals), nan=0.0)
    return data


def matrix_for_subject(data, taxa):
    """taxa x 4 matrix, renormalised to fractions per time column."""
    M = np.array([data.get(t, np.zeros(4)) for t in taxa])  # taxa x 4
    col = M.sum(0)
    col[col == 0] = 1.0
    return M / col  # fractions summing to 1 per time point


def bray_curtis(u, v):
    return np.abs(u - v).sum() / (u.sum() + v.sum())


def load_alpha():
    wb = openpyxl.load_workbook(
        os.path.join(ARM, "microbiota_allrounds", "alphadiversity_totale.xlsx"),
        read_only=True, data_only=True)
    ws = wb["alphadiv"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    rec = {}
    for r in rows[1:]:
        if not r[0]:
            continue
        s, rich, shan, equit, t = r[0], _num(r[1]), _num(r[2]), _num(r[3]), r[4]
        # Shannon index stored as integer x1000 in this sheet (e.g. 3656 -> 3.656)
        shan = shan / 1000.0 if shan and shan > 100 else shan
        rec.setdefault(str(s).strip(), {})[str(t).strip()] = (rich, shan, equit)
    return rec


def main():
    subs = sorted(glob.glob(os.path.join(ARM, "longitudinalAnalysis", "WL*.xlsx")))
    ids = [os.path.splitext(os.path.basename(p))[0] for p in subs]
    genus = {i: load_subject(p, "Genus_tot") for i, p in zip(ids, subs)}
    phylum = {i: load_subject(p, "Phylum_tot") for i, p in zip(ids, subs)}

    # ---- tidy CSV (genus level) ----------------------------------------
    all_taxa = sorted({t for d in genus.values() for t in d})
    with open(os.path.join(OUT, "armonia_genus_tidy.csv"), "w") as f:
        f.write("subject,time,taxon,rel_abund\n")
        for i in ids:
            for k, t in enumerate(TIMES):
                for tax in all_taxa:
                    v = genus[i].get(tax, np.zeros(4))[k]
                    if v != 0:
                        f.write(f"{i},{t},{tax},{v:.6f}\n")

    # ---- pairwise Bray-Curtis structure (mean over subjects) -----------
    print("=== Genus-level Bray-Curtis, mean over 10 subjects ===")
    BCpair = np.zeros((4, 4))
    per_sub = {i: np.zeros((4, 4)) for i in ids}
    for i in ids:
        taxa = sorted(genus[i])
        M = matrix_for_subject(genus[i], taxa)  # taxa x 4
        for a in range(4):
            for b in range(4):
                d = bray_curtis(M[:, a], M[:, b])
                per_sub[i][a, b] = d
                BCpair[a, b] += d / len(ids)
    hdr = "        " + "".join(f"{t:>8}" for t in TIMES)
    print(hdr)
    for a, t in enumerate(TIMES):
        print(f"{t:>8}" + "".join(f"{BCpair[a,b]:8.3f}" for b in range(4)))

    # consecutive transitions (mean +/- sd over subjects)
    print("\n=== Consecutive-transition displacement (mean +/- sd) ===")
    for a, b, lab in [(0, 1, "T1->T2"), (1, 2, "T2->T3"), (2, 3, "T3->T4")]:
        vals = np.array([per_sub[i][a, b] for i in ids])
        print(f"  {lab}: BC = {vals.mean():.3f} +/- {vals.std():.3f}")

    # baseline-drift null vs diet displacement, referenced to T1&T2 baseline
    print("\n=== Resistance test (baseline = mean of T1,T2) ===")
    base_drift = np.array([per_sub[i][0, 1] for i in ids])
    disp3, disp4, resist_ratio = [], [], []
    pair3, pair4 = [], []          # apples-to-apples: pairwise BC to the two baselines
    for i in ids:
        taxa = sorted(genus[i])
        M = matrix_for_subject(genus[i], taxa)
        base = M[:, :2].mean(1)
        d3 = bray_curtis(M[:, 2], base)
        d4 = bray_curtis(M[:, 3], base)
        disp3.append(d3); disp4.append(d4)
        # mean pairwise BC of each post-diet sample to T1 and T2 (same units as
        # the T1<->T2 baseline drift, so this comparison is apples-to-apples)
        pair3.append(0.5 * (per_sub[i][2, 0] + per_sub[i][2, 1]))
        pair4.append(0.5 * (per_sub[i][3, 0] + per_sub[i][3, 1]))
        resist_ratio.append((d3 + d4) / 2 / max(per_sub[i][0, 1], 1e-6))
    disp3, disp4 = np.array(disp3), np.array(disp4)
    pair3, pair4 = np.array(pair3), np.array(pair4)
    print(f"  baseline drift  T1<->T2 (pairwise) : {base_drift.mean():.3f} +/- {base_drift.std():.3f}")
    print(f"  post-diet disp  T3 to centroid     : {disp3.mean():.3f} +/- {disp3.std():.3f}")
    print(f"  post-diet disp  T4 to centroid     : {disp4.mean():.3f} +/- {disp4.std():.3f}")
    print(f"  post-diet disp  T3 pairwise-to-base: {pair3.mean():.3f} +/- {pair3.std():.3f}")
    print(f"  post-diet disp  T4 pairwise-to-base: {pair4.mean():.3f} +/- {pair4.std():.3f}")
    print(f"  (apples-to-apples: pairwise post-diet {0.5*(pair3.mean()+pair4.mean()):.3f} "
          f"vs pairwise baseline drift {base_drift.mean():.3f})")
    print(f"  displacement / baseline-drift ratio: {np.mean(resist_ratio):.2f} "
          f"(1 = diet moves community no more than intrinsic drift)")

    # ---- enterotype / dominant genus stability (basin-switch test) -----
    print("\n=== Dominant-genus (enterotype proxy) stability ===")
    switches = 0
    for i in ids:
        taxa = sorted(genus[i])
        M = matrix_for_subject(genus[i], taxa)
        dom = [taxa[np.argmax(M[:, k])] for k in range(4)]
        changed = len(set(dom)) > 1
        switches += changed
        flag = "  <-- changes" if changed else ""
        print(f"  {i}: {' -> '.join(dom)}{flag}")
    print(f"  subjects whose dominant genus ever changes: {switches}/{len(ids)}")

    # ---- alpha diversity trajectories ----------------------------------
    alpha = load_alpha()
    print("\n=== Alpha diversity (mean +/- sd over subjects) ===")
    print("  time   richness      Shannon      equitability")
    A = {m: np.full((len(ids), 4), np.nan) for m in ("rich", "shan", "equit")}
    for si, i in enumerate(ids):
        for k, t in enumerate(TIMES):
            if i in alpha and t in alpha[i]:
                rich, shan, equit = alpha[i][t]
                A["rich"][si, k], A["shan"][si, k], A["equit"][si, k] = rich, shan, equit
    for k, t in enumerate(TIMES):
        r = np.nanmean(A["rich"][:, k]); rs = np.nanstd(A["rich"][:, k])
        s = np.nanmean(A["shan"][:, k]); ss = np.nanstd(A["shan"][:, k])
        e = np.nanmean(A["equit"][:, k]); es = np.nanstd(A["equit"][:, k])
        print(f"  {t}  {r:6.0f}+/-{rs:3.0f}   {s:5.2f}+/-{ss:4.2f}   {e:5.3f}+/-{es:5.3f}")

    np.savez(os.path.join(OUT, "armonia_metrics.npz"),
             BCpair=BCpair, base_drift=base_drift, disp3=disp3, disp4=disp4,
             pair3=pair3, pair4=pair4,
             ids=ids, rich=A["rich"], shan=A["shan"], equit=A["equit"])
    print(f"\nWrote tidy CSV + metrics to {OUT}")


if __name__ == "__main__":
    main()
