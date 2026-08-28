r"""
Assemble the publishable code archive for the PLOS submission
==============================================================

PLOS will not accept "code available on request", so the analysis code has to be public
before submission. This builds a curated tree under a staging directory and zips it to
``manuscript/code_archive.zip``.

Use this rather than publishing the working repository directly. ``data/`` is tracked in
git here, and that includes ``data/armonia/``, whose consent does not cover public
redistribution: pushing the working tree public would contradict the manuscript's Data
availability statement, and deleting the files afterwards would not help, because the git
history would still carry them. Publish this archive, or a fresh repository built from it.

What goes in: every analysis script, the derived ``.npz`` caches that let the
slow figures regenerate without re-running multi-hour sweeps, the rendered
figures, and a DATA.md explaining where each raw dataset comes from.

What stays out, deliberately, by never copying ``data/`` at all:
  * raw third-party datasets -- they belong to their original publications, are
    cited in the manuscript, and are downloadable from the sources in DATA.md;
  * the ArMonIA data -- a published study (Bianchetti et al., Nutrients 15:3931, 2023)
    whose data are available on request as stated there, not for redistribution here.

Besides scripts, caches and figures the archive carries its own README.md (written here,
not copied from the working repository's), DATA.md, LICENSE (MIT, copied from the root),
CITATION.cff and a .gitignore that keeps data/ out of the public repository built from it.

Usage:
    python3 scripts/make_code_archive.py
"""
import datetime
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "manuscript" / "_code_archive_staging"

DATA_MD = """# Data provenance

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
"""


GITIGNORE = """__pycache__/
*.pyc
.DS_Store
data/
"""


def paper_meta():
    """Title, author, ORCID and affiliation, from the submission builder so they cannot drift."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_plos_submission as mps
    aff = re.sub(r"\s*\\\\+\s*", "; ", mps.PLACEHOLDERS["affiliation"])
    aff = aff.replace("\\`a", "à").replace("Universit\\`a", "Università")
    return {"title": mps.TITLE, "orcid": mps.PLACEHOLDERS["orcid"], "affiliation": aff,
            "author": "Giuseppe Maulucci"}


def figure_map():
    """(label, png, script) for every figure of the paper, main text 1..N then S1 A..Z.

    The split is positional on \\appendix, as in make_plos_submission.py. The generating
    script is the one whose source names the png; composites name their panels' scripts.
    """
    tex = (ROOT / "manuscript_PRX.tex").read_text(encoding="utf-8")
    cut = tex.index("\\appendix")
    pat = r"\\includegraphics\[[^\]]*\]\{figures/([^}]+?\.png)\}"
    main = re.findall(pat, tex[:cut]); si = re.findall(pat, tex[cut:])
    sources = {p.name: p.read_text(encoding="utf-8", errors="replace") for p in (ROOT / "scripts").glob("*.py")}
    def who(png):
        stem = png[:-4]
        hits = sorted(n for n, src in sources.items() if png in src or f'"{stem}"' in src)
        return ", ".join(hits) if hits else "(see the code-to-figure table in the paper)"
    rows = [(f"Fig {i}", png, who(png)) for i, png in enumerate(main, 1)]
    rows += [(f"Fig {chr(65 + i)} (S1)", png, who(png)) for i, png in enumerate(si)]
    return rows


def readme_text(meta, rows):
    table = "\n".join(f"| {lab} | `figures/{png}` | `{scr}` |" for lab, png, scr in rows)
    return f"""# Code for "{meta['title']}"

{meta['author']} (ORCID {meta['orcid']}), {meta['affiliation']}.
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
{table}
"""


def citation_cff(meta):
    today = datetime.date.today().isoformat()
    return f"""cff-version: 1.2.0
message: "If you use this code, please cite the paper below."
title: "Code for: {meta['title']}"
authors:
  - family-names: Maulucci
    given-names: Giuseppe
    orcid: "https://orcid.org/{meta['orcid']}"
    affiliation: "{meta['affiliation']}"
license: MIT
date-released: "{today}"
version: "1.0.0"
preferred-citation:
  type: article
  title: "{meta['title']}"
  authors:
    - family-names: Maulucci
      given-names: Giuseppe
      orcid: "https://orcid.org/{meta['orcid']}"
  journal: "PLOS Computational Biology"
  year: 2026
  status: submitted
"""


def main():
    if DEST.exists():
        shutil.rmtree(DEST)
    (DEST / "scripts").mkdir(parents=True)
    (DEST / "figures").mkdir(parents=True)

    n_py = 0
    for src in sorted((ROOT / "scripts").glob("*.py")):
        shutil.copy2(src, DEST / "scripts" / src.name)
        n_py += 1
    for extra in ("polish.matplotlibrc", "refs_crossref.json"):
        src = ROOT / "scripts" / extra
        if src.exists():
            shutil.copy2(src, DEST / "scripts" / extra)

    n_fig = 0
    for pattern in ("*.png", "*.npz"):
        for src in sorted((ROOT / "figures").glob(pattern)):
            shutil.copy2(src, DEST / "figures" / src.name)
            n_fig += 1

    meta = paper_meta()
    rows = figure_map()
    (DEST / "README.md").write_text(readme_text(meta, rows), encoding="utf-8")
    (DEST / "DATA.md").write_text(DATA_MD, encoding="utf-8")
    (DEST / "CITATION.cff").write_text(citation_cff(meta), encoding="utf-8")
    (DEST / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    shutil.copy2(ROOT / "LICENSE", DEST / "LICENSE")
    unmapped = [png for _, png, scr in rows if scr.startswith("(")]
    if unmapped:
        print("  no generating script found for:", ", ".join(unmapped))

    # Entries live under a "code_archive/" root inside the zip regardless of what the
    # staging directory is called, so the reviewer unpacks one tidy folder.
    zip_path = DEST.parent / "code_archive.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(DEST.rglob("*")):
            if path.is_file():
                zf.write(path, Path("code_archive") / path.relative_to(DEST))

    shutil.rmtree(DEST)   # the zip is the deliverable; do not leave a 9 MB duplicate

    print(f"{n_py} scripts, {n_fig} figure files, {len(rows)} figures mapped to scripts")
    print(f"zip: {zip_path.relative_to(ROOT)} ({zip_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
