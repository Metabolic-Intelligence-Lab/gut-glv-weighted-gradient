#!/usr/bin/env python3
"""Assemble ``manuscript/submission/``: exactly the files to upload to Editorial Manager.

    python3 scripts/make_submission_folder.py

Run it after the three builds it depends on::

    python3 scripts/make_plos_submission.py     # manuscript.docx, S1_Text.docx (+ reading copies)
    python3 scripts/prepare_plos_figures.py     # figures/Fig*.tif
    pandoc cover_letter.md -o cover_letter.docx # see cover_letter_docx() below

The folder is rebuilt from scratch every time, so nothing stale survives in it. It holds
the upload files under their final names, plus three Markdown files that are *not*
uploaded: ``README.md`` (upload order), ``PORTAL_STATEMENTS.md`` (section 2 of
``SUBMISSION_CHECKLIST.md``, unwrapped so each statement pastes as one paragraph) and
``SUGGESTED_REVIEWERS.md``. The reading copies stay in ``data_paper/``: the point of this
folder is that there is nothing in it you should not upload.

Any bracketed placeholder still present in the manuscript, the cover letter, the S1 Text
or the statements is reported at the end, and the exit status is non-zero so a wrapper
notices; the folder is still written, so you can look at it.
"""
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "manuscript" / "data_paper"
DEST = ROOT / "manuscript" / "submission"
CHECKLIST = ROOT / "manuscript" / "SUBMISSION_CHECKLIST.md"

PLACEHOLDER = re.compile(r"\[(?:GitHub repository URL|committee contact address|[A-Za-z][A-Za-z ]{2,40} (?:URL|DOI|address|placeholder))\]")


def figure_files():
    """Fig1.tif .. FigN.tif in numeric order, checked against the figure count in the .tex."""
    tifs = sorted(SRC.glob("figures/Fig[0-9]*.tif"), key=lambda p: int(re.sub(r"\D", "", p.stem)))
    numbers = [int(re.sub(r"\D", "", p.stem)) for p in tifs]
    if numbers != list(range(1, len(tifs) + 1)):
        sys.exit(f"figure numbering is not 1..N: {numbers}")
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from prepare_plos_figures import main_text_figures
        n_tex = len(main_text_figures())
        if n_tex != len(tifs):
            sys.exit(f"{len(tifs)} TIFFs but the .tex has {n_tex} main-text figures; run prepare_plos_figures.py")
    except ImportError:
        pass
    return tifs


def cover_letter_docx():
    """Rebuild cover_letter.docx from cover_letter.md with real sub/superscripts.

    pandoc drops the raw ``<sub>``/``<sup>`` tags of the Markdown source when writing
    .docx, so they are turned into pandoc's ``~x~`` / ``^x^`` first.
    """
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        print("  pandoc not found: keeping the existing cover_letter.docx")
        return
    md = (SRC / "cover_letter.md").read_text(encoding="utf-8")
    md = re.sub(r"<sub>([^<]*)</sub>", r"~\1~", md)
    md = re.sub(r"<sup>([^<]*)</sup>", r"^\1^", md)
    tmp = SRC / "_cover_letter_pandoc.md"
    tmp.write_text(md, encoding="utf-8")
    try:
        subprocess.run([pandoc, str(tmp), "-f", "markdown", "-t", "docx", "-o", str(SRC / "cover_letter.docx")], check=True)
    finally:
        tmp.unlink()
    print("  rebuilt cover_letter.docx")


def docx_text(path):
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    return re.sub(r"<[^>]+>", "", xml)


def portal_statements():
    """Section 2 of the checklist, blockquotes unwrapped into single paragraphs."""
    text = CHECKLIST.read_text(encoding="utf-8")
    m = re.search(r"^## 2\. Statements to paste into Editorial Manager\n(.*?)(?=^## )", text, re.S | re.M)
    if not m:
        sys.exit("section '## 2. Statements to paste into Editorial Manager' not found in the checklist")
    out, para = [], []
    def flush():
        if para:
            out.append(" ".join(para)); para.clear()
    for line in m.group(1).splitlines():
        if line.startswith(">"):
            body = line[1:].strip()
            if body:
                para.append(body)
            else:
                flush(); out.append("")
        else:
            flush(); out.append(line)
    flush()
    return "# Statements to paste into Editorial Manager\n\nCopied from `SUBMISSION_CHECKLIST.md` §2 by `make_submission_folder.py`; edit there, not here.\n\n" + "\n".join(out).strip() + "\n"


def main():
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir()
    cover_letter_docx()
    tifs = figure_files()
    uploads = [("1", "Cover Letter", SRC / "cover_letter.docx"),
               ("2", "Manuscript", SRC / "manuscript.docx")]
    uploads += [(str(3 + i), f"Figure {i + 1}", p) for i, p in enumerate(tifs)]
    uploads += [(str(3 + len(tifs)), "Supporting Information", SRC / "S1_Text.docx")]
    for _, label, src in uploads:
        if not src.exists():
            sys.exit(f"missing {src.relative_to(ROOT)}: run the build first")
        shutil.copy2(src, DEST / src.name)

    statements = portal_statements()
    (DEST / "PORTAL_STATEMENTS.md").write_text(statements, encoding="utf-8")
    shutil.copy2(SRC / "suggested_reviewers.md", DEST / "SUGGESTED_REVIEWERS.md")

    rows = "\n".join(f"| {n} | {label} | `{src.name}` |" for n, label, src in uploads)
    (DEST / "README.md").write_text(f"""# Upload to Editorial Manager, in this order

Generated by `scripts/make_submission_folder.py` from `manuscript/data_paper/`; do not edit
here, rebuild. Everything with a `.docx` or `.tif` extension in this folder is uploaded;
the `.md` files are for you.

| Order | Editorial Manager item | File |
|---|---|---|
{rows}

`manuscript.docx` carries the figure captions in place and no embedded images, as PLOS
requires; the figures go up as separate TIFFs (RGB, LZW, 300 dpi). The portal fields
(Financial Disclosure, Competing Interests, Data Availability, Ethics Statement, Author
Contributions) are in `PORTAL_STATEMENTS.md`; the reviewer panel is in
`SUGGESTED_REVIEWERS.md`. Portal: <https://www.editorialmanager.com/pcompbiol/>.
""", encoding="utf-8")

    # Placeholder audit over everything that leaves this folder.
    found = {}
    for name in ("manuscript.docx", "cover_letter.docx", "S1_Text.docx"):
        hits = sorted(set(PLACEHOLDER.findall(docx_text(DEST / name))))
        if hits:
            found[name] = hits
    hits = sorted(set(PLACEHOLDER.findall(statements)))
    if hits:
        found["PORTAL_STATEMENTS.md"] = hits

    print(f"  {DEST.relative_to(ROOT)}: {len(uploads)} upload files ({len(tifs)} figures) + 3 notes")
    if found:
        for name, hits in found.items():
            print(f"  PLACEHOLDER in {name}: {', '.join(hits)}")
        print("  not ready to upload until the placeholders above are filled and the package rebuilt")
        sys.exit(1)
    print("  no placeholders left: ready to upload")


if __name__ == "__main__":
    main()
