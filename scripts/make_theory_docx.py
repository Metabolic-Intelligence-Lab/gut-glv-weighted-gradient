r"""
Build a readable .docx of the companion theory manuscript
=========================================================

`manuscript_theory.tex` has no journal target yet, so this produces one plain
reading copy with every figure embedded, numbered sequentially, and the references
resolved in Vancouver style from the CrossRef cache:

    submission_theory/manuscript_theory.docx

It reuses the helpers of make_plos_submission.py rather than duplicating them; the
only thing it does differently is that everything is one document (no main/SI
split), so figures, tables, equations and sections are numbered 1, 2, 3, ...

Usage:
    python3 scripts/make_theory_docx.py [--pandoc /path/to/pandoc]
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_plos_submission import (           # noqa: E402
    de_emdash, pandoc_run, tex_journal_abbrev, vancouver,
)

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "manuscript_theory.tex"
REFS = ROOT / "scripts" / "refs_crossref.json"
OUT = ROOT / "manuscript" / "theory_paper"
BUILD = ROOT / "manuscript" / "_build"


def clean(text):
    """Strip the REVTeX-isms pandoc cannot read. Mirrors make_plos_submission."""
    text = text.replace("\\begin{ruledtabular}", "").replace("\\end{ruledtabular}", "")
    text = text.replace("\\colrule", "\\hline")
    text = re.sub(r"\\newcommand\{\\[nr]vec\}\{[^\n]*\}\s*\n", "", text)
    text = text.replace("\\nvec", "\\mathbf{n}").replace("\\rvec", "\\mathbf{r}")
    text = re.sub(r"\\bm\{([^{}]+)\}", r"\\mathbf{\1}", text)
    for a, b in [("rm", "mathrm"), ("bf", "mathbf"), ("it", "mathit"),
                 ("sf", "mathsf"), ("tt", "mathtt")]:
        text = re.sub(r"\{\\%s\s+([^{}]+)\}" % a, r"\\%s{\1}" % b, text)
    text = re.sub(r"\\(begin|end)\{figure\*\}", r"\\\1{figure}", text)
    text = re.sub(r"\\(begin|end)\{table\*\}", r"\\\1{table}", text)
    return text


def number_everything(src):
    """Assign 1,2,3,... to every float/equation/section label, in reading order."""
    labels = {}
    fig = tab = eq = sec = 0
    token = re.compile(
        r"\\section\*?\{|\\begin\{(equation|figure\*?|table\*?)\}|\\label\{([^}]*)\}")
    cur = None
    for m in token.finditer(src):
        env, lab = m.group(1), m.group(2)
        if m.group(0).startswith("\\section"):
            sec += 1
            cur = ("sec", sec)
        elif env:
            if env.startswith("figure"):
                fig += 1
                cur = ("fig", fig)
            elif env.startswith("table"):
                tab += 1
                cur = ("tab", tab)
            else:
                eq += 1
                cur = ("eq", eq)
        elif lab and cur:
            labels[lab] = cur
    return labels


def resolve(text, labels, citenum):
    """\\ref/\\eqref/\\cite -> numbers, keeping the leading noun readable."""
    def num(lab):
        return str(labels[lab][1]) if lab in labels else "??" + lab

    # "Appendix~\ref{app:cavity}" -> "Appendix 1" reads oddly in a single document;
    # this manuscript has exactly one appendix, so name it.
    text = re.sub(r"Appendix(?:es)?~?\\ref\{[^}]*\}", "the Appendix", text)
    text = re.sub(r"(Figs?\.?)~?\\ref\{([^}]*)\}([a-h])?(?![a-zA-Z])",
                  lambda m: ("Figs " if m.group(1).startswith("Figs") else "Fig ")
                  + num(m.group(2)) + (m.group(3) or ""), text)
    text = re.sub(r"Table~?\\ref\{([^}]*)\}", lambda m: "Table " + num(m.group(1)), text)
    text = re.sub(r"(?:Secs?\.|Section)~?\\ref\{([^}]*)\}",
                  lambda m: "Section " + num(m.group(1)), text)
    text = re.sub(r"(?:Eqs?\.)~?\\eqref\{([^}]*)\}",
                  lambda m: "Eq " + num(m.group(1)), text)
    text = re.sub(r"\\eqref\{([^}]*)\}", lambda m: "(" + num(m.group(1)) + ")", text)
    text = re.sub(r"\\ref\{([^}]*)\}", lambda m: num(m.group(1)), text)
    text = re.sub(r"\\cite\{([^}]*)\}",
                  lambda m: "[" + ",".join(str(citenum.get(k.strip(), "?"))
                                           for k in m.group(1).split(",")) + "]", text)
    return text


def number_equations(text):
    n = [0]

    def tag(m):
        n[0] += 1
        return f"\\begin{{equation}}{m.group(1).rstrip()}\\qquad({n[0]})\\end{{equation}}"
    return re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", tag, text, flags=re.S)


def number_captions(text, kind):
    n = [0]
    noun = "Fig" if kind == "figure" else "Table"

    def repl(m):
        n[0] += 1
        return m.group(0).replace("\\caption{",
                                  "\\caption{\\textbf{%s %d.} " % (noun, n[0]), 1)
    return re.sub(r"\\begin\{%s\}.*?\\end\{%s\}" % (kind, kind), repl, text, flags=re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pandoc", default=shutil.which("pandoc"))
    args = ap.parse_args()
    if not args.pandoc:
        sys.exit("pandoc not found; pass --pandoc /path/to/pandoc")

    src = TEX.read_text(encoding="utf-8")
    body = src[src.index("\\title{"):src.index("\\begin{thebibliography}")]
    bib_src = src[src.index("\\begin{thebibliography}"):]

    # --- references, in \bibitem order, rewritten from the CrossRef cache
    cache = json.loads(REFS.read_text(encoding="utf-8"))
    keys = re.findall(r"\\bibitem\{([^}]*)\}", bib_src)
    citenum = {k: i + 1 for i, k in enumerate(keys)}
    items = re.findall(r"\\bibitem\{[^}]*\}(.*?)(?=\\bibitem\{|\\end\{thebibliography\})",
                       bib_src, re.S)
    lines, unresolved = [], []
    for i, (k, raw) in enumerate(zip(keys, items), 1):
        rec = cache.get(k)
        if rec:
            rec = dict(rec, bibitem=raw)
            line = vancouver(k, rec, tex_journal_abbrev(raw))
        else:                       # e.g. the companion-paper self-citation
            unresolved.append(k)
            line = re.sub(r"\\emph\{([^}]*)\}", r"\1", raw).replace("~", " ").strip()
        line = re.sub(r"\s+", " ", line).strip()
        lines.append(f"{i}. {line}\n")
    references = "\\section*{References}\n\n" + "\n".join(lines)

    # --- title block -> plain headings pandoc can style
    title = re.search(r"\\title\{(.*?)\}\s*\n\s*\\author", body, re.S).group(1)
    title = title.replace("\\\\", " ").strip()
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S).group(1)
    abstract = re.sub(r"(?m)^%.*$", "", abstract).strip()
    body = body[body.index("\\maketitle") + len("\\maketitle"):]

    front = (f"\\section*{{{title}}}\n\n"
             "\\noindent Giuseppe Maulucci\n\n"
             "\\noindent Dipartimento di Neuroscienze, Sezione di Biofisica, "
             "Universit\\`a Cattolica del Sacro Cuore, Rome, Italy\n\n"
             "\\noindent Fondazione Policlinico Universitario A. Gemelli IRCCS, Rome, Italy\n\n"
             "\\section*{Abstract}\n\n" + abstract + "\n\n")

    labels = number_everything(body)
    body = clean(body)
    body = re.sub(r"\\appendix\s*", "\\\\section*{Appendix}\n\n", body)
    body = resolve(body, labels, citenum)
    body = number_equations(body)
    body = number_captions(body, "figure")
    body = number_captions(body, "table")
    body = re.sub(r"\\label\{[^}]*\}", "", body)
    body = de_emdash(body)
    # run-in bold-italic headings become real headings, as in the PLOS build
    body = re.sub(r"(?m)^\\emph\{([A-Z][^}]*?)\.\}\s*",
                  lambda m: "\\subsection*{" + m.group(1) + "}\n", body)

    doc = ("\\documentclass[11pt]{article}\n"
           "\\usepackage{amsmath,amssymb,graphicx,booktabs}\n"
           "\\begin{document}\n" + front + body + "\n" + references + "\n\\end{document}\n")

    BUILD.mkdir(parents=True, exist_ok=True)
    tex_out = BUILD / "_theory.tex"
    tex_out.write_text(doc, encoding="utf-8")

    todos = len(re.findall(r"(?m)^%\s*TODO", doc)) + doc.count("TODO(author)")
    dangling = sorted(set(re.findall(r"\?\?[A-Za-z:_]+", doc)))
    print(f"figures={sum(1 for v in labels.values() if v[0]=='fig')}  "
          f"tables={sum(1 for v in labels.values() if v[0]=='tab')}  "
          f"equations={sum(1 for v in labels.values() if v[0]=='eq')}  "
          f"references={len(keys)}")
    if unresolved:
        print(f"  references kept verbatim (not in CrossRef cache): {', '.join(unresolved)}")
    if dangling:
        print(f"  WARNING: unresolved cross-references: {dangling}", file=sys.stderr)
    if todos:
        print(f"  NOTE: {todos} author TODO markers still in the source "
              f"(Intro, Discussion and Conclusions are stubs)", file=sys.stderr)

    pandoc_run(args.pandoc, tex_out, OUT / "manuscript_theory.docx", ".:figures")


if __name__ == "__main__":
    main()
