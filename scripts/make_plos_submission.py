r"""
Build the PLOS Computational Biology submission package from manuscript_PRX.tex
================================================================================

The REVTeX source stays the single point of truth; nothing here edits it. This
script emits, into ``submission_PLOS_CB/``:

  manuscript.docx               PLOS-compliant main text, figure captions only
  manuscript_with_figures.docx  same text with the main figures embedded
  S1_Text.docx                  the appendices, with their figures embedded
  figures/Fig1.tif .. FigN.tif  main figures at PLOS resolution (separate step)

What it rearranges, relative to the REVTeX layout:

* PLOS wants Introduction / Results / Discussion / Materials and methods. The
  REVTeX body sections become Results subsections; "Conclusions" is folded into
  the Discussion; the method paragraphs of Appendix A (inference pipeline,
  symmetrizability diagnostics, constrained fit, cavity derivation, hub scaling,
  normalisation table) are lifted into a real "Materials and methods" section.
* Everything else in the appendices becomes S1 Text. Its floats are relabelled
  in PLOS supporting-information style: "Fig A in S1 Text", "Table A in S1 Text",
  equations "(A1)", sections "Section A in S1 Text".
* \cite is resolved to numbered references and the bibliography is rewritten in
  Vancouver style from the CrossRef cache (scripts/refs_crossref.json).
* The author placeholders (affiliation, funding, ethics, data availability) are
  filled from PLACEHOLDERS below.

Usage:
    python3 scripts/make_plos_submission.py [--pandoc /path/to/pandoc]
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "manuscript_PRX.tex"
REFS = ROOT / "scripts" / "refs_crossref.json"
OUT = ROOT / "manuscript" / "data_paper"
BUILD = ROOT / "manuscript" / "_build"   # pandoc inputs, not part of the upload

# NOTE: the main-text / S1 split is positional -- everything before \appendix in
# manuscript_PRX.tex is main text, everything after is S1 Text (see build()). To
# move a figure between the two, move it across \appendix in the .tex; there is no
# list here to edit. (A MAIN_FIGS constant used to sit here and was dead code.)

# Appendix paragraphs that are genuinely Methods and move to the main text, in the
# order they should appear. Keys are the leading \emph{...} marker. The cavity /
# self-consistency / phase-boundary / hub-scaling / normalisation blocks left with
# the companion theory paper (manuscript_theory.tex).
METHODS_BLOCKS = [
    "Dynamics and inference.",
    "Datasets and preprocessing.",
    "When a symmetrizer exists.",
    "Symmetrizability diagnostics.",
    "Invariance of the diagnostics.",
    "Constrained fit, end-to-end null and predictive comparison.",
    "The descent test.",
    "Specificity of the descent test.",
    "Ethics.",
]

# PLOS headings carry no numbers, so a main-text "Sec.~\ref{}" cannot resolve to
# one: the generated docx used to say "Section 4" with nothing called that in it.
# Section references resolve to the heading text instead. Two headings are too
# long to sit inside a sentence, so they get a short form that still opens with
# the heading's own words and is therefore findable by scanning the headings.
SECTION_REFS = {
    "sec:cohorts": "Tiers 2 and 3",
    "sec:support": "What the data do support",
}

# References CrossRef cannot resolve, written out in Vancouver style. Without an
# entry here (and with none in the cache) a reference renders as a bare ".".
# Empty at present: the companion-theory entry lived here while the paper cited an
# unpublished manuscript, and was dropped with that citation. Add a key here if a
# future reference is again unresolvable, or the bibliography will render a bare ".".
MANUAL_REFS = {}

PLACEHOLDERS = {
    "affiliation": (
        "Dipartimento di Neuroscienze, Sezione di Biofisica, "
        "Universit\\`a Cattolica del Sacro Cuore, Rome, Italy \\\\\n"
        "Fondazione Policlinico Universitario A. Gemelli IRCCS, Rome, Italy"
    ),
    "email": "byome@chirale.it",
    "orcid": "0000-0002-2154-319X",
    # No "ethics" entry: every dataset analysed here comes from an already-published
    # study that carries its own approval, ArMonIA included (Bianchetti et al.,
    # Nutrients 2023), so the manuscript no longer restates one. Editorial Manager's
    # Ethics Statement field is a separate, still-mandatory obligation and its wording
    # lives in manuscript/SUBMISSION_CHECKLIST.md.
    "funding": (
        "This research received no specific grant from any funding agency in the public, "
        "commercial or not-for-profit sectors."
    ),
    # A public repository URL, not an archival DOI: PLOS accepts either, and the code is
    # going to GitHub. Must be filled in and the repository made public BEFORE submitting --
    # PLOS does not accept "code available on request" at any stage.
    "repository": "\\texttt{https://github.com/Metabolic-Intelligence-Lab/gut-glv-weighted-gradient}",
}

# ---------------------------------------------------------------- helpers


def roman(n):
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
            (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


def letters(n):
    """1 -> A, 26 -> Z, 27 -> AA (enough for 25 SI figures)."""
    out = ""
    while n:
        n, r = divmod(n - 1, 26)
        out = chr(ord("A") + r) + out
    return out


def env_span(src, start, name):
    """Return (begin, end) offsets of the \\begin{name}...\\end{name} at `start`."""
    end = src.index("\\end{" + name + "}", start) + len("\\end{" + name + "}")
    return start, end


def find_float(src, label):
    """Locate the figure/table environment carrying \\label{label}."""
    pos = src.index("\\label{" + label + "}")
    best = None
    for name in ("figure*", "figure", "table*", "table"):
        for m in re.finditer(r"\\begin\{" + re.escape(name) + r"\}", src[:pos]):
            try:
                b, e = env_span(src, m.start(), name)
            except ValueError:
                continue
            if b <= pos < e and (best is None or b > best[0]):
                best = (b, e, name)
    if best is None:
        raise KeyError(label)
    return best


# ---------------------------------------------------------------- references


def clean_surnames(authors):
    """CrossRef stores some surnames in caps ("MAY RM"); Vancouver wants "May RM"."""
    out = []
    for a in authors:
        parts = a.split()
        if parts and parts[0].isupper() and len(parts[0]) > 1:
            parts[0] = parts[0].capitalize()
        out.append(" ".join(parts))
    return out


def clean_title(title):
    """Strip publisher cruft: EPL footnote markers, trailing punctuation, inline math."""
    title = re.sub(r"\(a\)\s*$", "", title.strip())            # EPL "...ecosystems(a)"
    title = re.sub(r"\$\\mathit\{([^}]*)\}\$", r"\1", title)   # arXiv "$\mathit{vs}$"
    title = re.sub(r"\$([^$]*)\$", r"\1", title)
    return title.strip().rstrip(".")


def terminate(text):
    """Add the sentence period unless the text already ends in ? or !."""
    return text if text.endswith(("?", "!")) else text + "."


def vancouver(key, rec, tex_abbrev):
    """One PLOS/Vancouver reference line, or the raw bibitem if unresolved."""
    if rec.get("status") == "ARXIV" and rec.get("title"):
        au = clean_surnames(rec["authors"])
        who = ", ".join(au[:6]) + (", et al" if len(au) > 6 else "")
        return (f"{who}. {terminate(clean_title(rec['title']))} "
                f"arXiv:{rec['arxiv']} [Preprint]. {rec['year']}.")
    if rec.get("status") == "OK":
        au = clean_surnames(rec["authors"])
        who = ", ".join(au[:6]) + (", et al" if len(au) > 6 else "")
        title = clean_title(rec["title"])
        jour = tex_abbrev or rec.get("container", "")
        # CrossRef's page field is unreliable for some publishers: missing for
        # Sci Transl Med, and an internal id ("MSB178157") for Mol Syst Biol.
        # The citation as typed in the .tex is the better source when they differ.
        tex_page = (rec.get("parsed") or {}).get("page")
        page = (rec.get("page") or "").replace("--", "-")
        if tex_page and (not page or not page.lower().startswith(tex_page.lower()[:3])):
            page = tex_page
        bits = f"{who}. {terminate(title)} {jour}. {rec['year']}"
        if rec.get("volume"):
            bits += f";{rec['volume']}"
            if page:
                bits += f":{page}"
        elif page:                                  # book chapter: no volume
            bits += f". p. {page}"
        elif not jour:                              # eLife reviewed preprint: no container
            bits = f"{who}. {terminate(title)} eLife. {rec['year']};14:RP105948"
        bits += "."
        if rec.get("doi"):
            bits += f" doi:{rec['doi']}"
        return bits
    if rec.get("status") == "ARXIV":
        raw = re.sub(r"\\emph\{([^}]*)\}", r"\1", rec["bibitem"]).replace("~", " ")
        return raw.rstrip(".") + "."
    raw = re.sub(r"\\emph\{([^}]*)\}", r"\1", rec.get("bibitem", "")).replace("~", " ")
    return raw.rstrip(".") + "."


def tex_journal_abbrev(bibitem):
    """'Nat.\\ Rev.\\ Microbiol.\\ \\textbf{15}' -> 'Nat Rev Microbiol'."""
    m = re.search(r"\\textbf\{\d+\}", bibitem)
    if not m:
        return None
    head = bibitem[:m.start()]
    head = re.sub(r"\\emph\{([^}]*)\}", r"\1", head).replace("~", " ").replace("\\ ", " ")
    head = head.replace("\\", "")
    chunk = head.rstrip().rstrip(",").split(",")[-1].strip()
    return chunk.replace(".", "").strip() or None


# ---------------------------------------------------------------- main build


# ---------------------------------------------------------------- punctuation

# Sites where replacing a single em dash with a comma would splice two
# independent clauses, or blur a definition. Keyed by a distinctive snippet of
# the text that follows the dash.
DASH_COLON = [
    "a diagonal congruence reweights",
    "idealizations and analogies",
    "one-way support, sign discordance",
    "reciprocal support, sign compatibility",
    "a corrected linear-stability edge",
    "a symmetrizable ground truth of matched support",
    "absolute-abundance mono- and pairwise cultures",
    "at matched marginal stability the numerical abscissa",
    "the ecological threshold of a coexisting community",
    "primary polysaccharide degraders feeding fermenters",
    "a weak constraint, not a placement",
    "finding, on the available data",
    "one we ground-truth below",
    "Venturelli's cycle inconsistency and Clark",
]

DASH_SEMICOLON = [
    "so a purely topological centrality",
    "and is discarded",
    "so we do not read this",
    "what is coordinate-invariant is the test",
    "a paired sign test (direction only)",
]


def de_emdash(text):
    """Remove em dashes (LaTeX ---), which the author does not want in the text.

    A pair of dashes inside one sentence is a parenthetical, so it becomes real
    parentheses. A lone dash becomes a comma by default, or a colon/semicolon
    where a comma would splice two independent clauses -- see the lists above.
    Numeric ranges and compound names use -- and are left alone.
    """
    out, i = [], 0
    while True:
        a = text.find("---", i)
        if a < 0:
            out.append(text[i:])
            break
        b = text.find("---", a + 3)
        span = text[a + 3:b] if b > 0 else ""
        # a parenthetical: second dash in the same sentence, nothing sentence-final between
        paired = (0 < b and len(span) < 260
                  and "(" not in span            # nested parentheses read badly
                  and not re.search(r"[.:;]\s+[A-Z(]", span))
        out.append(text[i:a])
        if paired:
            out.append(" (" + span.strip() + ") ")
            i = b + 3
            # the closing parenthesis swallows any space that followed the dash
            while i < len(text) and text[i] == " ":
                i += 1
        else:
            tail = text[a + 3:a + 120].lstrip()
            if any(tail.startswith(k) for k in DASH_COLON):
                out.append(": ")
            elif any(tail.startswith(k) for k in DASH_SEMICOLON):
                out.append("; ")
            else:
                out.append(", ")
            i = a + 3
            while i < len(text) and text[i] == " ":
                i += 1
    text = "".join(out)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r" +([,.;:])", r"\1", text)
    text = re.sub(r"([,;:]) +([,.;:])", r"\2", text)
    text = re.sub(r"  +", " ", text)
    return text


def fill_placeholders(text):
    """Replace the author TODOs that the .tex deliberately left bracketed."""
    subs = [
        ("The author thanks [colleagues] for discussions.",
         "The author thanks the members of the Metabolic Intelligence Lab for discussions."),
        ("This work received\n[funding source and grant number, or ``no specific external funding''].",
         PLACEHOLDERS["funding"]),
        ("[GitHub repository URL]", PLACEHOLDERS["repository"]),
    ]
    for old, new in subs:
        if old not in text:
            print(f"  WARNING: placeholder not found: {old[:48]!r}", file=sys.stderr)
        text = text.replace(old, new)
    text = re.sub(r"% TODO\(author\)[^\n]*\n", "", text)
    return text


def build(src):
    src = src.replace("\r\n", "\n")

    # ---- 1. split body / appendix / back matter
    app_at = src.index("\\appendix")
    # drop the REVTeX preamble/title block: front matter is rebuilt for PLOS
    body = src[src.index("\\section{Introduction}"):app_at]
    rest = src[app_at:]

    ack_m = re.search(r"\\begin\{acknowledgments\}.*?\\end\{acknowledgments\}", rest, re.S)
    back_at = ack_m.start()
    appendix = rest[:back_at]
    back = rest[back_at:]
    bib_at = back.index("\\begin{thebibliography}")
    bibliography = back[bib_at:]
    back = back[:bib_at]

    # ---- 2. lift the Methods paragraphs out of Appendix A
    methods = []
    for marker in METHODS_BLOCKS:
        start = appendix.index("\\emph{" + marker + "}")
        # A block runs to the next \emph{...} paragraph marker at line start, or to
        # the next \section -- without the \section guard the last Methods block of
        # an appendix swallows the whole following appendix.
        nxt = re.search(r"\n(?:\\emph\{[A-Z][^}]*\.\}|\\section\*?\{)", appendix[start + 1:])
        stop = start + 1 + (nxt.start() if nxt else len(appendix) - start - 1)
        block = appendix[start:stop]
        # A lifted block may enclose floats. Tables (the normalisation map) travel
        # with the Methods prose; figures stay behind in S1 Text at this position.
        figenv = re.compile(r"\\begin\{(figure\*?)\}.*?\\end\{\1\}", re.S)
        stranded = figenv.findall(block) and [m.group(0) for m in figenv.finditer(block)] or []
        methods.append(figenv.sub("", block).strip())
        appendix = appendix[:start] + "\n".join(stranded) + appendix[stop:]

    # A section whose entire body was lifted into Methods would otherwise ship as a
    # bare heading in S1 Text.
    appendix = re.sub(r"\\section\{[^}]*\}\s*(?:\\label\{[^}]*\}\s*)?(?=\\section\{)",
                      "", appendix)

    # the code-to-figure table belongs with Data availability, in the main text
    cb, ce, _ = find_float(back, "tab:code")
    code_table = back[cb:ce]
    back = back[:cb] + back[ce:]

    # ---- 3. numbering, shared across both documents
    labels = {}

    def scan(text, dest):
        """Assign numbers to every label in `text` ('main' or 'si')."""
        fig = tab = eq = sec = 0
        token = re.compile(
            r"\\section\*?\{|\\begin\{(equation|align|figure\*?|table\*?)\}|\\label\{([^}]*)\}")
        cur = None
        for m in token.finditer(text):
            t = m.group(0)
            if t.startswith("\\section"):
                sec += 1
                cur = ("sec", letters(sec) if dest == "si" else str(sec))
            elif t.startswith("\\begin"):
                env = m.group(1)
                if env.startswith("figure"):
                    fig += 1
                    cur = ("fig", letters(fig) if dest == "si" else str(fig))
                elif env.startswith("table"):
                    tab += 1
                    cur = ("tab", letters(tab) if dest == "si" else str(tab))
                else:
                    eq += 1
                    cur = ("eq", (letters(1) + str(eq)) if dest == "si" else str(eq))
            elif cur:
                labels[m.group(2)] = (dest, cur[0], cur[1])
        return fig, tab, eq

    # main text = body + methods + back matter (+ the code table)
    main_src = body + "\n\\section{Materials and methods}\n" + "\n\n".join(methods) \
        + "\n" + back + "\n" + code_table
    scan(main_src, "main")
    scan(appendix, "si")

    # heading text for every labelled section, for the cross-references below
    sec_titles = dict(
        (lab, title) for title, lab in
        re.findall(r"\\section\*?\{([^}]*)\}\s*\\label\{([^}]*)\}", src))

    # ---- 4. resolve cross-references
    bibkeys = re.findall(r"\\bibitem\{([^}]*)\}", bibliography)
    raw_bibitems = dict(re.findall(
        r"\\bibitem\{([^}]*)\}(.*?)(?=\\bibitem\{|\\end\{thebibliography\})",
        bibliography, re.S))

    # PLOS/Vancouver numbers references in order of first mention. The \bibitem
    # order in the REVTeX source is thematic, so the numbering is derived here
    # from the text instead: main text first, then anything cited only in S1.
    cite_order = []
    for _text in (main_src, appendix):
        for _m in re.finditer(r"\\cite\{([^}]*)\}", _text):
            for _k in (k.strip() for k in _m.group(1).split(",")):
                if _k and _k not in cite_order:
                    cite_order.append(_k)
    uncited = [k for k in bibkeys if k not in cite_order]
    dangling = [k for k in cite_order if k not in bibkeys]
    bibkeys = [k for k in cite_order if k in bibkeys] + uncited
    citenum = {k: i + 1 for i, k in enumerate(bibkeys)}

    def ref_text(label, kind_hint=None):
        if label not in labels:
            return "??" + label
        dest, kind, num = labels[label]
        if dest == "si":
            noun = {"fig": "Fig", "tab": "Table", "sec": "Section", "eq": "Eq"}[kind]
            if kind == "eq":
                return f"({num})"
            return f"{noun} {num} in S1 Text"
        if kind == "sec":
            title = SECTION_REFS.get(label) or sec_titles.get(label)
            if not title:
                print(f"  WARNING: no title for section label {label!r}", file=sys.stderr)
                return "??" + label
            return "\\emph{" + title + "}"
        return f"({num})" if kind == "eq" else num

    def resolve(text, self_dest):
        # "Appendix~\ref{app:x}" reads as a Section of S1 Text
        text = re.sub(r"(Appendix(?:es)?)~?\\ref\{([^}]*)\}",
                      lambda m: ref_text(m.group(2)), text)
        # "Fig.~\ref{}" / "Figs.~\ref{}" -> PLOS "Fig N" (drop the redundant noun
        # when the target is in the other document and already carries one)
        def figref(m):
            # a trailing letter is a panel: "Fig 3a" in the main text, but it has
            # to go inside the label for a supporting figure -- "Fig S(a) in S1 Text"
            lab, panel = m.group(2), m.group(3) or ""
            out = ref_text(lab)
            if labels.get(lab, ("", ""))[0] == "si":
                return out.replace(" in S1 Text", f"({panel}) in S1 Text") if panel else out
            return ("Figs " if m.group(1).startswith("Figs") else "Fig ") + out + panel
        text = re.sub(r"(Figs?\.?)~?\\ref\{([^}]*)\}([a-h])?(?![a-zA-Z])", figref, text)

        def tabref(m):
            lab = m.group(1)
            out = ref_text(lab)
            return out if labels.get(lab, ("", ""))[0] == "si" else "Table " + out
        text = re.sub(r"Table~?\\ref\{([^}]*)\}", tabref, text)
        text = re.sub(r"(?:Eq\.|Eqs\.)~?\\eqref\{([^}]*)\}",
                      lambda m: "Eq " + ref_text(m.group(1)), text)
        text = re.sub(r"\\eqref\{([^}]*)\}", lambda m: ref_text(m.group(1)), text)
        def secref(m):
            # An S1 target reads "Section A in S1 Text" and keeps its noun; a
            # main-text one resolves to the italicised heading, which supplies its
            # own noun phrase, so "Sec." is dropped along with the \ref.
            return ref_text(m.group(1))
        text = re.sub(r"(?:Sec\.|Secs\.|Section)~?\\ref\{([^}]*)\}", secref, text)
        text = re.sub(r"\\ref\{([^}]*)\}", lambda m: ref_text(m.group(1)), text)
        text = re.sub(r"\\cite\{([^}]*)\}",
                      lambda m: "[" + ",".join(str(citenum.get(k.strip(), "?"))
                                               for k in m.group(1).split(",")) + "]", text)
        return text

    main_src = resolve(main_src, "main")
    si_src = resolve(appendix, "si")

    # ---- 5. pandoc-safe cleanup
    def clean(text):
        text = re.sub(r"\\label\{[^}]*\}", "", text)
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
        text = re.sub(r"\\appendix\s*", "", text)
        text = text.replace("\\begin{acknowledgments}",
                            "\\section*{Acknowledgments}")
        text = text.replace("\\end{acknowledgments}", "")
        return text

    main_src = de_emdash(fill_placeholders(main_src))
    si_src = de_emdash(si_src)
    main_src, si_src = clean(main_src), clean(si_src)

    # ---- 6. references section
    refs = json.loads(REFS.read_text())
    lines = []
    for i, k in enumerate(bibkeys, 1):
        if k in MANUAL_REFS:
            entry = MANUAL_REFS[k]
        else:
            rec = dict(refs.get(k) or {})
            # The cache is keyed by \bibitem name; a key it never resolved has no
            # "bibitem" field either, and used to render as a bare ".".
            if not rec.get("bibitem"):
                rec["bibitem"] = raw_bibitems.get(k, "").strip()
            entry = vancouver(k, rec, tex_journal_abbrev(rec.get("bibitem", "")))
        if len(entry.strip(" .")) < 12:
            print(f"  WARNING: reference {i} ({k}) is empty or near-empty: {entry!r}",
                  file=sys.stderr)
        lines.append(f"{i}. " + entry)
    reference_section = "\\section*{References}\n\n" + "\n\n".join(lines) + "\n"
    for k in uncited:
        print(f"  WARNING: \\bibitem{{{k}}} is never cited", file=sys.stderr)
    for k in dangling:
        print(f"  WARNING: \\cite{{{k}}} has no \\bibitem", file=sys.stderr)

    unresolved = sorted({m for m in re.findall(r"\?\?[A-Za-z:_]+", main_src + si_src)})
    return main_src, si_src, reference_section, labels, unresolved


# ---------------------------------------------------------------- assembly

# NB: this string is emitted as LaTeX. A bare "%" comments out the rest of its
# line and silently truncates the summary in every generated .docx, which is how
# an earlier build shipped "decreases on 86 is therefore compatible". Percent
# signs here must stay escaped.
AUTHOR_SUMMARY = r"""
Physicists borrow the mathematics of disordered magnets to ask why a gut community sometimes
absorbs a disturbance and sometimes flips to a diseased state. That borrowing is exact only if
the community runs downhill on an energy landscape, and the interaction matrices people estimate
from data look far too lopsided for that. We show the estimation itself manufactures most of it:
simulate a system that does run downhill, push it through the same pipeline, and the estimates
come out just as lopsided. We then test the landscape property directly rather than through the
matrix. Running downhill has a consequence checkable on every observation: a particular quantity
can only ever decrease. We check it on one designed laboratory consortium and its twelve
single-species dropouts, fitting on twelve and scoring the thirteenth: that quantity decreases
on 86\% of transitions, against 76\% for scrambled comparators. We also feed the procedure data
from systems built to run downhill and systems built not to: it tells them apart. These
communities are therefore compatible with an energy landscape after all, and the apparent
asymmetry says more about our inference than about the bacteria. We end by naming the
measurements that would settle it.
""".strip()

SECTION_PLAN = [
    # (REVTeX heading, action) -- signature first, artefact second, matching the title.
    ("Introduction", "section"),
    ("The data and the evidence hierarchy", "results-first"),
    ("What a potential requires, and what is invariant", "subsection"),
    ("The trajectories descend a weighted potential", "subsection"),
    ("Why the inferred matrices look asymmetric", "subsection"),
    ("Tiers 2 and 3: what the longitudinal cohorts add, and where they stop", "subsection"),
    ("What the data do support", "subsection"),
    ("Discussion", "section"),
]


def number_equations(text, prefix=""):
    """pandoc drops LaTeX equation numbers; carry them as trailing text."""
    n = [0]

    def tag(m):
        n[0] += 1
        body = m.group(2).rstrip()
        return (f"\\begin{{{m.group(1)}}}{body}\\qquad({prefix}{n[0]})"
                f"\\end{{{m.group(1)}}}")

    return re.sub(r"\\begin\{(equation)\}(.*?)\\end\{equation\}", tag, text, flags=re.S)


def captions_only(text, prefix=""):
    """Replace each figure float by its caption as a bold-led paragraph."""
    n = [0]

    def repl(m):
        n[0] += 1
        cap = re.search(r"\\caption\{(.*)\}\s*$", m.group(2).strip(), re.S)
        body = cap.group(1).strip() if cap else ""
        num = letters(n[0]) if prefix else str(n[0])
        return f"\n\n\\noindent\\textbf{{Fig {num}.}} {body}\n\n"

    return re.sub(r"\\begin\{(figure)\}(.*?)\\end\{figure\}", repl, text, flags=re.S)


def number_float_captions(text, kind, prefix=""):
    """Prefix each float caption with its PLOS label ('Fig 3.' / 'Table A.')."""
    n = [0]
    noun = "Fig" if kind == "figure" else "Table"

    def repl(m):
        n[0] += 1
        num = letters(n[0]) if prefix else str(n[0])
        return m.group(0).replace("\\caption{", "\\caption{\\textbf{%s %s.} " % (noun, num), 1)

    return re.sub(r"\\begin\{%s\}.*?\\end\{%s\}" % (kind, kind), repl, text, flags=re.S)


def restructure(main_src):
    """REVTeX flat sections -> PLOS Introduction/Results/Discussion/Methods."""
    for head, action in SECTION_PLAN:
        pat = "\\section{" + head + "}"
        if pat not in main_src:
            print(f"  WARNING: section not found: {head}", file=sys.stderr)
            continue
        if action == "results-first":
            main_src = main_src.replace(
                pat, "\\section{Results}\n\n\\subsection{" + head + "}", 1)
        elif action == "subsection":
            main_src = main_src.replace(pat, "\\subsection{" + head + "}", 1)
    # Run-in \emph{...} headings become real headings, but at the right depth. Inside
    # Results they sit under a SECTION_PLAN entry that is already a \subsection, so
    # promoting them to \subsection too made fifteen siblings of things that are not
    # siblings ("The symmetric case..." reading as a peer of "The data and the evidence
    # hierarchy"). There they go one level deeper. Under Methods and Discussion, which are
    # \section, \subsection is the correct depth.
    def promote(text, level):
        return re.sub(r"(?m)^\\emph\{([A-Z][^}]*?)\.\}\s*",
                      lambda m: "\\" + level + "{" + m.group(1) + "}\n", text)

    split = main_src.find("\\section{Discussion}")
    if split == -1:
        return promote(main_src, "subsection")
    return promote(main_src[:split], "subsubsection") + promote(main_src[split:], "subsection")


def front_matter(title, abstract):
    return (
        "\\section*{" + title + "}\n\n"
        "\\noindent Giuseppe Maulucci\\textsuperscript{1,2,*}\n\n"
        "\\noindent \\textsuperscript{1} Dipartimento di Neuroscienze, Sezione di Biofisica, "
        "Universit\\`a Cattolica del Sacro Cuore, Rome, Italy\n\n"
        "\\noindent \\textsuperscript{2} Fondazione Policlinico Universitario A. Gemelli IRCCS, "
        "Rome, Italy\n\n"
        "\\noindent \\textsuperscript{*} Corresponding author: "
        + PLACEHOLDERS["email"] + " (ORCID " + PLACEHOLDERS["orcid"] + ")\n\n"
        "\\section*{Abstract}\n\n" + abstract + "\n\n"
        "\\section*{Author summary}\n\n" + AUTHOR_SUMMARY + "\n\n"
    )


TITLE = ("A weighted-gradient signature in designed gut-community dynamics, "
         "and why inferred interaction matrices hide it")

# Trimmed to the PLOS 300-word abstract limit (the REVTeX abstract ran to 317)
# and, per the outstanding review item, "coordinate-invariant" is spelled out as
# invariance under positive species-wise rescaling.
ABSTRACT = r"""
A gut microbial community is often pictured as rolling downhill into a valley: perturb it and
it returns, push it hard enough and it settles somewhere else. The picture makes clinical
questions well posed, but it is exact for a generalized Lotka--Volterra (gLV) community only
under a specific algebraic condition on the interaction matrix, that some positive rescaling
of the species makes it symmetric. Matrices inferred from real communities appear to violate
that condition badly, which would rule the picture out. We show the appearance is an artefact
of the inference, and that the underlying trajectories carry the signature directly. Ranking
eleven datasets by how much of the inverse problem each removes, we find that the two
designed communities with bottom-up matrices do violate the algebraic conditions, but
regenerating trajectories from a model that satisfies them and re-estimating reproduces the
violations just as strongly ($p=0.24$ and $0.18$), while a leave-one-community-out predictive
comparison cannot separate the two kinds of model at a power above $0.28$. We therefore test
the defining property rather than a proxy for it. A community rolling downhill can never
increase its potential $H_D$, a one-sided prediction on every observed transition. Across $78$
held-out transitions from $13$ designed communities, fitting on twelve and scoring the
thirteenth, $H_D$ decreases on $86\%$ of transitions against $76\%$ for potentials of
identical magnitudes with the structure scrambled ($p<2\times10^{-4}$), a margin stable across
regularisation and carried by $11$ of the $13$. Across the longitudinal cohorts the fitted
coupling correlation is low, but only as a weakly identified statistic: simulator-based inversion
excludes no true value in $[0,0.8]$. The available designed-community data are thus positively
consistent with the downhill picture, and we name the measurements that would test it
decisively.
""".strip()

SI_BLURB = r"""
\section*{Supporting information}

\noindent\textbf{S1 Text. Methods, robustness analyses and supporting figures.} Contains the
calibration of the coupling-correlation estimator against synthetic data with a known answer; the
clustered-bootstrap and test-inversion confidence sets; the inference-aware null re-derived on
a composition-preserving simulator; the robustness of the descent test to regularisation and to
individual communities; the two-guild competitive-fold sweep; a worked example of the cycle
condition; the controlled compositionality test; and the keystone-ranking uncertainty and
baselines.
"""

def abstract_words(text):
    """Word count as the submission portal will see it, i.e. on the rendered text.

    Math used to be charged as one word each, which under-counted: pandoc renders
    "$p=0.24$" as "p = 0.24" and "$[0,0.8]$" as "[0, 0.8]", so an abstract that
    measured 299 here arrived at 304 in the .docx. Relations and commas inside math
    are therefore spaced out before counting, the way the renderer spaces them.
    """
    # NB: these are r-strings, so a single backslash in the pattern is a literal
    # backslash in the text. The earlier doubled forms matched "\\" and "\$", i.e.
    # never, and the printed count was wrong.
    def render_math(m):
        s = re.sub(r"\\(le|ge|approx|times|in|pm|sim|gtrsim|lesssim|neq)\b", " = ", m.group(1))
        s = re.sub(r"\\[a-zA-Z]+\*?", "g", s)      # \gamma renders as one visible token
        s = re.sub(r"[_^{}\\!,\s]+", "", s)        # subscripts stay attached to their symbol
        return " " + re.sub(r"([=<>+])", r" \1 ", s) + " "

    text = re.sub(r"\$([^$]*)\$", render_math, text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    return len(re.sub(r"[^A-Za-z0-9'\-%\[\]]+", " ", text).split())


PREAMBLE = (
    "\\documentclass[11pt]{article}\n"
    "\\usepackage{amsmath,amssymb,graphicx,booktabs}\n"
    "\\begin{document}\n"
)


def check_percent(name, text):
    """A mid-line '%' comments out the rest of the line and silently eats prose.

    manuscript_PRX.tex only ever puts comments at the start of a line, so any
    unescaped '%' with text before it on the same line is a bug, not a comment.
    """
    bad = []
    for i, line in enumerate(text.split("\n"), 1):
        for m in re.finditer(r"%", line):
            j = m.start()
            if j == 0 or line[:j].strip() == "":
                break                       # a real, line-leading comment
            if line[j - 1] == "\\":
                continue                    # escaped
            bad.append((i, line.strip()[:90]))
            break
    for i, snippet in bad:
        print(f"  ERROR: unescaped '%' in {name} line {i}: {snippet}", file=sys.stderr)
    return not bad


def pandoc_run(pandoc, tex_path, docx_path, resource_path):
    cmd = [pandoc, str(tex_path), "-o", str(docx_path),
           f"--resource-path={resource_path}", "--wrap=none"]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    print(f"  wrote {docx_path.relative_to(ROOT)} "
          f"({docx_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pandoc", default=shutil.which("pandoc"))
    args = ap.parse_args()

    main_body, si_body, references, labels, unresolved = build(TEX.read_text())
    OUT.mkdir(exist_ok=True)
    BUILD.mkdir(exist_ok=True)

    # --- main text: PLOS section hierarchy, numbered equations, labelled floats
    main_body = restructure(main_body)
    main_body = number_equations(main_body)
    main_body = number_float_captions(main_body, "table")
    # Single source of truth: the abstract is read from the .tex. It used to be
    # maintained here as well, and the two silently diverged (305 words in the .tex,
    # over the PLOS limit, against 284 here). ABSTRACT below is only a fallback.
    _m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", TEX.read_text(), re.S)
    abstract = _m.group(1).strip() if _m else ABSTRACT
    front = de_emdash(front_matter(TITLE, abstract))

    with_figs = front + number_float_captions(main_body, "figure") + SI_BLURB + references
    caption_only = front + captions_only(main_body) + SI_BLURB + references

    # --- S1 Text
    si = ("\\section*{S1 Text}\n\n\\noindent Supporting information for ``" + TITLE
          + "'' (G. Maulucci).\n\n"
          + number_float_captions(
              number_float_captions(number_equations(si_body, prefix="A"), "table", prefix="A"),
              "figure", prefix="A"))
    si = re.sub(r"\\section\{", "\\\\subsection*{", si)

    ok = all([check_percent("main text", caption_only), check_percent("S1 Text", si)])
    if not ok:
        sys.exit("refusing to build: unescaped % would truncate the text")

    (BUILD / "_manuscript_with_figures.tex").write_text(PREAMBLE + with_figs + "\n\\end{document}\n")
    (BUILD / "_manuscript.tex").write_text(PREAMBLE + caption_only + "\n\\end{document}\n")
    (BUILD / "_S1_Text.tex").write_text(PREAMBLE + si + "\n\\end{document}\n")
    # One file to read the work as a whole: main text with its figures, then the full
    # S1 Text, then the shared reference list. Not uploaded.
    reading = with_figs.replace(references, "") + "\n\\newpage\n" + si
    (BUILD / "_reading_copy.tex").write_text(
        PREAMBLE + reading + "\n" + references + "\n\\end{document}\n")

    nfig_main = sum(1 for v in labels.values() if v[0] == "main" and v[1] == "fig")
    nfig_si = sum(1 for v in labels.values() if v[0] == "si" and v[1] == "fig")
    print(f"main figures={nfig_main}  SI figures={nfig_si}  "
          f"abstract words={abstract_words(abstract)}")
    for _label, _text, _limit in (("abstract", abstract, 300),
                                  ("author summary", AUTHOR_SUMMARY, 200)):
        _n = abstract_words(_text)
        if _n > _limit:
            print(f"  WARNING: {_label} is {_n} words, over the PLOS limit of {_limit}",
                  file=sys.stderr)
    print(f"unresolved cross-references: {unresolved or 'none'}")

    if not args.pandoc:
        sys.exit("pandoc not found; pass --pandoc /path/to/pandoc")
    for stem, dest in [("_manuscript", "manuscript.docx"),
                       ("_manuscript_with_figures", "manuscript_with_figures.docx"),
                       ("_S1_Text", "S1_Text.docx"),
                       ("_reading_copy", "READING_COPY_full.docx")]:
        pandoc_run(args.pandoc, BUILD / (stem + ".tex"), OUT / dest, ".:figures")
