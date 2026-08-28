r"""
Resolve every \bibitem of the (retired) manuscript_PRX.tex against CrossRef

RETIRED ALONGSIDE THE .tex. It still runs, reading the frozen source from
git (the .tex was deleted), and rebuilds scripts/refs_crossref.json. The reference
list in the .docx was generated from that cache and is now edited by hand, so
re-running this changes nothing in the manuscripts.
=============================================================

The REVTeX bibliography carries only author/journal/volume/page/year -- PLOS
Computational Biology requires Vancouver style, which needs the *article title*
and the full author list. Rather than write those from memory, this script
queries the CrossRef REST API for each entry and caches the answer in
``scripts/refs_crossref.json``.

Matching is strict on purpose: a candidate is accepted only when the journal,
the volume, the first page *and* the first author's surname all agree with the
bibitem. A looser rule (volume alone) was tried first and produced obvious
false positives -- "Bimetric QED" for Bunin, Phys. Rev. E 95, 042414 -- because
volume numbers collide across journals. Anything that fails the check is cached
as ``"status": "UNMATCHED"`` so it stays visible instead of being silently
guessed; arXiv preprints are not on CrossRef and are flagged ``ARXIV``.

Usage:
    python3 scripts/fetch_ref_metadata.py          # fill gaps
    python3 scripts/fetch_ref_metadata.py --fresh  # ignore the cache
"""
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "_retired_tex" / "manuscript_PRX.tex"
CACHE = ROOT / "scripts" / "refs_crossref.json"
MAILTO = "byome@chirale.it"  # CrossRef polite pool

# Abbreviation -> the container-title CrossRef actually stores.
JOURNAL = {
    "Nat. Rev. Microbiol.": "Nature Reviews Microbiology",
    "Nat. Commun.": "Nature Communications",
    "Nat. Microbiol.": "Nature Microbiology",
    "Proc. Natl. Acad. Sci. USA": "Proceedings of the National Academy of Sciences",
    "Phys. Rev. E": "Physical Review E",
    "Phys. Rev. Lett.": "Physical Review Letters",
    "Europhys. Lett.": "Europhysics Letters",
    "New J. Phys.": "New Journal of Physics",
    "PLoS Comput. Biol.": "PLOS Computational Biology",
    "Mol. Syst. Biol.": "Molecular Systems Biology",
    "Sci. Transl. Med.": "Science Translational Medicine",
    "Front. Microbiol.": "Frontiers in Microbiology",
    "ISME J.": "The ISME Journal",
    "Genome Biol.": "Genome Biology",
    "J. Phys. A: Math. Theor.": "Journal of Physics A: Mathematical and Theoretical",
    "Gut Microbes": "Gut Microbes",
    "Nature": "Nature",
    "Science": "Science",
    "Cell": "Cell",
    "Nutrients": "Nutrients",
    "PRX Life": "PRX Life",
}


def strip_tex(s):
    """Undo the LaTeX accents/markup that appear in this bibliography."""
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    for pat, rep in [
        (r"\\'\{?([a-zA-Z])\}?", r"\1"), (r'\\"\{?([a-zA-Z])\}?', r"\1"),
        (r"\\`\{?([a-zA-Z])\}?", r"\1"), (r"\\\^\{?([a-zA-Z])\}?", r"\1"),
        (r"\\c\{?c\}?", "c"), (r"\\~\{?n\}?", "n"), (r"\\H\{?o\}?", "o"),
    ]:
        s = re.sub(pat, rep, s)
    return s.replace("~", " ").replace("\\ ", " ").replace("\\", "").strip()


def norm(s):
    """Fold to lowercase ASCII letters+digits for tolerant comparison."""
    s = unicodedata.normalize("NFKD", s or "")
    return re.sub(r"[^a-z0-9]", "", s.encode("ascii", "ignore").decode().lower())


def parse_bibitems(src):
    """-> [{key, raw, first_author, journal, volume, page, year, arxiv?}]"""
    out = []
    for m in re.finditer(r"\\bibitem\{([^}]*)\}\s*(.+?)(?=\n\\bibitem|\n\\end\{thebibliography\})",
                         src, re.S):
        key, raw = m.group(1), " ".join(m.group(2).split())
        plain = strip_tex(raw)
        rec = {"key": key, "raw": raw}

        arx = re.search(r"arXiv:(\d{4}\.\d{4,5})", raw)
        if arx:
            rec["arxiv"] = arx.group(1)

        # journal = the comma-separated chunk immediately before the bold volume
        vol_m = re.search(r"\\textbf\{(\d+)\}", raw)
        rec["volume"] = vol_m.group(1) if vol_m else None
        if vol_m:
            head = strip_tex(raw[:vol_m.start()])
            chunk = head.rstrip().rstrip(",").split(",")[-1].strip()
            rec["journal_abbrev"] = chunk
            rec["journal"] = JOURNAL.get(chunk, chunk)
        else:
            rec["journal_abbrev"] = rec["journal"] = None

        pg = re.search(r"\\textbf\{\d+\},\s*([A-Za-z0-9]+)", raw)
        rec["page"] = pg.group(1) if pg else None
        yr = re.search(r"\((\d{4})\)", raw)
        rec["year"] = yr.group(1) if yr else None
        au = re.match(r"^((?:[A-Z]\.[- ]?)+)([A-Z][A-Za-z'-]+)", plain)
        rec["first_author"] = au.group(2) if au else None
        out.append(rec)
    return out


# Entries the bibliographic search cannot pin down, resolved by hand to an exact
# DOI and then fetched from CrossRef like everything else (never typed from
# memory). Reasons, in order: a book chapter with no volume; a Science Transl.
# Med. record that carries no page field; an MSB record whose page is
# "MSB178157" rather than "e8157"; and a Nature Communications record whose
# CrossRef year (2025) disagrees with the year in the .tex (2026).
MANUAL_DOI = {
    "Tap2022": "10.1016/b978-0-12-819265-8.00039-5",
    "Bokulich2016": "10.1126/scitranslmed.aad7121",
    "Venturelli2018": "10.15252/msb.20178157",
    "RiosGarza2026": "10.1038/s41467-025-67036-5",
    # added for referee 9: an eLife reviewed preprint (no volume/page in CrossRef)
    # and a PRX Life paper the bibliographic search does not reach.
    "Pasqualini2025": "10.7554/elife.105948.2",
    "Aguilar2026": "10.1103/8fb1-jmqh",
}


def fetch_arxiv(arxid):
    """arXiv preprints are not on CrossRef; take title and authors from its own API."""
    url = f"http://export.arxiv.org/api/query?id_list={arxid}"
    with urllib.request.urlopen(url, timeout=60) as fh:
        x = fh.read().decode()
    entry = re.search(r"<entry>(.*?)</entry>", x, re.S).group(1)
    title = " ".join(re.search(r"<title>(.*?)</title>", entry, re.S).group(1).split())
    names = re.findall(r"<name>(.*?)</name>", entry)
    year = re.search(r"<published>(\d{4})", entry).group(1)
    authors = []
    for n in names:
        parts = n.split()
        authors.append(f"{parts[-1]} {''.join(w[0] for w in parts[:-1])}")
    return {"status": "ARXIV", "arxiv": arxid, "title": title,
            "authors": authors, "year": year}


def fetch_doi(doi):
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    req = urllib.request.Request(url, headers={"User-Agent": f"ref-check (mailto:{MAILTO})"})
    with urllib.request.urlopen(req, timeout=60) as fh:
        return json.load(fh)["message"]


def query(rec):
    terms = [t for t in (rec["first_author"], rec["journal"], rec["volume"],
                         rec["page"], rec["year"]) if t]
    params = {"query.bibliographic": " ".join(terms), "rows": "20", "mailto": MAILTO}
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": f"ref-check (mailto:{MAILTO})"})
    with urllib.request.urlopen(req, timeout=60) as fh:
        return json.load(fh)["message"]["items"]


def year_of(it):
    for f in ("published-print", "published-online", "issued"):
        parts = it.get(f, {}).get("date-parts", [[None]])
        if parts and parts[0] and parts[0][0]:
            return str(parts[0][0])
    return None


def accept(rec, it):
    """Strict: journal, volume, first page AND the first author must all agree.

    Requiring only one of page/author let through false positives that share a
    volume number -- e.g. Genome Biol. 15 matched "Corset" for David2014, because
    an *substring* surname test makes "David" match "Davidson". Surnames are
    therefore compared as whole tokens and both checks must pass.
    """
    if rec["year"] and year_of(it) != rec["year"]:
        return None
    cont = norm((it.get("container-title") or [""])[0])
    want = norm(rec["journal"])
    if not cont or not want or not (want in cont or cont in want):
        return None
    if not (rec["volume"] and str(it.get("volume") or "") == rec["volume"]):
        return None
    page = str(it.get("page") or "").split("-")[0]
    art = str(it.get("article-number") or "")
    if not (rec["page"] and norm(rec["page"]) in (norm(page), norm(art))):
        return None
    surnames = {norm(a.get("family", "")) for a in it.get("author", [])}
    if not (rec["first_author"] and norm(rec["first_author"]) in surnames):
        return None
    return "journal+volume+page+author"


def main():
    fresh = "--fresh" in sys.argv
    if not TEX.exists():
        raise SystemExit(
            f"{TEX} does not exist. This script parsed the bibliography out of the\n"
            "LaTeX manuscript, which was retired on 2026-08-08 in favour of the .docx.\n"
            "Recover it with:  git show f5f0a3b:manuscript_PRX.tex > " + str(TEX) + "\n"
            "The existing cache scripts/refs_crossref.json is unaffected and still\n"
            "documents how each of the 50 references was resolved.")
    recs = parse_bibitems(TEX.read_text())
    cache = {} if fresh else (json.loads(CACHE.read_text()) if CACHE.exists() else {})
    for i, rec in enumerate(recs, 1):
        cached = cache.get(rec["key"], {})
        if cached.get("status") == "OK" or (cached.get("status") == "ARXIV"
                                            and cached.get("title")):
            continue
        if rec.get("arxiv"):
            try:
                res = fetch_arxiv(rec["arxiv"])
            except Exception as exc:                   # noqa: BLE001
                res = {"status": "ARXIV", "arxiv": rec["arxiv"], "error": str(exc)}
        else:
            try:
                if rec["key"] in MANUAL_DOI:
                    hit, how = fetch_doi(MANUAL_DOI[rec["key"]]), "manual DOI"
                else:
                    hit = next((it for it in query(rec) if accept(rec, it)), None)
                    how = accept(rec, hit) if hit else None
                res = {"status": "UNMATCHED"} if hit is None else {
                    "status": "OK",
                    "title": re.sub(r"<[^>]+>", "", (hit.get("title") or [""])[0]),
                    "authors": [f"{a.get('family','')} "
                                f"{''.join(w[0] for w in a.get('given','').replace('-',' ').split() if w)}".strip()
                                for a in hit.get("author", []) if a.get("family")],
                    "container": (hit.get("container-title") or [""])[0],
                    "volume": hit.get("volume"),
                    "page": hit.get("page") or hit.get("article-number"),
                    "year": year_of(hit), "doi": hit.get("DOI"),
                    "matched_on": how,
                }
            except Exception as exc:                   # noqa: BLE001 - report, don't crash
                res = {"status": "ERROR", "error": str(exc)}
        res["bibitem"] = rec["raw"]
        res["parsed"] = {k: rec.get(k) for k in ("first_author", "journal", "volume", "page", "year")}
        cache[rec["key"]] = res
        print(f"[{i:2d}/{len(recs)}] {rec['key']:<18} {res['status']:<9} {res.get('title','')[:64]}")
        sys.stdout.flush()
        time.sleep(0.3)
    CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    bad = [k for k, v in cache.items() if v["status"] not in ("OK", "ARXIV")]
    print(f"\ncached {len(cache)} entries -> {CACHE}")
    print(f"needs manual entry ({len(bad)}): {bad or 'none'}")


if __name__ == "__main__":
    main()
