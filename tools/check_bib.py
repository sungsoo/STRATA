#!/usr/bin/env python3
"""Check refs.bib against Crossref / OpenAlex / DBLP records before submission (R46, R50).

Why: a wrong coauthor name in refs.bib ("Young-Keun" for "Young-Kuk") was found by
hand in v15, and no gate looked at the bibliography.

How:
  * Entries with a DOI are compared with the Crossref record for that DOI (title with
    subtitle): author family names, given names, author order and count, title, year,
    volume, and pages.
  * Entries without a DOI are searched by title (Crossref, then OpenAlex, then DBLP).
    A record whose title matches is compared the same way and reported as a
    *candidate*; the user confirms it by adding its DOI to refs.bib.
  * Published spellings pass ("Sung-Soo" = "Sungsoo"); given names are compared as
    letters (one may be a prefix), or by initial when either side is abbreviated.
  * Every fetched record is stored in reports/bib_cache.json, so later runs compare
    offline. Without --fetch the tool never touches the network.
  * reports/bib_manual.json may hold checks made by hand (source URL, date, note),
    shown in the report, and "accept" rules for known errors in a record itself
    (field, record substring, reason).

What: reports/bib_check.md lists every entry as match / accepted / check / mismatch /
unverified. "check" means a title-search record differs (perhaps another work or
version). Exit code 1 on any mismatch. Optional check, not a gate (needs network once):
    python3 tools/check_bib.py --fetch        # or: python3 tools/run_gates.py --bib
"""
from __future__ import annotations

import argparse
import datetime as _dt
import difflib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
VERSION_ROOT = CODE.parent.parent
PAPER = VERSION_ROOT / "STRATA-latex" / "STRATA-paper-EN"
REPORTS = VERSION_ROOT / "reports"
BIB = PAPER / "refs.bib"
TEX = (PAPER / "main.tex", PAPER / "supplementary.tex")
CACHE = REPORTS / "bib_cache.json"
MANUAL = REPORTS / "bib_manual.json"
REPORT = REPORTS / "bib_check.md"

TITLE_MATCH = 0.90          # SequenceMatcher ratio on normalized titles
PAUSE = 0.2                 # seconds between requests
USER_AGENT = "STRATA-check_bib/1.0 (reference check before submission)"

# ----------------------------------------------------------------------------- parsing


def _braced(text, i):
    """Return (content, index after) for a {...} group starting at text[i] == '{'."""
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced braces in bib")


def parse_bib(text):
    """Parse @type{key, field={...}, ...} entries into {key: {"type":..., field: value}}."""
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        kind = m.group(1).lower()
        if kind in ("comment", "string", "preamble"):
            continue
        body, _ = _braced(text, m.start() + m.group(0).index("{"))
        body = body[body.index(",") + 1:]
        fields, i = {"type": kind}, 0
        while True:
            f = re.compile(r"\s*,?\s*(\w+)\s*=\s*").match(body, i)
            if not f:
                break
            name, i = f.group(1).lower(), f.end()
            if i < len(body) and body[i] == "{":
                value, i = _braced(body, i)
            elif i < len(body) and body[i] == '"':
                j = body.index('"', i + 1)
                value, i = body[i + 1:j], j + 1
            else:
                v = re.compile(r"[^,\s}]+").match(body, i)
                value, i = (v.group(0), v.end()) if v else ("", i)
            fields[name] = " ".join(value.split())
        entries[m.group(2)] = fields
    return entries


def tex_files(roots=TEX):
    """The documents and every file they \\input or \\include, recursively (v17: a
    table file cites two references that a main.tex-only scan missed)."""
    seen, todo = [], [Path(p) for p in roots]
    while todo:
        p = todo.pop(0)
        if p in seen or not p.exists():
            continue
        seen.append(p)
        text = re.sub(r"(?<!\\)%.*", "", p.read_text(encoding="utf-8"))
        for name in re.findall(r"\\(?:input|include)\s*\{([^}]+)\}", text):
            child = p.parent / name.strip()
            todo.append(child if child.suffix == ".tex" else child.with_name(child.name + ".tex"))
    return seen


def cited_keys(paths=TEX):
    keys = set()
    for p in tex_files(paths):
        text = re.sub(r"(?<!\\)%.*", "", p.read_text(encoding="utf-8"))
        for group in re.findall(r"\\(?:no)?cite\w*\*?(?:\[[^\]]*\])?\{([^}]+)\}", text):
            keys |= {k.strip() for k in group.split(",") if k.strip()}
    return keys

def unused_keys(bib=BIB, tex=TEX):
    """refs.bib keys that neither main.tex nor supplementary.tex cites (R47)."""
    entries = parse_bib(Path(bib).read_text(encoding="utf-8"))
    cited = cited_keys(tex)
    return sorted(k for k in entries if k not in cited)


# ----------------------------------------------------------------------------- normalizing

_SPECIAL = str.maketrans({"ı": "i", "ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d",
                          "ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe"})
_ACCENT = re.compile(r"\\[`'^\"~=.]\s*\{?\s*\\?([A-Za-z])\}?"
                     r"|\\[uvHtcdbk](?:\s+|\{)\s*\\?([A-Za-z])\}?")
_LETTER = re.compile(r"\\(ss|ae|AE|oe|OE|aa|AA|o|O|l|L|i|j)(?![A-Za-z])\s*")


def plain(s):
    """LaTeX / Unicode text -> lowercase ASCII letters, digits and single spaces."""
    s = _ACCENT.sub(lambda m: m.group(1) or m.group(2), s or "")
    s = _LETTER.sub(lambda m: m.group(1), s)
    s = re.sub(r"\$[^$]*\$", lambda m: re.sub(r"[^A-Za-z0-9]", "", m.group(0)), s)
    s = re.sub(r"\\[A-Za-z]+\s*", "", s)              # \textit{x} -> {x}; keep the argument
    s = s.replace("{", "").replace("}", "")
    s = s.translate(_SPECIAL)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).lower()
    return " ".join(s.split())


def split_authors(field):
    """BibTeX author field -> ([{"given":, "family":}], truncated)."""
    names, truncated = [], False
    for raw in re.split(r"\s+and\s+", field or ""):
        raw = raw.strip().strip("{}")
        if not raw:
            continue
        if raw.lower() == "others":
            truncated = True
            continue
        if "," in raw:
            family, given = (x.strip() for x in raw.split(",", 1))
        else:
            parts = raw.split()
            family, given = parts[-1], " ".join(parts[:-1])
        names.append({"given": given, "family": family})
    return names, truncated


PARTICLES = {"von", "van", "de", "der", "den", "di", "da", "du", "le", "la", "del", "della",
             "dos", "das", "ter", "ten", "zu", "vom"}


def family_key(name):
    """Family name as letters only, without leading particles ("von Niederhausern" and
    Crossref's "Niederhausern" with "Luc Von" as given name agree)."""
    words = plain(name.get("family", "")).split()
    while len(words) > 1 and words[0] in PARTICLES:
        words = words[1:]
    return "".join(words)


def same_given(a, b):
    """Given names agree: by initials if either is abbreviated ("M. Tamer"), otherwise
    as letters only, one may be a prefix ("Sung-Soo" = "Sungsoo", "Ben" ~ "Benjamin",
    but "Young-Kuk" != "Young-Keun")."""
    ta, tb = plain(a).split(), plain(b).split()
    if not ta or not tb:
        return True
    if any(len(t) == 1 for t in ta + tb):
        return ta[0][0] == tb[0][0]
    ja, jb = "".join(ta), "".join(tb)
    return ja.startswith(jb) or jb.startswith(ja)


def first_page(p):
    m = re.match(r"\s*([A-Za-z]?\d+)", str(p or ""))
    return m.group(1).lower() if m else ""


def last_page(p):
    m = re.search(r"[-\u2013]+\s*([A-Za-z]?\d+)\s*$", str(p or ""))
    return m.group(1).lower() if m else ""

# ----------------------------------------------------------------------------- comparing


def compare(entry, record):
    """List of (field, bib value, record value) that disagree; empty means match."""
    issues = []
    bib_auth, truncated = split_authors(entry.get("author", ""))
    rec_auth = record.get("authors") or []
    if bib_auth and rec_auth:
        n = len(bib_auth)
        if (not truncated and n != len(rec_auth)) or (truncated and n > len(rec_auth)):
            issues.append(("author count", str(n) + ("+others" if truncated else ""),
                           str(len(rec_auth))))
        bib_f = [family_key(a) for a in bib_auth]
        rec_f = [family_key(a) for a in rec_auth[:n]]
        if bib_f != rec_f:
            all_rec = [family_key(a) for a in rec_auth]
            if sorted(bib_f) == sorted(all_rec[:n]) or sorted(bib_f) == sorted(all_rec):
                issues.append(("author order", ", ".join(bib_f), ", ".join(all_rec)))
            else:
                issues.append(("author names", ", ".join(bib_f), ", ".join(all_rec)))
        else:
            for a, r in zip(bib_auth, rec_auth):
                if not same_given(a.get("given", ""), r.get("given", "")):
                    issues.append(("given name", f"{a['given']} {a['family']}",
                                   f"{r['given']} {r['family']}"))
    tb, tr = plain(entry.get("title")), plain(record.get("title"))
    if tb and tr and difflib.SequenceMatcher(None, tb, tr).ratio() < TITLE_MATCH:
        issues.append(("title", entry.get("title", ""), record.get("title", "")))
    years = [int(y) for y in record.get("years") or []]
    if entry.get("year") and years and int(re.sub(r"\D", "", entry["year"]) or 0) not in years:
        issues.append(("year", entry["year"], "/".join(map(str, sorted(set(years))))))
    if entry.get("volume") and record.get("volume") and \
            plain(entry["volume"]) != plain(record["volume"]):
        issues.append(("volume", entry["volume"], record["volume"]))
    if entry.get("pages") and record.get("page"):
        fb, fr = first_page(entry["pages"]), first_page(record["page"])
        lb, lr = last_page(entry["pages"]), last_page(record["page"])
        if (fb and fr and fb != fr) or (lb and lr and lb != lr):
            issues.append(("pages", entry["pages"], record["page"]))
    return issues

# ----------------------------------------------------------------------------- fetching


def _get_json(url, timeout=20):
    time.sleep(PAUSE)                                    # polite to Crossref / DBLP
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _years(msg):
    out = []
    for k in ("issued", "published-print", "published-online", "published"):
        parts = (msg.get(k) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            out.append(int(parts[0][0]))
    return sorted(set(out))


def crossref_record(msg, source):
    return {"source": source,
            "doi": msg.get("DOI", ""),
            "title": " ".join(": ".join(x for x in ((msg.get("title") or [""])[0],
                                                   (msg.get("subtitle") or [""])[0]) if x).split()),
            "authors": [{"given": a.get("given", ""), "family": a.get("family", a.get("name", ""))}
                        for a in msg.get("author") or []],
            "years": _years(msg),
            "volume": msg.get("volume", ""),
            "page": msg.get("page", "") or msg.get("article-number", ""),
            "venue": " ".join((msg.get("container-title") or [""])[0].split())}


def dblp_record(info):
    authors = (info.get("authors") or {}).get("author") or []
    if isinstance(authors, dict):
        authors = [authors]
    names = []
    for a in authors:
        text = re.sub(r"\s+\d{4}$", "", a.get("text", "") if isinstance(a, dict) else str(a))
        parts = text.split()
        names.append({"given": " ".join(parts[:-1]), "family": parts[-1] if parts else ""})
    year = info.get("year")
    return {"source": "dblp", "doi": info.get("doi", ""),
            "title": info.get("title", "").rstrip("."), "authors": names,
            "years": [int(year)] if year else [], "volume": info.get("volume", ""),
            "page": info.get("pages", ""), "venue": info.get("venue", "")}


class Incomplete(Exception):
    """One source failed for this entry (bad JSON, rate limit, bot page); try again later."""


def best_candidate(entry, records):
    """The record for the same work: titles agree (ratio >= TITLE_MATCH) with the same
    numbers ("Algorithm 1000" is not "Algorithm 1037"); among those, the one whose year
    matches refs.bib (a journal version and its conference version share a title)."""
    title = plain(entry.get("title"))
    digits = re.findall(r"\d+", title)
    year = int(re.sub(r"\D", "", entry.get("year", "")) or 0)
    best, best_key = None, None
    for rec in records:
        t = plain(rec["title"])
        ratio = difflib.SequenceMatcher(None, title, t).ratio()
        if ratio < TITLE_MATCH or re.findall(r"\d+", t) != digits:
            continue
        key = (year in (rec.get("years") or []), ratio)
        if best_key is None or key > best_key:
            best, best_key = rec, key
    return best


def openalex_record(work):
    """OpenAlex work -> record (display names are "Given Family")."""
    names = []
    for a in work.get("authorships") or []:
        parts = ((a.get("author") or {}).get("display_name") or "").split()
        names.append({"given": " ".join(parts[:-1]), "family": parts[-1] if parts else ""})
    b = work.get("biblio") or {}
    page = "-".join(x for x in (b.get("first_page"), b.get("last_page")) if x)
    year = work.get("publication_year")
    venue = ((work.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    return {"source": "openalex", "doi": doi, "title": work.get("display_name") or work.get("title") or "",
            "authors": names, "years": [int(year)] if year else [], "volume": b.get("volume") or "",
            "page": page, "venue": venue}


def lookup(entry, get_json=_get_json):
    """Return a record dict (or None) for one entry. Connection errors propagate (the
    caller stops fetching); a source that answers with something unusable raises
    Incomplete for this entry only."""
    doi = entry.get("doi", "").strip()
    if doi:
        try:
            msg = get_json("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe=""))
        except ValueError as exc:                        # not JSON
            raise Incomplete(f"Crossref: {type(exc).__name__}") from exc
        return crossref_record(msg["message"], "crossref-doi")
    if not plain(entry.get("title")):
        return None
    first = split_authors(entry.get("author", ""))[0][:1]
    q = entry.get("title", "") + (" " + first[0]["family"] if first else "")
    try:
        msg = get_json("https://api.crossref.org/works?rows=5&query.bibliographic="
                       + urllib.parse.quote(plain(q)))
    except (ValueError, urllib.error.HTTPError) as exc:
        raise Incomplete(f"Crossref: {type(exc).__name__}") from exc
    rec = best_candidate(entry, [crossref_record(i, "crossref-title")
                                 for i in msg.get("message", {}).get("items", [])])
    if rec:
        return rec
    try:                                                 # v19 (R50): DBLP blocks scripts
        msg = get_json("https://api.openalex.org/works?per-page=5&search="
                       + urllib.parse.quote(plain(entry.get("title"))))
        rec = best_candidate(entry, [openalex_record(w) for w in msg.get("results") or []])
        if rec:
            return rec
    except (ValueError, urllib.error.HTTPError):
        pass                                             # fall through to DBLP
    try:
        msg = get_json("https://dblp.org/search/publ/api?format=json&h=5&q="
                       + urllib.parse.quote(plain(entry.get("title"))))
    except (ValueError, urllib.error.HTTPError) as exc:   # DBLP bot protection / rate limit
        raise Incomplete(f"DBLP: {type(exc).__name__}") from exc
    hits = ((msg.get("result") or {}).get("hits") or {}).get("hit") or []
    return best_candidate(entry, [dblp_record(h.get("info") or {}) for h in hits])


SELECTOR = 3   # v17: same numbers / bib year; v19: Crossref subtitles, OpenAlex


def query_id(entry):
    """What the cached record answers; a changed DOI or title invalidates the cache."""
    return ("doi:" + entry["doi"].strip().lower()) if entry.get("doi") \
        else ("title:" + plain(entry.get("title")))


def cache_valid(hit, qid):
    """A cached answer still holds: same query, and a title answer made by the current
    candidate rule (a DOI answer does not depend on it)."""
    if not hit or hit.get("query") != qid:
        return False
    return hit.get("selector", 1) >= SELECTOR          # v19: DOI answers too (subtitles)

# ----------------------------------------------------------------------------- running


def is_accepted(issue, rules):
    """bib_manual.json "accept": [{"field": ..., "record": substring, "reason": ...}] marks a
    known error in the record itself (e.g. a misspelled name in Crossref)."""
    field, _, record_value = issue
    return any(r.get("field") == field and r.get("record", "") in record_value and r.get("reason")
               for r in rules or [])


STATUSES = ("match", "accepted", "check", "mismatch", "unverified")


def check(entries, keys, cache, manual=None, fetch=False, get_json=_get_json, today=None):
    """Classify every key. Returns (rows, cache); rows are dicts for the report."""
    today = today or _dt.date.today().isoformat()
    manual = manual or {}
    rows, offline = [], ""
    for key in sorted(keys):
        e = entries.get(key)
        if e is None:
            rows.append({"key": key, "status": "unverified", "why": "not in refs.bib"})
            continue
        qid, hit = query_id(e), cache.get(key)
        if not cache_valid(hit, qid):
            hit = None                                   # bib or candidate rule changed
        err = ""
        if fetch and hit is None and not offline:
            try:
                rec = lookup(e, get_json)
                hit = {"query": qid, "fetched": today, "record": rec, "selector": SELECTOR}
                cache[key] = hit
            except Incomplete as exc:                    # this entry only; retried next run
                err = f"fetch incomplete ({exc}); run --fetch again"
            except urllib.error.HTTPError as exc:        # e.g. 404: the DOI does not exist
                if exc.code == 404 and e.get("doi"):
                    hit = {"query": qid, "fetched": today, "record": None, "doi_missing": True}
                    cache[key] = hit
                else:
                    err = f"fetch failed: HTTP {exc.code}; run --fetch again"
            except (urllib.error.URLError, OSError) as exc:   # refused / timeout: stop
                offline = f"network unavailable ({type(exc).__name__}); not fetched"
                err = offline
        elif fetch and hit is None:
            err = offline
        row = {"key": key, "doi": e.get("doi", ""), "manual": manual.get(key)}
        if hit is None:
            row.update(status="unverified", why=err or "no cached record (run with --fetch)")
        elif hit.get("doi_missing"):
            row.update(status="mismatch", issues=[("doi", e["doi"], "not found in Crossref")],
                       source="crossref-doi", fetched=hit["fetched"], candidate_doi="")
        elif hit.get("record") is None:
            row.update(status="unverified", why=f"no matching record found ({hit['fetched']})")
        else:
            rec = hit["record"]
            issues = compare(e, rec)
            accepted = [i for i in issues if is_accepted(i, (manual.get(key) or {}).get("accept"))]
            issues = [i for i in issues if i not in accepted]
            if issues:
                # A DOI names the work, so a difference is an error in refs.bib (or in the
                # record); a title search may have found another work or version: check it.
                status = "mismatch" if rec["source"] == "crossref-doi" else "check"
            else:
                status = "accepted" if accepted else "match"
            row.update(status=status, issues=issues, accepted=accepted,
                       source=rec["source"], fetched=hit["fetched"],
                       candidate_doi="" if e.get("doi") else rec.get("doi", ""))
        rows.append(row)
    return rows, cache


def render(rows, n_entries, scope, today=None, unused=()):
    today = today or _dt.date.today().isoformat()
    count = {s: sum(r["status"] == s for r in rows) for s in STATUSES}
    out = [f"# refs.bib check ({today})", "",
           f"Scope: {scope} ({len(rows)} of {n_entries} entries). "
           + " / ".join(f"{s} {count[s]}" for s in STATUSES) + ".", "",
           "mismatch: the record found by DOI differs (exit 1). check: a record found by "
           "title differs; it may be another work or version, so look before changing "
           "refs.bib. accepted: every difference is a known record error listed in "
           "reports/bib_manual.json.", "",
           f"Uncited refs.bib entries: {len(unused)}"
           + (f" ({', '.join(unused)})" if unused else "") + " (must be 0).", "",
           "Records come from Crossref (by DOI, or by title as a candidate), OpenAlex, and DBLP; "
           "the cache is reports/bib_cache.json. Manual checks (reports/bib_manual.json) "
           "are listed for reference and never count as a match.", "",
           "| key | status | source | detail |", "|---|---|---|---|"]
    for r in rows:
        if r["status"] == "unverified":
            detail = r.get("why", "")
        else:
            detail = "; ".join(f"{f}: bib `{b}` vs record `{v}`" for f, b, v in r.get("issues", []))
            if r.get("accepted"):
                detail = (detail + "; " if detail else "") + "accepted: " + "; ".join(
                    f"{f} `{v}`" for f, _, v in r["accepted"])
        if r.get("candidate_doi"):
            detail = (detail + "; " if detail else "") + f"candidate DOI {r['candidate_doi']} (confirm, then add to refs.bib)"
        if r.get("manual"):
            m = r["manual"]
            detail = (detail + "; " if detail else "") + \
                f"manual check {m.get('checked', '')}: {m.get('note', '')} ({m.get('source', '')})"
        out.append(f"| {r['key']} | {r['status']} | {r.get('source', '')} | {detail} |")
    return "\n".join(out) + "\n"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true", help="query Crossref / OpenAlex / DBLP for uncached entries")
    ap.add_argument("--refresh", action="store_true", help="drop the cache and fetch everything")
    ap.add_argument("--all", action="store_true", help="check every entry, not only cited ones")
    ap.add_argument("--bib", default=str(BIB))
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--report", default=str(REPORT))
    args = ap.parse_args(argv)
    entries = parse_bib(Path(args.bib).read_text(encoding="utf-8"))
    keys = set(entries) if args.all else cited_keys()
    cache = {} if args.refresh else load_json(args.cache)
    rows, cache = check(entries, keys, cache, load_json(MANUAL),
                        fetch=args.fetch or args.refresh)
    for out in (args.cache, args.report):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    if args.fetch or args.refresh:
        Path(args.cache).write_text(json.dumps(cache, indent=1, sort_keys=True,
                                               ensure_ascii=False) + "\n", encoding="utf-8")
    scope = "all entries" if args.all else "cited entries"
    unused = unused_keys(args.bib)
    Path(args.report).write_text(render(rows, len(entries), scope, unused=unused),
                                 encoding="utf-8")
    bad = [r for r in rows if r["status"] == "mismatch"]
    for r in bad:
        print(f"[mismatch] {r['key']}: " + "; ".join(f"{f}: {b} vs {v}" for f, b, v in r["issues"]))
    n = {s: sum(r["status"] == s for r in rows) for s in STATUSES}
    print(f"[check_bib] {len(rows)} entries: " + " / ".join(f"{s} {n[s]}" for s in STATUSES)
          + f" -> {Path(args.report).name}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
