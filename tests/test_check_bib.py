"""Reference check against Crossref / DBLP records (STRATA-v16, R46).

All tests use fake records; none touches the network.
"""
import json
import urllib.error

import pytest

from tools import check_bib as cb

BIB = r"""
% comment
@article{aorm,
  title={{AORM}: Fast Incremental Arbitrary-Order Reachability Matrix Computation},
  author={Kim, Sung-Soo and Kim, Young-Kuk and Kang, Young-Min},
  journal={IEEE Access}, volume={9}, pages={69539--69558}, year={2021},
  doi={10.1109/ACCESS.2021.3077888}
}
@inproceedings{nodoi,
  title = "Sparsity-Aware Reachability Computation for Massive Graphs",
  author = {Kim, Sung-Soo and Kang, Young-Min and Kim, Young-Kuk},
  pages = {157--160}, year = 2022
}
"""

AORM = {"DOI": "10.1109/access.2021.3077888",
        "title": ["AORM: Fast Incremental Arbitrary-Order Reachability Matrix Computation"],
        "author": [{"given": "Sung-Soo", "family": "Kim"}, {"given": "Young-Kuk", "family": "Kim"},
                   {"given": "Young-Min", "family": "Kang"}],
        "issued": {"date-parts": [[2021]]}, "volume": "9", "page": "69539-69558",
        "container-title": ["IEEE Access"]}


def entries():
    return cb.parse_bib(BIB)


def rec(**change):
    msg = json.loads(json.dumps(AORM))
    msg.update(change)
    return cb.crossref_record(msg, "crossref-doi")


def test_parse_bib_braces_quotes_numbers():
    e = entries()
    assert set(e) == {"aorm", "nodoi"}
    assert e["aorm"]["title"].startswith("{AORM}: Fast")
    assert e["aorm"]["doi"] == "10.1109/ACCESS.2021.3077888"
    assert e["nodoi"]["title"] == "Sparsity-Aware Reachability Computation for Massive Graphs"
    assert e["nodoi"]["year"] == "2022" and e["nodoi"]["type"] == "inproceedings"


def test_plain_latex_equals_unicode():
    assert cb.plain(r"Bulu\c{c}, Ayd{\i}n") == cb.plain("Buluç, Aydın") == "buluc aydin"
    assert cb.plain(r"Kunegis, J{\'e}r{\^o}me") == cb.plain("Kunegis, Jérôme")
    assert cb.plain(r"Efficient $k$-hop") == "efficient k hop"
    assert cb.plain(r"Gau{\ss}, \O{}stergaard \textit{Ayd{\i}n}") == "gauss ostergaard aydin"


def test_match_and_published_spelling():
    e = entries()["aorm"]
    assert cb.compare(e, rec()) == []
    sungsoo = rec(author=[{"given": "Sungsoo", "family": "Kim"}, {"given": "Young-Kuk", "family": "Kim"},
                          {"given": "Young-Min", "family": "Kang"}])
    assert cb.compare(e, sungsoo) == []


def test_wrong_given_name_is_caught():
    """The v15 error: "Young-Keun" in refs.bib for "Young-Kuk"."""
    e = dict(entries()["aorm"], author="Kim, Sung-Soo and Kim, Young-Keun and Kang, Young-Min")
    issues = cb.compare(e, rec())
    assert [i[0] for i in issues] == ["given name"]
    assert cb.same_given("M. Tamer", "M. Tamer") and cb.same_given("Ben", "Benjamin")
    assert cb.same_given("J.", "Julian") and not cb.same_given("Young-Kuk", "Young-Keun")


def test_author_order_names_and_count():
    e = entries()["aorm"]
    swapped = dict(e, author="Kim, Sung-Soo and Kang, Young-Min and Kim, Young-Kuk")
    assert [i[0] for i in cb.compare(swapped, rec())] == ["author order"]
    wrong = dict(e, author="Kim, Sung-Soo and Lee, Young-Kuk and Kang, Young-Min")
    assert [i[0] for i in cb.compare(wrong, rec())] == ["author names"]
    short = dict(e, author="Kim, Sung-Soo and Kim, Young-Kuk")
    assert "author count" in [i[0] for i in cb.compare(short, rec())]
    truncated = dict(e, author="Kim, Sung-Soo and others")
    assert cb.compare(truncated, rec()) == []


def test_title_year_volume_pages():
    e = entries()["aorm"]
    issues = cb.compare(e, rec(title=["Something Else Entirely"], issued={"date-parts": [[2020]]},
                               volume="8", page="1-20"))
    assert [i[0] for i in issues] == ["title", "year", "volume", "pages"]
    online = rec(issued={"date-parts": [[2020]]}, **{"published-print": {"date-parts": [[2021]]}})
    assert cb.compare(e, online) == []          # any published year is accepted


def fake(responses):
    calls = []

    def get_json(url):
        calls.append(url)
        for part, value in responses.items():
            if part in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise AssertionError("unexpected url " + url)
    get_json.calls = calls
    return get_json


def test_check_doi_match_and_cache_reuse():
    e = entries()
    get = fake({"api.crossref.org/works/10.1109": {"message": AORM}})
    rows, cache = cb.check(e, {"aorm"}, {}, fetch=True, get_json=get, today="2026-11-17")
    assert cache["aorm"]["selector"] == cb.SELECTOR
    assert rows[0]["status"] == "match" and rows[0]["source"] == "crossref-doi"
    assert cache["aorm"]["query"] == "doi:10.1109/access.2021.3077888"
    rows, _ = cb.check(e, {"aorm"}, cache, fetch=False)          # offline from cache
    assert rows[0]["status"] == "match"
    e["aorm"]["doi"] = "10.1109/OTHER"                          # bib changed: cache is stale
    rows, _ = cb.check(e, {"aorm"}, cache, fetch=False)
    assert rows[0]["status"] == "unverified"


def test_title_search_gives_candidate_and_dblp_fallback():
    e = entries()
    item = {"DOI": "10.1109/bigcomp54360.2022.00038",
            "title": ["Sparsity-Aware Reachability Computation for Massive Graphs"],
            "author": [{"given": "Sung-Soo", "family": "Kim"}, {"given": "Young-Min", "family": "Kang"},
                       {"given": "Young-Kuk", "family": "Kim"}],
            "issued": {"date-parts": [[2022]]}, "page": "157-160"}
    get = fake({"query.bibliographic": {"message": {"items": [item]}}})
    rows, _ = cb.check(e, {"nodoi"}, {}, fetch=True, get_json=get)
    assert rows[0]["status"] == "match" and rows[0]["candidate_doi"] == item["DOI"]
    hit = {"info": {"title": "Sparsity-Aware Reachability Computation for Massive Graphs.",
                    "authors": {"author": [{"text": "Sung-Soo Kim 0001"}, {"text": "Young-Min Kang"},
                                           {"text": "Young-Kuk Kim"}]},
                    "year": "2022", "pages": "157-160", "doi": "10.1/x"}}
    get = fake({"query.bibliographic": {"message": {"items": []}},
                "openalex.org": {"results": []},
                "dblp.org": {"result": {"hits": {"hit": [hit]}}}})
    rows, _ = cb.check(e, {"nodoi"}, {}, fetch=True, get_json=get)
    assert rows[0]["status"] == "match" and rows[0]["source"] == "dblp"


def test_missing_doi_is_mismatch_and_network_error_stops_fetching():
    e = entries()
    err404 = urllib.error.HTTPError("u", 404, "Not Found", None, None)
    rows, _ = cb.check(e, {"aorm"}, {}, fetch=True, get_json=fake({"crossref": err404}))
    assert rows[0]["status"] == "mismatch" and rows[0]["issues"][0][0] == "doi"
    get = fake({"crossref": urllib.error.URLError("refused")})
    rows, cache = cb.check(e, {"aorm", "nodoi"}, {}, fetch=True, get_json=get)
    assert [r["status"] for r in rows] == ["unverified", "unverified"]
    assert len(get.calls) == 1 and cache == {}                    # one try, then offline


def test_main_report_and_exit_code(tmp_path, monkeypatch):
    bib = tmp_path / "refs.bib"
    bib.write_text(BIB.replace("Kim, Young-Kuk and Kang", "Kim, Young-Keun and Kang"), encoding="utf-8")
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"aorm": {"query": "doi:10.1109/access.2021.3077888",
                                          "fetched": "2026-11-17", "record": rec(),
                                          "selector": cb.SELECTOR}}))
    monkeypatch.setattr(cb, "MANUAL", tmp_path / "none.json")
    report = tmp_path / "bib_check.md"
    code = cb.main(["--all", "--bib", str(bib), "--cache", str(cache), "--report", str(report)])
    text = report.read_text(encoding="utf-8")
    assert code == 1 and "| aorm | mismatch |" in text and "Young-Keun" in text
    assert "| nodoi | unverified |" in text and "mismatch 1 / unverified 1" in text


def test_bib_check_is_optional_not_a_gate():
    from tools import run_gates
    assert "bib" not in " ".join(run_gates.GATES)
    assert "--bib" in run_gates.__doc__


@pytest.mark.parametrize("pages,first,last", [("69539--69558", "69539", "69558"),
                                              ("e1009909", "e1009909", ""), ("26", "26", "")])
def test_page_parts(pages, first, last):
    assert cb.first_page(pages) == first and cb.last_page(pages) == last


# ----------------------------------------------------------------------------- v17 (R46 fixes, R47)

def test_name_particles():
    """refs.bib "von Niederh\"ausern, Luc" vs Crossref family "Niederhausern", given "Luc Von"."""
    e = dict(entries()["aorm"], author=r"Altman, Erik and von Niederh\"ausern, Luc")
    r = rec(author=[{"given": "Erik", "family": "Altman"}, {"given": "Luc Von", "family": "Niederhäusern"}])
    assert [i[0] for i in cb.compare(e, r) if i[0].startswith("author")] == []
    assert cb.family_key({"family": "van der Berg"}) == "berg"
    assert cb.family_key({"family": "De"}) == "de"                 # a lone particle is the name


def test_candidate_needs_same_numbers_and_prefers_bib_year():
    e = {"title": "Algorithm 1000: SuiteSparse:GraphBLAS: Graph Algorithms in the Language of "
                  "Sparse Linear Algebra", "year": "2019"}
    other = {"title": "Algorithm 1037: SuiteSparse:GraphBLAS: Parallel Graph Algorithms in the "
                      "Language of Sparse Linear Algebra", "years": [2023]}
    assert cb.best_candidate(e, [other]) is None
    g = {"title": "Theoretically Efficient Parallel Graph Algorithms Can Be Fast and Scalable", "year": "2021"}
    spaa = dict(title=g["title"], years=[2018], doi="spaa")
    topc = dict(title=g["title"], years=[2021], doi="topc")
    assert cb.best_candidate(g, [spaa, topc])["doi"] == "topc"
    assert cb.best_candidate(g, [spaa])["doi"] == "spaa"            # then compare() flags the year


def test_bad_answer_is_per_entry_and_not_cached():
    """DBLP's bot page (not JSON) must not stop the run or be cached as 'no record'."""
    e = entries()
    get = fake({"query.bibliographic": {"message": {"items": []}},
                "openalex.org": {"results": []},
                "dblp.org": ValueError("Expecting value"),
                "api.crossref.org/works/10.1109": {"message": AORM}})
    rows, cache = cb.check(e, {"aorm", "nodoi"}, {}, fetch=True, get_json=get)
    status = {r["key"]: r for r in rows}
    assert status["aorm"]["status"] == "match"
    assert status["nodoi"]["status"] == "unverified" and "incomplete" in status["nodoi"]["why"]
    assert "nodoi" not in cache


def test_old_title_answers_are_refetched():
    e = entries()
    old = {"nodoi": {"query": cb.query_id(e["nodoi"]), "fetched": "2026-11-17", "record": None}}
    rows, _ = cb.check(e, {"nodoi"}, old, fetch=False)
    assert rows[0]["status"] == "unverified" and "no cached record" in rows[0]["why"]
    new = {"nodoi": dict(old["nodoi"], selector=cb.SELECTOR)}
    rows, _ = cb.check(e, {"nodoi"}, new, fetch=False)
    assert "no matching record" in rows[0]["why"]
    doi = {"aorm": {"query": cb.query_id(e["aorm"]), "fetched": "x", "record": rec()}}
    assert cb.check(e, {"aorm"}, doi, fetch=False)[0][0]["status"] == "unverified"   # v19: refetch
    doi["aorm"]["selector"] = cb.SELECTOR
    assert cb.check(e, {"aorm"}, doi, fetch=False)[0][0]["status"] == "match"


def test_unused_keys(tmp_path):
    bib = tmp_path / "refs.bib"
    bib.write_text(BIB, encoding="utf-8")
    tex = tmp_path / "main.tex"
    tex.write_text(r"see~\cite{aorm} and \citep[p.~3]{x}", encoding="utf-8")
    assert cb.unused_keys(bib, (tex,)) == ["nodoi"]
    tex.write_text(r"\cite{aorm, nodoi}", encoding="utf-8")
    assert cb.unused_keys(bib, (tex,)) == []
    (tmp_path / "table.tex").write_text(r"row \cite{nodoi} \\", encoding="utf-8")
    tex.write_text("\\cite{aorm}\n% \\cite{gone}\n\\input{table}\n", encoding="utf-8")
    assert cb.unused_keys(bib, (tex,)) == []                     # cited in an \input file
    assert "gone" not in cb.cited_keys((tex,))                   # comments do not cite


def test_paper_cites_every_entry_and_bundle_skips_unused():
    """The real refs.bib (R47 gate) and the bundle rule."""
    from tools import make_submission_bundle as mb
    assert cb.unused_keys() == []
    assert "refs_unused.bib" in mb.SKIP_FILES
    assert all(not dest.endswith("refs_unused.bib") for _, dest in mb.collect())


# ----------------------------------------------------------------------------- v19 (R50)

def test_crossref_subtitle_is_part_of_the_title():
    """Crossref stores "Algorithm 1000" and the rest of the title as a subtitle."""
    e = {"title": "Algorithm 1000: {SuiteSparse:GraphBLAS}: Graph Algorithms in the Language of "
                  "Sparse Linear Algebra", "author": "Davis, Timothy A.", "year": "2019"}
    r = cb.crossref_record({"title": ["Algorithm 1000"], "subtitle": [
        "SuiteSparse:GraphBLAS: Graph Algorithms in the Language of Sparse Linear Algebra"],
        "author": [{"given": "Timothy A.", "family": "Davis"}],
        "issued": {"date-parts": [[2019]]}}, "crossref-doi")
    assert cb.compare(e, r) == []


def test_title_candidate_difference_is_check_not_mismatch():
    e = entries()
    other = dict(AORM, author=AORM["author"][:1], DOI="10.1/other")
    get = fake({"query.bibliographic": {"message": {"items": [dict(other, title=[e["nodoi"]["title"]],
                                                                    issued={"date-parts": [[2022]]})]}}})
    rows, _ = cb.check(e, {"nodoi"}, {}, fetch=True, get_json=get)
    assert rows[0]["status"] == "check" and rows[0]["issues"]


def test_accepted_record_error():
    """sparsityaware2022: Crossref spells the third author's family name "Kimt"."""
    e = entries()
    item = {"DOI": "10.1109/x", "title": [e["nodoi"]["title"]], "issued": {"date-parts": [[2022]]},
            "author": [{"given": "Sung-Soo", "family": "Kim"}, {"given": "Young-Min", "family": "Kang"},
                       {"given": "Young-Kuk", "family": "Kimt"}], "page": "157-160"}
    e["nodoi"]["doi"] = "10.1109/x"
    get = fake({"api.crossref.org/works/10.1109": {"message": item}})
    rows, cache = cb.check(e, {"nodoi"}, {}, fetch=True, get_json=get)
    assert rows[0]["status"] == "mismatch"
    rule = {"nodoi": {"accept": [{"field": "author names", "record": "kimt", "reason": "Crossref typo"}]}}
    rows, _ = cb.check(e, {"nodoi"}, cache, manual=rule, fetch=False)
    assert rows[0]["status"] == "accepted" and rows[0]["accepted"]
    no_reason = {"nodoi": {"accept": [{"field": "author names", "record": "kimt"}]}}
    assert cb.check(e, {"nodoi"}, cache, manual=no_reason)[0][0]["status"] == "mismatch"


def test_openalex_record_and_fallback():
    work = {"display_name": "Sparsity-Aware Reachability Computation for Massive Graphs",
            "doi": "https://doi.org/10.1109/bigcomp54360.2022.00038", "publication_year": 2022,
            "authorships": [{"author": {"display_name": "Sung-Soo Kim"}},
                            {"author": {"display_name": "Young-Min Kang"}},
                            {"author": {"display_name": "Young-Kuk Kim"}}],
            "biblio": {"first_page": "157", "last_page": "160"},
            "primary_location": {"source": {"display_name": "BigComp"}}}
    r = cb.openalex_record(work)
    assert r["doi"] == "10.1109/bigcomp54360.2022.00038" and r["page"] == "157-160"
    assert r["authors"][2] == {"given": "Young-Kuk", "family": "Kim"}
    get = fake({"query.bibliographic": {"message": {"items": []}}, "openalex.org": {"results": [work]}})
    rows, _ = cb.check(entries(), {"nodoi"}, {}, fetch=True, get_json=get)
    assert rows[0]["status"] == "match" and rows[0]["source"] == "openalex"
    assert not any("dblp" in u for u in get.calls)
