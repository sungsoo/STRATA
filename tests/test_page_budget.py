"""Page budget search (R66): filler placement and the doubling / bisection search."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from tools import page_budget as pb  # noqa: E402


def test_largest_fitting_matches_brute_force():
    for slack in (0, 1, 2, 7, 33):
        calls = []

        def pages(k, s=slack):
            calls.append(k)
            return 12 if k <= s else 13
        assert pb.largest_fitting(pages, 12) == slack
        assert len(calls) < 20


def test_over_limit_and_cap():
    assert pb.largest_fitting(lambda k: 13, 12) == -1
    assert pb.largest_fitting(lambda k: 12, 12, cap=50) == 50


def test_filler_goes_before_the_anchor():
    text = "intro\n% AI-use disclosure\nend"
    out = pb.with_filler(text, "% AI-use disclosure", 2)
    assert out.index("word") < out.index("% AI-use") and out.count("word") == 18
    assert pb.with_filler(text, "% AI-use disclosure", 0) == text


def test_anchors_exist_in_the_paper():
    for doc, anchor in pb.ANCHORS.items():
        assert anchor in (pb.PAPER / f"{doc}.tex").read_text()
