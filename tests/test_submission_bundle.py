"""Submission bundle copies files unchanged; URLs are withheld only with --anonymity (R48)."""
import zipfile

from tools import make_submission_bundle as mb

WITHHELD = "withheld" + " for review"      # split so this file does not match itself
URL = "https://dknife.github.io/STRATA/"


def test_default_bundle_is_byte_identical(tmp_path):
    out = tmp_path / "bundle.zip"
    assert mb.main(["--out", str(out)]) == 0
    files = dict((f"STRATA-submission/{arc}", p) for p, arc in mb.collect())
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert sorted(names) == sorted(files)
        seen_url = False
        for name in names:
            data = z.read(name)
            assert data == files[name].read_bytes(), name
            assert WITHHELD.encode() not in data, name
            seen_url |= b"dknife.github.io" in data
    assert seen_url                                   # the review copy keeps the links


def test_anonymity_withholds_urls(tmp_path, monkeypatch):
    src = tmp_path / "README.md"
    src.write_text(f"Code: {URL}\n", encoding="utf-8")
    png = tmp_path / "fig.png"
    png.write_bytes(b"\x89PNG dknife.github.io")
    monkeypatch.setattr(mb, "collect", lambda: [(src, "code/README.md"), (png, "paper/fig.png")])
    assert mb.bundle_text(src) == src.read_text()
    assert WITHHELD in mb.bundle_text(src, anonymity=True)
    out = tmp_path / "anon.zip"
    assert mb.main(["--out", str(out), "--anonymity"]) == 0
    with zipfile.ZipFile(out) as z:
        text = z.read("STRATA-submission/code/README.md").decode()
        assert WITHHELD in text and "dknife" not in text
        assert z.read("STRATA-submission/paper/fig.png") == png.read_bytes()   # binary untouched
    out2 = tmp_path / "plain.zip"
    assert mb.main(["--out", str(out2)]) == 0
    with zipfile.ZipFile(out2) as z:
        assert z.read("STRATA-submission/code/README.md") == src.read_bytes()
