"""TGLib adapter logic against a stand-in that follows TGLib's scan (v22, R57); no
third-party code runs here."""
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

INF = 2 ** 63 - 1


def fake_tglib():
    """Python copy of TGLib's load_ordered_edge_list + earliest_arrival_times."""
    m = types.ModuleType("pytglib")

    class TGS:
        def __init__(self, edges, nmap):
            self.edges, self.nmap = edges, nmap

        def getNodeMap(self):
            return self.nmap

        def getTimeInterval(self):
            return (min(e[2] for e in self.edges), max(e[2] + e[3] for e in self.edges))

    def load(path):
        nmap, edges = {}, []
        for line in open(path):
            u, v, t, tt = (int(x) for x in line.split())
            for x in (u, v):
                nmap.setdefault(x, len(nmap))
            edges.append((nmap[u], nmap[v], t, tt))
        edges.sort(key=lambda e: (e[2], e[3], e[0], e[1]))   # TGLib: (t, tt, u, v)
        return TGS(edges, nmap)

    def earliest(tgs, nid, ti):
        arr = [INF] * len(tgs.nmap)
        arr[nid] = 0
        for u, v, t, tt in tgs.edges:
            if arr[u] <= t and arr[v] > t + tt:
                arr[v] = t + tt
        return arr

    m.load_ordered_edge_list, m.earliest_arrival_times = load, earliest
    return m


def test_strict_matches_and_single_pass_closure_does_not(tmp_path):
    from experiments import tglib_baseline as tb
    tgl = fake_tglib()
    rows = []
    for name, n, ts, us, vs in tb.toy_streams():
        rows += tb.check_one(tgl, name, n, ts, us, vs, list(range(n)), str(tmp_path))
    strict = [r for r in rows if r["mode"] == "strict"]
    closure = [r for r in rows if r["mode"] == "closure"]
    assert all(r["matched"] for r in strict)
    assert not all(r["matched"] for r in closure)     # reverse-order same-time chain


def test_strict_on_random_streams_with_relabeling(tmp_path):
    from experiments import tglib_baseline as tb
    from temporal_strata.stream import generate_temporal_arrays, sort_validate_arrays
    tgl = fake_tglib()
    n = 60
    ts, us, vs = sort_validate_arrays(n, *generate_temporal_arrays(n, 500, 40, seed=9))
    rows = tb.check_one(tgl, "rand", n, ts, us, vs, list(range(n)), str(tmp_path))
    assert rows[0]["mode"] == "strict" and rows[0]["matched"]


def test_tg_file_format(tmp_path):
    from experiments.tglib_baseline import write_tg
    p = tmp_path / "g.tg"
    write_tg(p, np.array([5, 7]), np.array([0, 2]), np.array([1, 0]), 1)
    assert p.read_text() == "0 1 5 1\n2 0 7 1\n"


# ---- v23: source build ------ -------------------------------------------------
class _R:
    def __init__(self, rc, out=""):
        self.returncode, self.stdout, self.stderr = rc, out, ""


def test_build_plan_pins_commit_and_links_this_python(tmp_path):
    from experiments import tglib_baseline as tb
    plan = tb.build_plan(str(tmp_path / "tglib"), "/py/bin/python3", jobs=4)
    assert "--recurse-submodules" in plan["fetch"][0]
    assert tb.COMMIT in plan["pin"][1]
    assert plan["pip"][0][:4] == ["/py/bin/python3", "-m", "pip", "install"]
    conf = plan["cmake"][0]
    assert "-DCMAKE_BUILD_TYPE=Release" in conf and "-DPYTHON_EXECUTABLE=/py/bin/python3" in conf
    assert plan["build_dir"].endswith(os.path.join("tglib_cpp", "build-release"))


def test_build_failure_is_recorded_not_substituted(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from experiments import tglib_baseline as tb
    calls = []

    def runner(c):
        calls.append(c)
        return _R(128 if c[:2] == ["git", "clone"] else 0)
    monkeypatch.setattr(tb, "importable", lambda *a, **k: (False, "no module"))
    out = tmp_path / "b.json"
    args = SimpleNamespace(dest=str(tmp_path / "ext"), jobs=2, out=str(out))
    assert tb.build(args, runner=runner, python="/py") == 2
    rec = json.loads(out.read_text())
    assert rec["ok"] is False and rec["route"] is None
    assert rec["steps"][-1]["stage"] == "fetch" and rec["steps"][-1]["rc"] == 128
    assert not any(c[:3] == ["/py", "-m", "pip"] for c in calls)


def test_build_falls_back_to_cmake_and_registers_module(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from experiments import tglib_baseline as tb
    ext = tmp_path / "ext"
    (ext / "tglib" / ".git").mkdir(parents=True)                 # already fetched
    (ext / "tglib" / "pytglib").mkdir()
    (ext / "tglib" / "pytglib" / "__init__.py").write_text("from ._pytglib import *\n")
    mod_dir = ext / "tglib" / "tglib_cpp" / "build-release" / "src" / "python_binding"

    def runner(c):
        if c[:3] == ["/py", "-m", "pip"]:
            return _R(1)                                        # pyproject route fails
        if c[1:2] == ["--build"]:
            mod_dir.mkdir(parents=True)
            (mod_dir / "_pytglib.cpython-313-darwin.so").write_bytes(b"")
        return _R(0, tb.COMMIT if "rev-parse" in c else "")
    site = tmp_path / "site"
    site.mkdir()
    import sysconfig
    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"purelib": str(site)})

    def importable(python, extra_path=None):                    # importable once registered
        ok = (site / "strata_tglib_build.pth").exists()
        return ok, "1.7.1" if ok else "no module"
    monkeypatch.setattr(tb, "importable", importable)
    args = SimpleNamespace(dest=str(ext), jobs=2, out=str(tmp_path / "b.json"))
    assert tb.build(args, runner=runner, python="/py") == 0
    rec = json.loads((tmp_path / "b.json").read_text())
    assert rec["route"] == "cmake" and rec["commit"] == tb.COMMIT and rec["ok"]
    root = (site / "strata_tglib_build.pth").read_text().strip()
    assert root == str((ext / "tglib_py").resolve())
    assert sorted(os.listdir(os.path.join(root, "pytglib"))) == \
        ["__init__.py", "_pytglib.cpython-313-darwin.so"]


def test_build_installs_cmake_wheel_when_missing(tmp_path, monkeypatch):
    import json
    import shutil
    from types import SimpleNamespace
    from experiments import tglib_baseline as tb
    ext = tmp_path / "ext"
    (ext / "tglib" / ".git").mkdir(parents=True)
    calls = []

    def runner(c):
        calls.append(c)
        return _R(1 if c[:3] == ["/venv/bin/python3", "-m", "pip"] and c[-1] != "cmake" else 0)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(tb, "importable", lambda *a, **k: (False, "no module"))
    args = SimpleNamespace(dest=str(ext), jobs=2, out=str(tmp_path / "b.json"))
    assert tb.build(args, runner=runner, python="/venv/bin/python3") == 2   # no module found
    assert ["/venv/bin/python3", "-m", "pip", "install", "cmake"] in calls
    assert any(c[0] == "/venv/bin/cmake" and c[1] == "--build" for c in calls)
    rec = json.loads((tmp_path / "b.json").read_text())
    assert rec["cmake_exe"] == "/venv/bin/cmake"


def test_library_source_prefers_successful_build(tmp_path):
    import json
    from experiments import tglib_baseline as tb
    p = tmp_path / "b.json"
    assert tb.library_source(str(p))["source"] == "pypi"
    p.write_text(json.dumps({"ok": True, "commit": "abc", "route": "cmake"}))
    assert tb.library_source(str(p)) == {"source": "build", "commit": "abc", "route": "cmake"}


# ---- v24: OpenMP on macOS, -Werror fallback -----------------------------------
def test_macos_without_libomp_stops_with_hint(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from experiments import tglib_baseline as tb
    calls = []

    def runner(c, env=None):
        calls.append(c)
        return _R(1)                                            # brew --prefix libomp fails
    monkeypatch.setattr(tb, "importable", lambda *a, **k: (False, "no module"))
    args = SimpleNamespace(dest=str(tmp_path / "ext"), jobs=2, out=str(tmp_path / "b.json"))
    assert tb.build(args, runner=runner, python="/py", platform="darwin") == 2
    rec = json.loads((tmp_path / "b.json").read_text())
    assert rec["openmp"] == {"found": False, "hint": "brew install libomp"}
    assert calls == [["brew", "--prefix", "libomp"]]          # nothing fetched or built


def test_macos_passes_libomp_to_both_routes_and_relaxes_werror_once(tmp_path, monkeypatch):
    import json
    import shutil
    from types import SimpleNamespace
    from experiments import tglib_baseline as tb
    omp = tmp_path / "libomp"
    (omp / "lib").mkdir(parents=True)
    (omp / "lib" / "libomp.dylib").write_bytes(b"")
    ext = tmp_path / "ext"
    (ext / "tglib" / ".git").mkdir(parents=True)
    (ext / "tglib" / "pytglib").mkdir()
    (ext / "tglib" / "pytglib" / "__init__.py").write_text("from ._pytglib import *\n")
    calls, envs = [], []

    def runner(c, env=None):
        calls.append(c)
        envs.append(env)
        if c[:2] == ["brew", "--prefix"]:
            return _R(0, str(omp) + "\n")
        if c[1:3] == ["-m", "pip"]:
            return _R(1)
        if c[0] == "/bin/cmake" and "-S" in c and not any("CMAKE_PROJECT_INCLUDE" in a for a in c):
            return _R(1, "error: unused variable [-Werror,-Wunused-variable]")
        if c[0] == "/bin/cmake" and c[1] == "--build":
            d = tmp_path / "ext" / "tglib" / "tglib_cpp" / "build-release-nowerror" / "src"
            d.mkdir(parents=True, exist_ok=True)
            (d / "_pytglib.cpython-313-darwin.so").write_bytes(b"")
        return _R(0, tb.COMMIT if "rev-parse" in c else "")
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/cmake")
    site = tmp_path / "site"
    site.mkdir()
    import sysconfig
    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"purelib": str(site)})
    monkeypatch.setattr(tb, "importable",
                        lambda *a, **k: ((site / "strata_tglib_build.pth").exists(), "1.7.1"))
    args = SimpleNamespace(dest=str(ext), jobs=2, out=str(tmp_path / "b.json"))
    assert tb.build(args, runner=runner, python="/py", platform="darwin") == 0
    rec = json.loads((tmp_path / "b.json").read_text())
    assert rec["openmp"]["found"] and rec["werror_relaxed"] and rec["route"] == "cmake"
    pip_env = next(e for c, e in zip(calls, envs) if c[1:3] == ["-m", "pip"])
    assert f"-DOpenMP_ROOT={omp}" in pip_env["CMAKE_ARGS"]
    confs = [c for c in calls if c[0] == "/bin/cmake" and "-S" in c]
    assert len(confs) == 2 and all(f"-DOpenMP_omp_LIBRARY={omp}/lib/libomp.dylib" in c for c in confs)
    inc = next(a for a in confs[1] if a.startswith("-DCMAKE_PROJECT_INCLUDE="))
    assert "REMOVE_ITEM _opts -Werror" in open(inc.split("=", 1)[1]).read()


def test_build_record_has_no_home_path():
    from experiments import tglib_baseline as tb
    home = os.path.join(os.sep, "Users", "alice")       # built at run time: no literal path
    t = '{"python": "%s/x/.venv/bin/python3", "cmd": ["-DPython_EXECUTABLE=%s/x/py"]}' % (home, home)
    out = tb.redact_home(t, home=home)
    assert home not in out and "~/x/.venv/bin/python3" in out



# ---- v25: link flags, policy minimum, binding target, symbol capture ----------
def test_link_and_policy_arguments():
    from experiments import tglib_baseline as tb
    args = tb.openmp_args("/opt/omp")
    for kind in ("EXE", "SHARED", "MODULE"):
        assert f"-DCMAKE_{kind}_LINKER_FLAGS=-L/opt/omp/lib -lomp" in args
    plan = tb.build_plan("/r", "/py")
    assert tb.POLICY_MIN in plan["cmake"][0]
    assert plan["cmake"][1][plan["cmake"][1].index("--target") + 1] == "_pytglib"


def test_undefined_symbols_are_kept():
    from experiments import tglib_baseline as tb
    text = "a\nUndefined symbols for architecture arm64:\n  ___kmpc_fork_call\nld: failed\n"
    assert tb.undefined_symbols(text)[:2] == ["Undefined symbols for architecture arm64:",
                                              "  ___kmpc_fork_call"]
    assert tb.undefined_symbols("fine") == []
