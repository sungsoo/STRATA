"""M5 session runner (backlog R15): commands, stop conditions, logging. No timing."""
from __future__ import annotations

import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from tools import run_m5_session as rs  # noqa: E402


def _norm(text):
    return re.sub(r"\s+", " ", text.replace("\\\n", " ")).strip()


@pytest.mark.parametrize("guide", ["REPRODUCTION_GUIDE_EN.md", "REPRODUCTION_GUIDE_KO.md"])
def test_every_session_command_is_in_the_guide(guide):
    text = open(os.path.join(ROOT, guide)).read()
    sec = _norm(text[text.index("## 10."):])
    for num, _title, cmds, _outs in rs.STEPS:
        for c in cmds:
            assert _norm(c) in sec, f"{guide}: step {num} command not in guide: {c}"


def test_steps_in_order():
    assert sorted(s[0] for s in rs.STEPS) == list(range(1, 34))   # v13 12-13, v21 14-16, v22 17-18, v23 19, v24 20, v25 21, v27 22, v28 23-25, v30 26-27, v31 28, v32 29, v34 30-32, v36 33
    assert rs.STEPS[-1][0] == 13                                     # regeneration runs last


def test_optional_commands_point_at_real_commands():
    for (num, idx), (reason, codes) in rs.OPTIONAL_COMMANDS.items():
        cmds = [s for s in rs.STEPS if s[0] == num][0][2]
        assert idx < len(cmds) and reason and (codes is None or all(c != 0 for c in codes))


def test_to_argv_env_and_interpreter():
    env, argv = rs.to_argv("NUMBA_NUM_THREADS=18 python3 tools/check_numbers.py")
    assert env == {"NUMBA_NUM_THREADS": "18"} and argv == [sys.executable, "tools/check_numbers.py"]
    env, argv = rs.to_argv("pip install raphtory")
    assert env == {} and argv == [sys.executable, "-m", "pip", "install", "raphtory"]


def test_dry_run_lists_all_commands(capsys):
    class A:
        dry_run, from_step, only_step, steps, only_external = True, 1, None, None, False
    assert rs.run(A()) == 0
    out = capsys.readouterr().out
    assert out.count("$ ") == sum(len(s[2]) for s in rs.STEPS)
    assert "Step 7" in out and "Step 11" in out


def test_idle_worker_and_external_checks(tmp_path):
    plan = {"plans": [
        {"dataset": "CollegeMsg", "W": 30, "threads": 18, "idle_workers": 17},
        {"dataset": "synthetic-PA(n=50000,...)", "W": 782, "threads": 18, "idle_workers": 0},
        {"dataset": "BitcoinOTC", "W": 92, "threads": 18, "idle_workers": 0},
    ]}
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))
    bad = rs.idle_real_workers(p)
    assert [b["dataset"] for b in bad] == ["CollegeMsg"]
    c = tmp_path / "check.json"
    c.write_text(json.dumps({"rows": [{"matched_semantics": "strict"}, {"matched_semantics": None}]}))
    assert rs.external_check_matches(c) is False
    c.write_text(json.dumps({"rows": [{"matched_semantics": "strict"}]}))
    assert rs.external_check_matches(c) is True


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    res = tmp_path / "results" / "real"
    res.mkdir(parents=True)
    monkeypatch.setattr(rs, "RES", res)
    monkeypatch.setattr(rs, "LOGS", res / "logs")
    return res


class _Args:
    dry_run, from_step, only_step, steps, only_external = False, 1, None, None, False


def _session(res):
    return json.loads(sorted((res / "logs").glob("session.*.json"))[-1].read_text())


def test_run_completes_and_preserves_existing_outputs(sandbox, monkeypatch):
    (sandbox / "out.json").write_text("old")
    monkeypatch.setattr(rs, "STEPS", [(2, "ok", ["python3 -c \"print('hi')\""], ["out.json"])])
    assert rs.run(_Args()) == 0
    s = _session(sandbox)
    assert s["status"] == "completed" and s["steps"][0]["commands"][0]["exit_code"] == 0
    assert not (sandbox / "out.json").exists()
    assert len(list(sandbox.glob("out.*.json"))) == 1   # old result kept, renamed


def test_run_stops_on_failure(sandbox, monkeypatch):
    monkeypatch.setattr(rs, "STEPS", [
        (1, "fail", ["python3 -c \"import sys; sys.exit(4)\""], []),
        (2, "never", ["python3 -c \"print(2)\""], []),
    ])
    assert rs.run(_Args()) == 1
    s = _session(sandbox)
    assert s["status"] == "stopped" and len(s["steps"]) == 1


def _writer(tmp, target, payload):
    """A tiny script that writes payload (JSON) to target; avoids shell quoting."""
    script = tmp / f"write_{target.stem}.py"
    script.write_text(f"import json, pathlib\npathlib.Path({str(target)!r}).write_text("
                      f"json.dumps({payload!r}))\n")
    return f"python3 {script}"


def test_run_stops_on_idle_plan(sandbox, monkeypatch, tmp_path):
    payload = {"plans": [{"dataset": "CollegeMsg", "W": 30, "threads": 18, "idle_workers": 17}]}
    cmd = _writer(tmp_path, sandbox / "plan_check.json", payload) + " --plan-only"
    monkeypatch.setattr(rs, "STEPS", [(1, "plan", [cmd], [])])
    assert rs.run(_Args()) == 1
    assert "idle workers" in _session(sandbox)["stop_reason"]


def test_external_timed_run_skipped_without_match(sandbox, monkeypatch, tmp_path):
    write = _writer(tmp_path, sandbox / "external_check.json",
                    {"rows": [{"matched_semantics": None}]})
    monkeypatch.setattr(rs, "STEPS", [(6, "ext", [
        "python3 -c \"print(0)\"",
        "python3 -c \"print('installed')\"",
        write,
        "python3 -c \"import sys; sys.exit(9)\"",   # would fail if it ran
    ], [])])
    assert rs.run(_Args()) == 0
    cmds = _session(sandbox)["steps"][0]["commands"]
    assert cmds[3]["status"] == "skipped"


def test_external_unavailable_is_not_fatal(sandbox, monkeypatch):
    monkeypatch.setattr(rs, "STEPS", [(6, "ext", [
        "python3 -c \"print(0)\"",
        "python3 -c \"import sys; sys.exit(1)\"",   # pip install fails
        "python3 -c \"import sys; sys.exit(3)\"",   # adapter unavailable
        "python3 -c \"import sys; sys.exit(9)\"",
    ], [])])
    assert rs.run(_Args()) == 0
    cmds = _session(sandbox)["steps"][0]["commands"]
    assert cmds[1]["status"] == "skipped-external" and cmds[3]["status"] == "skipped"


def test_steps_subset_and_only_external(capsys):
    class A:
        dry_run, from_step, only_step, steps, only_external = True, 1, None, "1,2,3,6,7", False
    rs.run(A())
    out = capsys.readouterr().out
    assert "Step 4" not in out and "Step 5" not in out and "Step 6" in out and "Step 7" in out

    class B:
        dry_run, from_step, only_step, steps, only_external = True, 1, None, None, True
    rs.run(B())
    out = capsys.readouterr().out
    assert "external baseline only" in out and "compiled_real" not in out
    assert out.count("$ ") == 3 and "pip install raphtory" in out


def test_only_external_run_keeps_semantics_gate(sandbox, monkeypatch, tmp_path):
    write = _writer(tmp_path, sandbox / "external_check.json",
                    {"rows": [{"matched_semantics": "strict"}]})
    monkeypatch.setattr(rs, "STEPS", [(6, "ext", [
        "python3 -c \"import sys; sys.exit(9)\"",   # compiled_real: must NOT run
        "python3 -c \"print('installed')\"",
        write,
        "python3 -c \"print('timed')\"",
    ], ["c.json", "e1.json", "e2.json"])])

    class A(_Args):
        only_external = True
    assert rs.run(A()) == 0
    cmds = _session(sandbox)["steps"][0]["commands"]
    assert len(cmds) == 3 and cmds[2]["exit_code"] == 0


def test_regeneration_step_follows_measurement_steps():
    """R42: measuring adds step 13; step 1 alone or --no-regen does not."""
    assert rs.with_regen({2, 3}) == {2, 3, rs.REGEN_STEP}
    assert rs.with_regen({1}) == {1}
    assert rs.with_regen({2}, no_regen=True) == {2}
    assert rs.with_regen(None) is None
    assert rs.STEPS[-1][0] == rs.REGEN_STEP and rs.STEPS[-1][2] == rs.REGEN_COMMANDS
    assert rs.REGEN_COMMANDS[-1] == "python3 tools/check_numbers.py"


def test_regenerated_files_lists_existing_paper_files():
    files = rs.regenerated_files()
    names = {f["path"] for f in files}
    assert "m5_numbers.tex" in names and "figures/fig_scaling.pdf" in names
    assert all(len(f["sha256"]) == 64 for f in files)


# ---- v26 (R71 / R72): version folder, unknown steps, environment record --------
def test_version_check_stops_in_an_older_folder(tmp_path):
    for v in ("STRATA-v24", "STRATA-v25"):
        (tmp_path / v / "STRATA-code" / "05_Temporal_STRATA").mkdir(parents=True)
    old = tmp_path / "STRATA-v24" / "STRATA-code" / "05_Temporal_STRATA"
    new = tmp_path / "STRATA-v25" / "STRATA-code" / "05_Temporal_STRATA"
    assert "STRATA-v25 exists" in rs.check_version(old)
    assert rs.check_version(old, allow_old=True) is None
    assert rs.check_version(new) is None
    assert rs.check_version(tmp_path) is None                      # outside the layout
    cur, latest = rs.version_folders(old)
    assert (cur.name, latest.name) == ("STRATA-v24", "STRATA-v25")


def test_version_numbers_compare_as_numbers(tmp_path):
    for v in ("STRATA-v9", "STRATA-v10"):
        (tmp_path / v).mkdir()
    assert rs.version_folders(tmp_path / "STRATA-v9")[1].name == "STRATA-v10"


def test_unknown_steps_stop_before_measuring(capsys):
    assert rs.unknown_steps({18, 99}) == [99]
    assert rs.unknown_steps(None) == []

    class A:
        dry_run, from_step, only_step, steps, only_external = True, 1, None, "18,99", False
    assert rs.run(A()) == 4
    assert "unknown step(s) [99]" in capsys.readouterr().out


def test_environment_record_warns_on_mixed_versions():
    home = os.path.expanduser("~")
    r = rs.environment_record(cwd=f"{home}/R/STRATA-v25/c", prefix=f"{home}/R/STRATA-v24/c/.venv313",
                              executable=f"{home}/R/STRATA-v24/c/.venv313/bin/python3",
                              runner=f"{home}/R/STRATA-v25/c/tools/run_m5_session.py")
    assert r["cwd"].startswith("~") and "STRATA-v24" in r["warning"] and "STRATA-v25" in r["warning"]
    ok = rs.environment_record(cwd="/x/STRATA-v26/c", prefix="/x/STRATA-v26/c/.venv",
                               executable="/x/STRATA-v26/c/.venv/bin/python3",
                               runner="/x/STRATA-v26/c/tools/run_m5_session.py")
    assert "warning" not in ok
