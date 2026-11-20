"""Second-machine session (v34): commands rewritten for the machine."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def test_thread_grid_and_rewrite():
    from tools.run_server_session import rewrite, thread_grid
    assert thread_grid(64) == [1, 8, 16, 32, 64]
    assert thread_grid(18) == [1, 2, 4, 9, 18]
    c = rewrite("NUMBA_NUM_THREADS=18 python3 x.py --threads 1,6,12,18 --out results/real/a.json "
                "--threads best:results/real/b.json", 64, "srv")
    assert "NUMBA_NUM_THREADS=64" in c and "--threads 1,8,16,32,64" in c
    assert "results/real/" not in c and "best:results/server/srv/b.json" in c


def test_server_plan_never_writes_m5_results():
    from tools.run_server_session import SERVER_STEPS, server_commands
    plan = server_commands(32, "srv")
    assert [num for num, _, _ in plan] == list(SERVER_STEPS)
    for _, _, cmds in plan:
        for c in cmds:
            assert "results/real/" not in c
            assert "NUMBA_NUM_THREADS=18" not in c
