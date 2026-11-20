#!/usr/bin/env bash
# Thin wrapper (backlog R15): run from STRATA-code/05_Temporal_STRATA inside the
# activated virtual environment. All logic and commands live in run_m5_session.py.
exec python3 "$(dirname "$0")/run_m5_session.py" "$@"
