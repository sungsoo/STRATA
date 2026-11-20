"""Locations of the bundled datasets, relative to the artifact root.

``datasets_real_temporal/standard/`` holds the real-timestamp graphs in the
standard ``src dst timestamp`` format; ``static/facebook_combined.txt`` is the
SNAP ego-Facebook graph (no timestamps; scripts assign them).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets_real_temporal"
STANDARD = DATA_DIR / "standard"
FACEBOOK = DATA_DIR / "static" / "facebook_combined.txt"
