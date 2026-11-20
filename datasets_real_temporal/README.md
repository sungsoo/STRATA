# Real-timestamp temporal datasets (v21)

These are networks with **genuine, observed** timestamps, used in the paper's
§"Real-Timestamp Datasets: Temporal vs Static". They were obtained from public
GitHub mirrors because the canonical hosts (snap.stanford.edu, TGB object store,
zenodo.org) return HTTP 403 (`host_not_allowed`) from the build sandbox. On an
unrestricted machine, `experiments/fetch_real_temporal.py` downloads them (and a
larger ≥10^7-edge suite) directly from SNAP.

## Contents

- `raw/CollegeMsg.txt` — original `src dst unix_ts`.
- `raw/soc-sign-bitcoinotc.csv` — original `src,dst,rating,unix_ts`.
- `standard/CollegeMsg.txt`, `standard/BitcoinOTC.txt`, `standard/BitcoinOTCpos.txt`
  — standardized `src dst timestamp` (0-based compact ids, `# meta:` header),
  produced by `temporal_strata.io.convert_snap_to_standard`.

## Datasets and sources

| Standard file | Dataset | n | m | timestamps |
|---|---|---|---|---|
| CollegeMsg.txt | CollegeMsg (private messages, Univ. of California, Irvine) | 1,899 | 59,835 | real UNIX, ~194 days |
| BitcoinOTC.txt | Bitcoin-OTC who-trusts-whom | 5,881 | 35,592 | real UNIX, ~1,900 days |
| BitcoinOTCpos.txt | Bitcoin-OTC positive-trust subgraph (rating>0) | 5,573 | 32,029 | real UNIX |

`BitcoinOTCpos` is a signed induced subgraph of `BitcoinOTC` (kept only edges with
rating>0); it is disclosed as such in the paper, not as an independent dataset.

## Attribution

- CollegeMsg: Panzarasa, Opsahl, Carley, "Patterns and dynamics of users'
  behavior and interaction," JASIST 60(5), 2009. SNAP `CollegeMsg`.
- Bitcoin-OTC: Kumar, Spezzano, Subrahmanian, Faloutsos, "Edge weight prediction
  in weighted signed networks," ICDM 2016. SNAP `soc-sign-bitcoinotc`.

Please cite the original works and respect the datasets' original terms of use.

## Static graph

- `static/facebook_combined.txt` — SNAP ego-Facebook (n = 4,039, 88,234 undirected
  edges, no timestamps); scripts that need it assign timestamps themselves
  (`temporal_strata.datasets.FACEBOOK`). J. McAuley and J. Leskovec, "Learning to
  discover social circles in ego networks," NIPS 2012. SNAP `ego-Facebook`.
