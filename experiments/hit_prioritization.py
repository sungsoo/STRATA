#!/usr/bin/env python3
"""
hit_prioritization.py
=========================
Apply STRATA (temporal causal-frontier reachability) to a drug-discovery
*hit prioritization* case study built from ChEMBL / TDC data.

WHAT THIS IS (honest framing)
-----------------------------
STRATA does NOT predict bioactivity and is NOT an activity-classification model.
It computes an *evidence-grounded, time-respecting, multi-source reachability*
signal over a heterogeneous evidence graph (compounds -> targets -> genes ->
pathways -> diseases / assays), where each edge carries the YEAR of the assay or
publication that asserts it. Under a time cutoff t, STRATA answers:

    "Using only evidence dated <= t, which compound-sources causally reach a
     given target/disease via a time-respecting evidence path?"

This reachability is used as a PRIOR / FILTER that ranks (prioritizes) candidate
compounds. Precise activity prediction is left to a downstream model. This is the
same family as RPath (Rivas-Barragan et al., PLoS Comput. Biol. 2022), which
prioritizes drugs by reasoning over causal paths in a knowledge graph; STRATA
contributes the exact, bulk many-source, auditable reachability computation with
explicit strict/closure same-time semantics for prospective (time-split)
evaluation.

DATA CONNECTION POINTS (real data plugs in here)
------------------------------------------------
Two loaders are provided:
  * load_from_chembl_tdc(...)  -- reads LOCAL ChEMBL/TDC files the user has
    downloaded (the sandbox cannot reach ebi.ac.uk / tdcommons.ai). It expects
    already-exported CSVs; see the docstring for the exact columns. If the files
    are absent it raises FileNotFoundError with instructions -- it NEVER
    fabricates bioactivity numbers.
  * make_synthetic_standin(...) -- builds a clearly-labelled SYNTHETIC evidence
    graph with the same schema so the pipeline can be exercised and verified
    end-to-end without any real data. Every output from this path is tagged
    data_source="SYNTHETIC-STANDIN" so it can never be mistaken for a real result.

OUTPUTS
-------
  * STRATA edge list in the standard format:  `compound target year`  (one per line)
  * a hit-prioritization evaluation CSV with hit@k, recall@k, and enrichment
    factor (EF) under a prospective time split, comparing:
        strata-strict, strata-closure, static-reachability (timestamps ignored),
        and a popularity/degree baseline.

Usage
-----
    # exercise with synthetic stand-in (no data needed):
    python hit_prioritization.py --synthetic --out-dir results/hitsel

    # with locally-downloaded ChEMBL/TDC exports:
    python hit_prioritization.py \
        --chembl-activities path/chembl_activities.csv \
        --tdc-dti          path/tdc_dti.csv \
        --cutoff-year 2018 --out-dir results/hitsel
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from temporal_strata.packed_reachability import (
        temporal_strata_packed, reachable_pairs,  # noqa: F401
    )
    import numpy as np
    _HAVE_CORE = True
except Exception:  # pragma: no cover
    _HAVE_CORE = False


# =====================================================================
# Evidence-graph schema
# =====================================================================
# A node id is an int. We keep a typed registry so the graph is heterogeneous:
#   compound:<chembl_id>, target:<uniprot>, gene:<symbol>, pathway:<id>,
#   disease:<id>, assay:<id>
# Edges are (year, u, v) triples in the STRATA tuple order (t, u, v).

class NodeRegistry:
    def __init__(self):
        self._id = {}
        self._name = []
        self._type = []

    def get(self, ntype: str, name: str) -> int:
        key = f"{ntype}:{name}"
        if key not in self._id:
            self._id[key] = len(self._name)
            self._name.append(key)
            self._type.append(ntype)
        return self._id[key]

    def __len__(self):
        return len(self._name)

    def type_of(self, i: int) -> str:
        return self._type[i]

    def name_of(self, i: int) -> str:
        return self._name[i]

    def ids_of_type(self, ntype: str):
        return [i for i, t in enumerate(self._type) if t == ntype]


# =====================================================================
# Loader 1: real ChEMBL/TDC (LOCAL files only -- honest, no fabrication)
# =====================================================================
def load_from_chembl_tdc(chembl_activities=None, tdc_dti=None,
                         activity_threshold_nm=1000.0):
    """Build the evidence graph from LOCAL ChEMBL/TDC exports.

    Expected `chembl_activities` CSV columns (export from your local ChEMBL):
        molecule_chembl_id, target_chembl_id, standard_type,
        standard_value, standard_units, year
      - standard_type in {IC50, Ki, Kd, EC50}; standard_value in nM.
      - An edge compound->target is created iff standard_value <= threshold_nm
        (i.e., an "active" measurement), timestamped by `year`.

    Expected `tdc_dti` CSV columns (export from a local TDC DTI dataset):
        compound_id, target_id, label, year   (label=1 => interaction)

    Raises FileNotFoundError (with guidance) if neither file is present. This
    function NEVER invents bioactivity values; it only reads what you provide.
    """
    reg = NodeRegistry()
    edges = []          # (year, u, v)
    pos_pairs = set()   # ground-truth (compound, target) actives, for labels

    found_any = False
    if chembl_activities and os.path.exists(chembl_activities):
        found_any = True
        with open(chembl_activities, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    val = float(row["standard_value"])
                    year = int(float(row["year"]))
                except (KeyError, ValueError):
                    continue
                if row.get("standard_type", "").upper() not in {"IC50", "KI", "KD", "EC50"}:
                    continue
                if val > activity_threshold_nm:
                    continue  # not an active measurement -> no evidence edge
                c = reg.get("compound", row["molecule_chembl_id"])
                t = reg.get("target", row["target_chembl_id"])
                edges.append((year, c, t))
                pos_pairs.add((c, t))

    if tdc_dti and os.path.exists(tdc_dti):
        found_any = True
        with open(tdc_dti, newline="") as fh:
            for row in csv.DictReader(fh):
                if str(row.get("label", "1")) not in {"1", "1.0", "True", "true"}:
                    continue
                try:
                    year = int(float(row.get("year", "0"))) or 0
                except ValueError:
                    year = 0
                c = reg.get("compound", row["compound_id"])
                t = reg.get("target", row["target_id"])
                edges.append((year, c, t))
                pos_pairs.add((c, t))

    if not found_any:
        raise FileNotFoundError(
            "No local ChEMBL/TDC files found. The sandbox cannot reach "
            "ebi.ac.uk / tdcommons.ai (HTTP 403). Download the data locally and "
            "pass --chembl-activities and/or --tdc-dti, OR run with --synthetic "
            "to exercise the pipeline on a clearly-labelled synthetic stand-in.\n"
            "Expected ChEMBL columns: molecule_chembl_id, target_chembl_id, "
            "standard_type, standard_value, standard_units, year."
        )
    return reg, edges, pos_pairs, "CHEMBL-TDC-LOCAL"


# =====================================================================
# Loader 2: synthetic stand-in (clearly labelled; for pipeline verification)
# =====================================================================
def make_synthetic_standin(n_compounds=1200, n_targets=80, n_genes=160,
                           n_pathways=40, n_years=12, seed=12345):
    """Build a SYNTHETIC heterogeneous evidence graph with the ChEMBL/TDC schema
    AND realistic multi-hop structure, so a time-respecting reachability prior is
    actually informative (not reducible to a direct compound-target edge).

    Structure:  compound -> gene -> pathway -> target
    A compound is *truly active* against a target iff there is a time-respecting
    path compound -> gene -> pathway -> target whose edge years are increasing
    (mechanistic evidence accrues over time). We hold out the *final* hop
    (pathway -> target) for some actives until after the cutoff, so that the
    PAST graph only contains the upstream evidence; a method that follows
    time-respecting multi-hop reachability (STRATA) can prioritize these
    prospective actives, while a popularity baseline cannot. Tagged SYNTHETIC.
    """
    rng = random.Random(seed)
    reg = NodeRegistry()
    comp = [reg.get("compound", f"SYN-C{i:04d}") for i in range(n_compounds)]
    gene = [reg.get("gene", f"SYN-G{j:03d}") for j in range(n_genes)]
    path = [reg.get("pathway", f"SYN-P{j:03d}") for j in range(n_pathways)]
    targ = [reg.get("target", f"SYN-T{j:03d}") for j in range(n_targets)]

    base = 2008
    edges = []
    pos_pairs = set()

    # Causal path must be TIME-INCREASING: compound (early) -> gene (mid) ->
    # pathway (later) -> target (latest). We therefore date each hop by its
    # position in the path, not by node type, so a time-respecting journey
    # compound -> ... -> target exists.
    gene_pathway = {g: rng.choice(path) for g in gene}
    pathway_targets = {p: set(rng.sample(targ, rng.randint(1, 3))) for p in path}

    # compound -> gene  (earliest hop)
    for c in comp:
        for g in rng.sample(gene, rng.randint(1, 3)):
            y_cg = base + rng.randint(0, 4)            # 2008-2012
            edges.append((y_cg, c, g))
            p = gene_pathway[g]
            # gene -> pathway  (later than the compound's binding)
            y_gp = y_cg + rng.randint(1, 3)            # strictly after c->g
            edges.append((y_gp, g, p))
            # pathway -> target (latest hop); this is the mechanistic link that
            # often only gets published later -> good prospective hold-out.
            for t in pathway_targets[p]:
                y_pt = y_gp + rng.randint(1, 3)        # strictly after g->p
                edges.append((y_pt, p, t))
                pos_pairs.add((c, t))

    # noise: spurious compound->gene edges (no onward causal completion bias)
    for _ in range(n_compounds // 2):
        c = rng.choice(comp)
        g = rng.choice(gene)
        edges.append((base + rng.randint(0, 4), c, g))

    # A fraction of mechanistic links are published in the SAME year as their
    # predecessor. Under closure these same-year hops chain (admitting extra,
    # weaker-evidence reachability); under strict they cannot, which is the
    # leakage-prevention behavior we want to surface. We add a few same-year
    # gene->pathway shortcuts to make strict and closure differ.
    for g in rng.sample(gene, k=max(1, n_genes // 4)):
        p = gene_pathway[g]
        # find a representative compound->gene year and attach a same-year g->p
        y_same = base + rng.randint(0, 4)
        edges.append((y_same, g, p))  # same-tick as some c->g edges

    return reg, edges, pos_pairs, "SYNTHETIC-STANDIN"


# =====================================================================
# Edge export (standard STRATA format: `compound target year`)
# =====================================================================
def export_strata_edges(reg, edges, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("# src dst timestamp  (compound-id target-id year); typed names in sidecar\n")
        for (year, u, v) in edges:
            fh.write(f"{u} {v} {year}\n")
    with open(path + ".nodes", "w") as fh:
        for i in range(len(reg)):
            fh.write(f"{i}\t{reg.name_of(i)}\n")
    return path


# =====================================================================
# Prioritization scorers
# =====================================================================
def _split(edges, cutoff_year):
    past = [(t, u, v) for (t, u, v) in edges if t <= cutoff_year]
    future = [(t, u, v) for (t, u, v) in edges if t > cutoff_year]
    return past, future


def score_strata(reg, past_edges, mode):
    """Prioritization score for each (compound, target) under STRATA reachability
    on PAST evidence. A prospective hit is typically NOT yet directly reachable
    (its final mechanistic hop arrives later), so a 0/1 direct-reachability score
    is uninformative for forward prediction. Instead we score by *upstream*
    time-respecting reachability: a compound is prioritized for a target if it
    reaches the target's known upstream regulators (genes/pathways) via a
    time-respecting evidence path. This is the auditable, evidence-grounded prior
    STRATA contributes; precise activity prediction is left downstream."""
    if not _HAVE_CORE:
        return None
    n = len(reg)
    state = temporal_strata_packed(n, past_edges, same_time_mode=mode)

    def reaches(v, c):
        return (int(state[v][c >> 6]) >> (c & 63)) & 1

    targets = reg.ids_of_type("target")
    compounds = reg.ids_of_type("compound")
    # upstream regulators of each target = gene/pathway nodes reaching it in past
    upstream = {t: [] for t in targets}
    for v in range(n):
        if reg.type_of(v) in ("gene", "pathway"):
            for t in targets:
                if reaches(t, v):
                    upstream[t].append(v)

    scores = {}
    for t in targets:
        ups = upstream[t]
        denom = float(len(ups)) if ups else 1.0
        for c in compounds:
            if reaches(t, c):
                scores[(c, t)] = 2.0  # direct time-respecting evidence path
            else:
                hits = sum(1 for v in ups if reaches(v, c))
                scores[(c, t)] = hits / denom
    return scores


def score_static(reg, past_edges):
    """Timestamp-ignoring reachability: collapse all past edges to one tick."""
    if not _HAVE_CORE:
        return None
    flat = [(0, u, v) for (_, u, v) in past_edges]
    return score_strata(reg, flat, mode="closure")


def score_popularity(reg, past_edges):
    """Degree/popularity baseline: a compound's score for a target is its overall
    evidence degree (how many edges it has), independent of the target."""
    deg = defaultdict(int)
    for (_, u, v) in past_edges:
        deg[u] += 1
    mx = max(deg.values()) if deg else 1
    scores = {}
    for t in reg.ids_of_type("target"):
        for c in reg.ids_of_type("compound"):
            scores[(c, t)] = deg[c] / mx
    return scores


# =====================================================================
# Evaluation: hit@k, recall@k, enrichment factor (prospective)
# =====================================================================
def evaluate(reg, scores, future_pos, k_frac=0.05):
    """For each target, rank compounds by score; a 'hit' is a (compound,target)
    that becomes a known active in the FUTURE split. Report hit@k, recall@k, and
    enrichment factor at the top k_frac fraction."""
    if scores is None:
        return {"hit_at_k": None, "recall_at_k": None, "enrichment_factor": None}
    compounds = reg.ids_of_type("compound")
    targets = reg.ids_of_type("target")
    k = max(1, int(round(len(compounds) * k_frac)))

    fut_by_target = defaultdict(set)
    for (c, t) in future_pos:
        fut_by_target[t].add(c)

    total_hits_at_k = 0
    total_topk = 0
    total_recall_num = 0
    total_recall_den = 0
    ef_terms = []
    for t in targets:
        pos = fut_by_target.get(t, set())
        if not pos:
            continue
        ranked = sorted(compounds, key=lambda c: scores.get((c, t), 0.0), reverse=True)
        topk = ranked[:k]
        hits = sum(1 for c in topk if c in pos)
        total_hits_at_k += hits
        total_topk += len(topk)
        total_recall_num += hits
        total_recall_den += len(pos)
        # enrichment factor: (hits/k) / (actives/total)
        base_rate = len(pos) / len(compounds)
        ef = (hits / k) / base_rate if base_rate > 0 else 0.0
        ef_terms.append(ef)

    return {
        "hit_at_k": (total_hits_at_k / total_topk) if total_topk else 0.0,
        "recall_at_k": (total_recall_num / total_recall_den) if total_recall_den else 0.0,
        "enrichment_factor": (sum(ef_terms) / len(ef_terms)) if ef_terms else 0.0,
        "k": k,
        "n_targets_evaluated": len(ef_terms),
    }


# =====================================================================
# Driver
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="use the labelled synthetic stand-in (no data needed)")
    ap.add_argument("--chembl-activities", default=None)
    ap.add_argument("--tdc-dti", default=None)
    ap.add_argument("--cutoff-year", type=int, default=2018)
    ap.add_argument("--k-frac", type=float, default=0.05)
    ap.add_argument("--out-dir", default="results/hitsel")
    args = ap.parse_args()

    if args.synthetic or (not args.chembl_activities and not args.tdc_dti):
        reg, edges, pos, source = make_synthetic_standin()
        if not args.synthetic:
            print("[note] no ChEMBL/TDC files given; using SYNTHETIC stand-in.")
    else:
        reg, edges, pos, source = load_from_chembl_tdc(
            args.chembl_activities, args.tdc_dti)

    os.makedirs(args.out_dir, exist_ok=True)
    edge_path = export_strata_edges(reg, edges, os.path.join(args.out_dir, "evidence_edges.txt"))
    print(f"data_source={source}  nodes={len(reg)}  edges={len(edges)}  "
          f"actives={len(pos)}")
    print(f"wrote {edge_path}")

    past, future = _split(edges, args.cutoff_year)

    # Prospective actives = true (compound,target) actives that are NOT yet
    # supported by a time-respecting path in the PAST graph but ARE in the FULL
    # graph. These are genuine forward predictions: a method that extrapolates
    # the time-respecting multi-hop evidence (STRATA) should rank them highly
    # from partial past evidence, whereas a popularity baseline cannot.
    future_pos = set()
    if _HAVE_CORE:
        n = len(reg)
        full_state = temporal_strata_packed(n, edges, same_time_mode="closure")
        past_state = temporal_strata_packed(n, past, same_time_mode="closure")

        def reaches(state, c, t):
            return (int(state[t][c >> 6]) >> (c & 63)) & 1
        for (c, t) in pos:
            if reaches(full_state, c, t) and not reaches(past_state, c, t):
                future_pos.add((c, t))
    else:
        future_pos = {(u, v) for (_, u, v) in future if (u, v) in pos}

    print(f"cutoff={args.cutoff_year}  past_edges={len(past)}  "
          f"future_edges={len(future)}  prospective_actives={len(future_pos)}")

    methods = {
        "strata-strict": lambda: score_strata(reg, past, "strict"),
        "strata-closure": lambda: score_strata(reg, past, "closure"),
        "static-reachability": lambda: score_static(reg, past),
        "popularity-baseline": lambda: score_popularity(reg, past),
    }

    rows = []
    for name, fn in methods.items():
        scores = fn()
        metrics = evaluate(reg, scores, future_pos, k_frac=args.k_frac)
        metrics["method"] = name
        metrics["data_source"] = source
        metrics["core_available"] = _HAVE_CORE
        rows.append(metrics)
        print(f"  {name:22s} hit@k={metrics['hit_at_k']} "
              f"recall@k={metrics['recall_at_k']} EF={metrics['enrichment_factor']}")

    out_csv = os.path.join(args.out_dir, "hit_prioritization.csv")
    fields = ["method", "data_source", "hit_at_k", "recall_at_k",
              "enrichment_factor", "k", "n_targets_evaluated", "core_available"]
    with open(out_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
