"""
Per-triple relation-graph induction analysis using the TRIX relation graph.

The TRIX relation structure is `data.relation_adj` (see
`kgfm/tasks.py:build_relation_adj` at line 528): a dict of four Data objects
keyed `hh / ht / th / tt`. Each carries `edge_index` over relation pairs
(r1, r2) and `edge_type` over entity indices — i.e. each edge in the
relation graph is tagged with the entity that mediates it.

We treat an edge in R0/R1 as a 4-tuple `(r1, r2, entity, type)` where
`type ∈ {hh, ht, th, tt}`. R0 is built from the conditioning graph
(`test_data.edge_index / edge_type`, already doubled). For each test triple
(h, r, t) we evaluate two directions:
  - tail (forward): add (h, t, r) to R0
  - head (inverse): add (t, h, r + num_base_rel) to R0
and report `nadded = |R1 \ R0|`.

Performance: we do NOT rebuild the full relation_adj for every test triple
(O(sum entity-degree^2), ~tens of millions of ops per build). Instead we
exploit the fact that adding edge (h, t, r) only mutates `heads[h]` and
`tails[t]`, so every edge in R1 \ R0 is incident to entity h or t.

CPU multiprocessing with `fork` start method; R0 set and the heads/tails
dicts are shared via inherited globals.
"""

from __future__ import annotations

import multiprocessing as mp
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch_geometric.data import Data

from kgfm.visibility import QUADRANT_NAMES


# An edge in TRIX's relation graph: (r1, r2, entity, type) with
# type in {0=hh, 1=ht, 2=th, 3=tt}. We use ints for the type to keep tuples
# small/hashable; helper TYPE_NAMES gives the string mapping for debugging.
TrixEdge = Tuple[int, int, int, int]
TYPE_HH, TYPE_HT, TYPE_TH, TYPE_TT = 0, 1, 2, 3
TYPE_NAMES = {TYPE_HH: "hh", TYPE_HT: "ht", TYPE_TH: "th", TYPE_TT: "tt"}


@dataclass(frozen=True)
class Row:
    h: int
    t: int
    r: int
    direction: str         # "tail" or "head"
    bucket: str            # SQSA / SQUA / UQSA / UQUA
    nadded: int
    r0_size: int
    r1_size: int           # = r0_size + nadded (no support shrinkage)


# ---------------------------------------------------------------------------
# Head/tail incidence sets from the conditioning graph
# ---------------------------------------------------------------------------

def build_head_tail_sets(
    edge_index: torch.Tensor, edge_type: torch.Tensor
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """Build heads[node] / tails[node] as sets of base relation ids.

    Matches the head_set / tail_set construction inside `build_relation_adj`
    (kgfm/tasks.py:571-585) but uses sets instead of count dicts — only
    membership matters for edge enumeration.
    """
    heads_l = edge_index[0].tolist()
    tails_l = edge_index[1].tolist()
    rels_l = edge_type.tolist()

    heads: Dict[int, Set[int]] = {}
    tails: Dict[int, Set[int]] = {}
    for h, t, r in zip(heads_l, tails_l, rels_l):
        heads.setdefault(h, set()).add(r)
        tails.setdefault(t, set()).add(r)
    return heads, tails


def relation_edge_set_trix(
    heads: Dict[int, Set[int]], tails: Dict[int, Set[int]]
) -> Set[TrixEdge]:
    """Enumerate every (r1, r2, entity, type) edge from heads/tails dicts.

    Reproduces the inner loop of `build_relation_adj` (kgfm/tasks.py:587-611)
    but materialises a Python set of 4-tuples instead of four edge_index
    tensors. Used to materialise R0 once.
    """
    out: Set[TrixEdge] = set()
    nodes = set(heads.keys()) | set(tails.keys())
    for node in nodes:
        hs = heads.get(node, set())
        ts = tails.get(node, set())
        for r1 in hs:
            for r2 in hs:
                if r1 != r2:
                    out.add((r1, r2, node, TYPE_HH))
            for r2 in ts:
                out.add((r1, r2, node, TYPE_HT))
        for r1 in ts:
            for r2 in ts:
                if r1 != r2:
                    out.add((r1, r2, node, TYPE_TT))
            for r2 in hs:
                out.add((r1, r2, node, TYPE_TH))
    return out


def _edges_at_entity(
    heads_at_e: Set[int], tails_at_e: Set[int], e: int
) -> Set[TrixEdge]:
    """Enumerate every relation-graph edge tagged with entity e for given
    heads_at_e / tails_at_e snapshots."""
    out: Set[TrixEdge] = set()
    for r1 in heads_at_e:
        for r2 in heads_at_e:
            if r1 != r2:
                out.add((r1, r2, e, TYPE_HH))
        for r2 in tails_at_e:
            out.add((r1, r2, e, TYPE_HT))
    for r1 in tails_at_e:
        for r2 in tails_at_e:
            if r1 != r2:
                out.add((r1, r2, e, TYPE_TT))
        for r2 in heads_at_e:
            out.add((r1, r2, e, TYPE_TH))
    return out


def added_edges_for_triple(
    heads: Dict[int, Set[int]],
    tails: Dict[int, Set[int]],
    r0: Set[TrixEdge],
    h: int,
    t: int,
    r: int,
) -> Set[TrixEdge]:
    """Return R1 \\ R0 after adding edge (h, t, r) to the conditioning graph.

    Only entities h and t get their incidence sets mutated, so the diff lives
    entirely on those entities. Computes the augmented edges at h (and t if
    distinct) and subtracts R0. Does not mutate `heads` / `tails`.
    """
    diff: Set[TrixEdge] = set()

    # Entity h: r is added as a head relation.
    heads_h_new = heads.get(h, set()) | {r}
    tails_h_new = tails.get(h, set()) | ({r} if h == t else set())
    edges_h = _edges_at_entity(heads_h_new, tails_h_new, h)
    diff |= edges_h - r0

    # Entity t: r is added as a tail relation. Skip if h == t (already done).
    if t != h:
        heads_t_new = heads.get(t, set())
        tails_t_new = tails.get(t, set()) | {r}
        edges_t = _edges_at_entity(heads_t_new, tails_t_new, t)
        diff |= edges_t - r0

    return diff


# ---------------------------------------------------------------------------
# Multiprocessing globals (inherited copy-on-write by fork workers)
# ---------------------------------------------------------------------------

_R0: Optional[Set[TrixEdge]] = None
_HEADS: Optional[Dict[int, Set[int]]] = None
_TAILS: Optional[Dict[int, Set[int]]] = None
_HEADS_T: Optional[List[int]] = None  # test heads
_TAILS_T: Optional[List[int]] = None  # test tails
_RELS_T: Optional[List[int]] = None   # test relations
_TAIL_LABELS: Optional[torch.Tensor] = None
_HEAD_LABELS: Optional[torch.Tensor] = None
_NUM_BASE_REL: int = 0
_MAX_ROWS: Optional[int] = None


def _set_globals(
    *,
    test_data: Data,
    r0: Set[TrixEdge],
    heads: Dict[int, Set[int]],
    tails: Dict[int, Set[int]],
    tail_labels: torch.Tensor,
    head_labels: torch.Tensor,
    num_base_rel: int,
    max_rows: Optional[int],
) -> None:
    global _R0, _HEADS, _TAILS
    global _HEADS_T, _TAILS_T, _RELS_T
    global _TAIL_LABELS, _HEAD_LABELS, _NUM_BASE_REL, _MAX_ROWS
    _R0 = r0
    _HEADS = heads
    _TAILS = tails
    _HEADS_T = test_data.target_edge_index[0].cpu().tolist()
    _TAILS_T = test_data.target_edge_index[1].cpu().tolist()
    _RELS_T = test_data.target_edge_type.cpu().tolist()
    _TAIL_LABELS = tail_labels
    _HEAD_LABELS = head_labels
    _NUM_BASE_REL = int(num_base_rel)
    _MAX_ROWS = max_rows


def _process_one(i: int) -> Tuple[Row, Row]:
    h = _HEADS_T[i]
    t = _TAILS_T[i]
    r = _RELS_T[i]
    r_inv = r + _NUM_BASE_REL

    diff_fwd = added_edges_for_triple(_HEADS, _TAILS, _R0, h, t, r)
    diff_inv = added_edges_for_triple(_HEADS, _TAILS, _R0, t, h, r_inv)

    r0_size = len(_R0)
    row_tail = Row(
        h=h, t=t, r=r,
        direction="tail",
        bucket=QUADRANT_NAMES[int(_TAIL_LABELS[i])],
        nadded=len(diff_fwd),
        r0_size=r0_size,
        r1_size=r0_size + len(diff_fwd),
    )
    row_head = Row(
        h=t, t=h, r=r_inv,
        direction="head",
        bucket=QUADRANT_NAMES[int(_HEAD_LABELS[i])],
        nadded=len(diff_inv),
        r0_size=r0_size,
        r1_size=r0_size + len(diff_inv),
    )
    return row_tail, row_head


def _worker_shard(args: Tuple[int, int]) -> List[Row]:
    shard_idx, num_shards = args
    n = len(_RELS_T)
    out: List[Row] = []
    processed = 0
    for i in range(n):
        if i % num_shards != shard_idx:
            continue
        if _MAX_ROWS is not None and processed >= _MAX_ROWS:
            break
        row_tail, row_head = _process_one(i)
        out.append(row_tail)
        out.append(row_head)
        processed += 1
    return out


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------

def compute_nadded_rows(
    test_data: Data,
    tail_labels: torch.Tensor,
    head_labels: torch.Tensor,
    *,
    num_workers: int = 1,
    max_rows: Optional[int] = None,
) -> Tuple[List[Row], int]:
    """Compute (tail, head) Rows for every test triple in test_data.

    Returns (rows, r0_size).
    """
    num_base_rel = int(test_data.num_relations) // 2
    heads, tails = build_head_tail_sets(test_data.edge_index, test_data.edge_type)
    r0 = relation_edge_set_trix(heads, tails)
    r0_size = len(r0)

    _set_globals(
        test_data=test_data,
        r0=r0,
        heads=heads,
        tails=tails,
        tail_labels=tail_labels,
        head_labels=head_labels,
        num_base_rel=num_base_rel,
        max_rows=max_rows,
    )

    if num_workers <= 1:
        return _worker_shard((0, 1)), r0_size

    ctx = mp.get_context("fork")
    shards = [(i, num_workers) for i in range(num_workers)]
    with ctx.Pool(num_workers) as pool:
        chunks = pool.map(_worker_shard, shards)

    rows: List[Row] = []
    for chunk in chunks:
        rows.extend(chunk)

    if max_rows is not None and len(rows) > 2 * max_rows:
        rows = rows[: 2 * max_rows]

    return rows, r0_size


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _stats_dict(values: Sequence[int]) -> Dict[str, float]:
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
    mean = sum(values) / n
    median = statistics.median(values)
    std = statistics.pstdev(values) if n > 1 else 0.0
    return {"mean": float(mean), "median": float(median), "std": float(std)}


def aggregate_per_bucket(
    rows: Sequence[Row], r0_size: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """Aggregate Row stats per bucket plus an "Orig" aggregate.

    Schema matches the ULTRA version (kgfm.relgraph_inference.aggregate_per_bucket):
        n_total, r0_size (Orig only), n_full, n_with_added, n_added_total,
        pct_with_added, nadded_mean / median / std.
    """
    splits = ["Orig", *QUADRANT_NAMES]
    grouped: Dict[str, List[Row]] = {s: [] for s in splits}
    for r in rows:
        grouped["Orig"].append(r)
        if r.bucket in grouped:
            grouped[r.bucket].append(r)

    out: Dict[str, Dict[str, float]] = {}
    for split in splits:
        bucket_rows = grouped[split]
        n_total = len(bucket_rows)
        nadded = [r.nadded for r in bucket_rows]
        n_full = sum(1 for x in nadded if x == 0)
        n_with_added = sum(1 for x in nadded if x >= 1)
        n_added_total = sum(nadded)
        pct_with_added = (100.0 * n_with_added / n_total) if n_total > 0 else float("nan")
        stats = _stats_dict(nadded)
        out[split] = {
            "n_total": n_total,
            "r0_size": r0_size if split == "Orig" else None,
            "n_full": n_full,
            "n_with_added": n_with_added,
            "n_added_total": n_added_total,
            "pct_with_added": pct_with_added,
            "nadded_mean": stats["mean"],
            "nadded_median": stats["median"],
            "nadded_std": stats["std"],
        }
    return out
