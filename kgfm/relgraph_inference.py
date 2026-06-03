"""
Per-triple relation-graph induction analysis.

For each test triple (h, r, t), measure how much the ULTRA-style relation graph
would grow if that triple were appended to the model's conditioning graph at
inference:

    R0     = relation_edge_set(test_data)
    R1_fwd = relation_edge_set(test_data + edge (h -> t, r))
    R1_inv = relation_edge_set(test_data + edge (t -> h, r + num_base_rel))
    nadded = |R1 \ R0|     # for each direction independently

Conditioning graphs (test_data.edge_index / edge_type) are already doubled
(forward + inverse) in both transductive and inductive settings, and every test
triple gets evaluated in both directions by the model, so both `tail`
(forward) and `head` (inverse) directions are computed here.

No filtering: the raw 4-type output of `kgfm.tasks.build_relation_graph`
(hh=0, tt=1, ht=2, th=3) is used on both sides of the set-difference.

CPU multiprocessing: workers are spawned with the `fork` start method and
inherit the baseline state through module-level globals (no pickling of the
large R0 set or the test_data tensors).
"""

from __future__ import annotations

import multiprocessing as mp
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch_geometric.data import Data

from kgfm.tasks import build_relation_graph
from kgfm.visibility import QUADRANT_NAMES


RelEdge = Tuple[int, int, int]   # (ri, type_k, rj)


@dataclass(frozen=True)
class Row:
    h: int
    t: int
    r: int
    direction: str         # "tail" or "head"
    bucket: str            # SQSA / SQUA / UQSA / UQUA
    nadded: int
    r0_size: int
    r1_size: int


# ---------------------------------------------------------------------------
# Relation-graph edge enumeration
# ---------------------------------------------------------------------------

def _to_cpu(data: Data) -> Data:
    """Detach a Data view onto CPU tensors without mutating the input."""
    return Data(
        edge_index=data.edge_index.detach().to("cpu"),
        edge_type=data.edge_type.detach().to("cpu"),
        num_nodes=int(data.num_nodes),
        num_relations=int(data.num_relations),
    )


def relation_edge_set(data: Data) -> Set[RelEdge]:
    """Run build_relation_graph on a CPU copy and return its raw edge set."""
    graph = build_relation_graph(_to_cpu(data)).relation_graph
    ei = graph.edge_index
    et = graph.edge_type
    return {(int(ei[0, i]), int(et[i]), int(ei[1, i])) for i in range(ei.shape[1])}


def add_directed_edge(data: Data, h: int, t: int, r: int) -> Data:
    """Return a new Data with one (h, t, r) edge appended. Input is not mutated."""
    new_ei = torch.cat(
        [data.edge_index, torch.tensor([[h], [t]], dtype=data.edge_index.dtype)],
        dim=1,
    )
    new_et = torch.cat(
        [data.edge_type, torch.tensor([r], dtype=data.edge_type.dtype)],
        dim=0,
    )
    return Data(
        edge_index=new_ei,
        edge_type=new_et,
        num_nodes=int(data.num_nodes),
        num_relations=int(data.num_relations),
    )


# ---------------------------------------------------------------------------
# Multiprocessing globals (inherited copy-on-write by fork workers)
# ---------------------------------------------------------------------------

_R0: Optional[Set[RelEdge]] = None
_TEST_DATA: Optional[Data] = None
_TAIL_LABELS: Optional[torch.Tensor] = None
_HEAD_LABELS: Optional[torch.Tensor] = None
_NUM_BASE_REL: int = 0
_HEADS: Optional[List[int]] = None
_TAILS: Optional[List[int]] = None
_RELS: Optional[List[int]] = None
_MAX_ROWS: Optional[int] = None


def _set_globals(
    *,
    test_data: Data,
    r0: Set[RelEdge],
    tail_labels: torch.Tensor,
    head_labels: torch.Tensor,
    num_base_rel: int,
    max_rows: Optional[int],
) -> None:
    global _R0, _TEST_DATA, _TAIL_LABELS, _HEAD_LABELS, _NUM_BASE_REL
    global _HEADS, _TAILS, _RELS, _MAX_ROWS
    _R0 = r0
    _TEST_DATA = test_data
    _TAIL_LABELS = tail_labels
    _HEAD_LABELS = head_labels
    _NUM_BASE_REL = int(num_base_rel)
    _HEADS = test_data.target_edge_index[0].cpu().tolist()
    _TAILS = test_data.target_edge_index[1].cpu().tolist()
    _RELS = test_data.target_edge_type.cpu().tolist()
    _MAX_ROWS = max_rows


def _process_one(i: int) -> Tuple[Row, Row]:
    """Compute the forward + inverse rows for test triple index i."""
    h = _HEADS[i]
    t = _TAILS[i]
    r = _RELS[i]
    r_inv = r + _NUM_BASE_REL

    r1_fwd = relation_edge_set(add_directed_edge(_TEST_DATA, h, t, r))
    r1_inv = relation_edge_set(add_directed_edge(_TEST_DATA, t, h, r_inv))

    r0_size = len(_R0)
    row_tail = Row(
        h=h, t=t, r=r,
        direction="tail",
        bucket=QUADRANT_NAMES[int(_TAIL_LABELS[i])],
        nadded=len(r1_fwd - _R0),
        r0_size=r0_size,
        r1_size=len(r1_fwd),
    )
    row_head = Row(
        h=t, t=h, r=r_inv,
        direction="head",
        bucket=QUADRANT_NAMES[int(_HEAD_LABELS[i])],
        nadded=len(r1_inv - _R0),
        r0_size=r0_size,
        r1_size=len(r1_inv),
    )
    return row_tail, row_head


def _worker_shard(args: Tuple[int, int]) -> List[Row]:
    shard_idx, num_shards = args
    n = len(_RELS)
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
    """Compute (forward, inverse) Rows for every test triple in test_data.

    test_data must carry target_edge_index / target_edge_type (the test
    triples) and edge_index / edge_type / num_nodes / num_relations (the
    conditioning graph, doubled).

    Returns (rows, r0_size) where r0_size is the cardinality of the baseline
    relation-graph edge set built from test_data.
    """
    num_base_rel = int(test_data.num_relations) // 2
    r0 = relation_edge_set(test_data)
    r0_size = len(r0)

    _set_globals(
        test_data=test_data,
        r0=r0,
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

    # When max_rows is set, the per-shard cap means the global total is
    # bounded by max_rows * num_workers; trim by index to be deterministic.
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
    """Aggregate Row stats per bucket, plus an "Orig" aggregate over all rows.

    Each entry has:
        n_total         -- directed triples in this split (rows)
        r0_size         -- |R0| baseline edge count. Filled only for "Orig";
                            None for the four quadrant entries.
        n_full          -- count where nadded == 0 (rel-graph already complete)
        n_with_added    -- count where nadded >= 1 (#triples needing >=1
                            inferred rel-graph link)
        n_added_total   -- sum of nadded across triples in this split
                            (total rel-graph links to infer)
        pct_with_added  -- 100 * n_with_added / n_total
        nadded_mean / median / std  -- distribution of per-triple nadded
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
