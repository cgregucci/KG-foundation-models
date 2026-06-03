"""
Query–answer visibility splits: SQSA / SQUA / UQSA / UQUA.

Definitions (from ULTRA-main/script/test_entity_splits_vs_nbfnet.py):

    seen_query  = {(h, r) : (h, *, r) is an edge in the reference graph}
    seen_answer = {(r, t) : (*, t, r) is an edge in the reference graph}

For a test triple (h, r, t):
    SQSA  – (h,r) seen AND (r,t) seen
    SQUA  – (h,r) seen AND (r,t) unseen
    UQSA  – (h,r) unseen AND (r,t) seen
    UQUA  – (h,r) unseen AND (r,t) unseen

Reference graph:
    Always the model's conditioning graph at test time, i.e.
    test_data.edge_index / edge_type.  This graph is already doubled
    (forward + inverse edges) in both transductive and inductive settings.
    For transductive, test_data.edge_index == doubled training edges.
    For inductive/fully-inductive, test_data.edge_index == doubled
    inference-graph edges.
"""

import torch
from typing import Dict, Optional, Set, Tuple

QUADRANT_NAMES = ["SQSA", "SQUA", "UQSA", "UQUA"]
SQSA, SQUA, UQSA, UQUA = 0, 1, 2, 3


def build_seen_sets_from_doubled_graph(edge_index, edge_type):
    """Build seen sets from a graph that already contains inverse edges.

    Used for inductive / fully-inductive settings where edge_index/edge_type
    is the inference graph (forward + inverse edges already present).

    Returns (seen_query, seen_answer) as Python sets of (int, int) tuples.
    """
    heads = edge_index[0].cpu().tolist()
    tails = edge_index[1].cpu().tolist()
    rels = edge_type.cpu().tolist()

    seen_query: Set[Tuple[int, int]] = set()
    seen_answer: Set[Tuple[int, int]] = set()
    for h, t, r in zip(heads, tails, rels):
        seen_query.add((h, r))
        seen_answer.add((r, t))
    return seen_query, seen_answer


def build_seen_sets_from_base_triples(target_edge_index, target_edge_type,
                                      num_base_rel):
    """Build seen sets from target triples that have base relation IDs only.

    Used for transductive settings.  Manually adds inverse-relation pairs
    (t, r_inv) / (r_inv, h) for every forward triple.
    """
    num_base_rel = int(num_base_rel)
    heads = target_edge_index[0].cpu().tolist()
    tails = target_edge_index[1].cpu().tolist()
    rels = target_edge_type.cpu().tolist()

    seen_query: Set[Tuple[int, int]] = set()
    seen_answer: Set[Tuple[int, int]] = set()
    for h, t, r in zip(heads, tails, rels):
        seen_query.add((h, r))
        seen_answer.add((r, t))
        r_inv = r + num_base_rel
        seen_query.add((t, r_inv))
        seen_answer.add((r_inv, h))
    return seen_query, seen_answer


def classify_test_triples(target_edge_index, target_edge_type,
                          seen_query, seen_answer, num_base_rel):
    """Assign a visibility quadrant to each test triple for both directions.

    For each test triple (h, t, r) with *base* relation id:
        Tail direction  – query=(h, r),          answer=(r, t)
        Head direction  – query=(t, r+num_base), answer=(r+num_base, h)

    Returns:
        tail_labels: LongTensor [N]  (values in {SQSA=0, SQUA=1, UQSA=2, UQUA=3})
        head_labels: LongTensor [N]  (same encoding)
    """
    num_base_rel = int(num_base_rel)
    heads = target_edge_index[0].cpu().tolist()
    tails = target_edge_index[1].cpu().tolist()
    rels = target_edge_type.cpu().tolist()

    n = len(rels)
    tail_labels = torch.empty(n, dtype=torch.long)
    head_labels = torch.empty(n, dtype=torch.long)

    for i, (h, t, r) in enumerate(zip(heads, tails, rels)):
        q_seen = (h, r) in seen_query
        a_seen = (r, t) in seen_answer
        if q_seen and a_seen:
            tail_labels[i] = SQSA
        elif q_seen:
            tail_labels[i] = SQUA
        elif a_seen:
            tail_labels[i] = UQSA
        else:
            tail_labels[i] = UQUA

        r_inv = r + num_base_rel
        q_seen_h = (t, r_inv) in seen_query
        a_seen_h = (r_inv, h) in seen_answer
        if q_seen_h and a_seen_h:
            head_labels[i] = SQSA
        elif q_seen_h:
            head_labels[i] = SQUA
        elif a_seen_h:
            head_labels[i] = UQSA
        else:
            head_labels[i] = UQUA

    return tail_labels, head_labels


def _metrics_from_ranks(ranking):
    """Compute MRR, H@1, H@3, H@10 from a 1-based ranking tensor."""
    rf = ranking.float()
    return {
        "mrr": (1.0 / rf).mean().item(),
        "hits@1": (ranking <= 1).float().mean().item(),
        "hits@3": (ranking <= 3).float().mean().item(),
        "hits@10": (ranking <= 10).float().mean().item(),
    }


def compute_quadrant_metrics(tail_ranks, head_ranks, tail_labels, head_labels):
    """Compute per-quadrant and Orig metrics from per-triple ranks + labels.

    Args:
        tail_ranks: LongTensor [N] – 1-based tail-direction ranks.
        head_ranks: LongTensor [N] – 1-based head-direction ranks.
        tail_labels: LongTensor [N] – quadrant label for tail direction.
        head_labels: LongTensor [N] – quadrant label for head direction.

    Returns:
        dict  {quadrant_name: {mrr, hits@1, hits@3, hits@10,
                               n_tail, n_head, n_combined}}
              plus an "Orig" entry for the aggregate.
    """
    results: Dict[str, dict] = {}

    all_ranks = torch.cat([tail_ranks, head_ranks])
    orig = _metrics_from_ranks(all_ranks)
    orig["n_tail"] = len(tail_ranks)
    orig["n_head"] = len(head_ranks)
    orig["n_combined"] = len(all_ranks)
    results["Orig"] = orig

    metric_keys = ["mrr", "hits@1", "hits@3", "hits@10"]

    for q_idx, q_name in enumerate(QUADRANT_NAMES):
        t_mask = (tail_labels == q_idx)
        h_mask = (head_labels == q_idx)
        t_ranks = tail_ranks[t_mask]
        h_ranks = head_ranks[h_mask]
        n_t = len(t_ranks)
        n_h = len(h_ranks)
        n_comb = n_t + n_h

        if n_comb == 0:
            entry = {k: float("nan") for k in metric_keys}
            entry["n_tail"] = 0
            entry["n_head"] = 0
            entry["n_combined"] = 0
            results[q_name] = entry
            continue

        t_metrics = _metrics_from_ranks(t_ranks) if n_t > 0 else None
        h_metrics = _metrics_from_ranks(h_ranks) if n_h > 0 else None

        entry: Dict[str, object] = {}
        for k in metric_keys:
            if t_metrics is not None and h_metrics is not None:
                entry[k] = (t_metrics[k] * n_t + h_metrics[k] * n_h) / n_comb
            elif t_metrics is not None:
                entry[k] = t_metrics[k]
            else:
                entry[k] = h_metrics[k]
        entry["n_tail"] = n_t
        entry["n_head"] = n_h
        entry["n_combined"] = n_comb
        results[q_name] = entry

    return results
