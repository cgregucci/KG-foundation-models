"""Per-motif-type sharded streaming for MOTIF on AristoV4.

AristoV4 has ~3210 relations (1605 base + inverses), which makes
build_relation_hypergraph (kgfm/tasks.py:256) produce a 215 GB
data_relation_hypergraph.pt. The arity-2 portion (edge_type in {0,1,2})
is small; the bulk is the four arity-3 motif types (TFH/TFT/HFT/HFH,
edge_type in {3,4,5,6}). The full graph fits in 1 TB host RAM but not
on a 40 GB A100, so the standard ``data.to(device)`` OOMs.

HypergraphLayer's sum-aggregation (kgfm/layers.py:288, the path used by
MOTIF_inference.yaml with ``aggregate_func: 'sum'``) is associative
across edge subsets: running the Triton kernel separately per shard and
summing the per-destination outputs equals one kernel call on the
concatenation, up to floating-point reorder. We exploit that to keep
the arity-3 shards on CPU host memory and stream them to the GPU one at
a time per layer call, with their CSR layouts precomputed once. (An
earlier revision pinned the host CSR tensors via ``.pin_memory()`` for
DMA H2D; on AristoV4 that allocated ~150 GB of duplicate pinned pages
without measurable benefit at the only call site (a one-shot precompute),
so pinning was removed.)

Activated only via the gate in script/run.py and script/run_many.py
(``cfg.dataset["class"] == "AristoV4" and cfg.model["class"] == "MOTIF"``);
all other dataset-model combinations skip this module entirely.
"""

import torch
from torch_geometric.data import Data

from kgfm.util import preprocess_triton_hypergraph


ARITY3_MOTIF_TYPES = (3, 4, 5, 6)


class HypergraphShards:
    """CPU-resident, motif-sharded replacement for a MOTIF relation_hypergraph.

    - ``arity2``: a small ``Data(edge_index, edge_type)`` with only edges of
      type {0,1,2}; moved to GPU via ``.to(device)`` like a normal Data.
    - ``arity3_csr_cpu``: list of dicts ``{rowptr, indices, etypes, pos_index}``,
      one per motif type in ``ARITY3_MOTIF_TYPES``, precomputed for the
      ``+1``-shifted edge_index. CSR tensors live in regular pageable host
      memory — ``pin_memory()`` was dropped because it allocated ~150 GB of
      duplicate pinned pages on AristoV4 for negligible benefit at the
      shim's only call site (a one-shot precompute that streams each shard
      to GPU exactly once).

    Exposes ``num_nodes`` / ``num_relations`` so existing callers (e.g.
    ``kgfm/models/motif.py``) read them as before.
    """

    def __init__(self, arity2, arity3_csr_cpu, num_nodes, num_relations):
        self.arity2 = arity2
        self.arity3_csr_cpu = arity3_csr_cpu
        self.num_nodes = num_nodes
        self.num_relations = num_relations

    @classmethod
    def from_data(cls, hg, arity3_subshards=1):
        """
        ``arity3_subshards``: split each arity-3 motif type's edges into K
        equal-sized sub-shards before preprocess. On AristoV4 a single
        motif type's CSR is ~25-38 GB on GPU, and PyTorch's stream-aware
        caching allocator can't reuse the previous chunk's memory before
        the new chunk's ``to(device)`` copy starts — so two adjacent
        chunks stack and OOM a 40 GB A100. With K=4 each sub-shard is
        ~6-10 GB; two adjacent sub-shards fit comfortably. Sum
        aggregation is associative across edge subsets, so this is
        mathematically exact (verified by chunked-equiv test).
        """
        ei = hg.edge_index
        et = hg.edge_type
        N = int(hg.num_nodes)
        R = int(hg.num_relations)

        a2_mask = et < 3
        arity2 = Data(
            edge_index=ei[:, a2_mask].contiguous(),
            edge_type=et[a2_mask].contiguous(),
            num_nodes=N,
            num_relations=R,
        )

        node_size_csr = N + 1  # +1 padding shift mirrors motif.py:105

        arity3_csr_cpu = []
        for t in ARITY3_MOTIF_TYPES:
            mask = et == t
            chunk_ei = ei[:, mask]
            chunk_et = et[mask]
            del mask
            E_t = chunk_ei.shape[1]
            if E_t == 0:
                arity3_csr_cpu.append(None)
                del chunk_ei, chunk_et
                continue
            # preprocess_triton_hypergraph expects [max_arity, E]; +1 shift mirrors
            # motif.py legacy path (the kernel uses index 0 as padding).
            shifted = chunk_ei + 1
            del chunk_ei

            K = max(1, int(arity3_subshards))
            sub_size = (E_t + K - 1) // K
            for k in range(K):
                a = k * sub_size
                b = min(a + sub_size, E_t)
                if a >= b:
                    continue
                sub_ei = shifted[:, a:b].contiguous()
                sub_et = chunk_et[a:b].contiguous()
                rowptr, indices, etypes, pos_index, _ = preprocess_triton_hypergraph(
                    sub_ei, sub_et, num_node=node_size_csr
                )
                del sub_ei, sub_et
                entry = {
                    "rowptr":    rowptr.contiguous(),
                    "indices":   indices.contiguous(),
                    "etypes":    etypes.contiguous(),
                    "pos_index": pos_index.contiguous(),
                }
                del rowptr, indices, etypes, pos_index
                arity3_csr_cpu.append(entry)

            del shifted, chunk_et

        return cls(arity2=arity2, arity3_csr_cpu=arity3_csr_cpu,
                   num_nodes=N, num_relations=R)

    def to(self, device, non_blocking=False):
        self.arity2 = self.arity2.to(device, non_blocking=non_blocking)
        return self

    def iter_arity3_csr(self, device, non_blocking=False):
        # ``non_blocking=True`` only helps when the source is in pinned memory
        # (which we no longer do). Default to blocking copies — correctness
        # guaranteed regardless.
        for entry in self.arity3_csr_cpu:
            if entry is None:
                continue
            yield {k: v.to(device, non_blocking=non_blocking) for k, v in entry.items()}


def shard_hypergraph_inplace(data, arity3_subshards=1):
    """Replace ``data.relation_hypergraph`` (a Data) with HypergraphShards.

    Idempotent. Caller must invoke before ``data.to(device)`` so the source
    tensors are on CPU.

    ``arity3_subshards``: split each arity-3 motif type into K sub-shards.
    K=1 preserves legacy 4-chunk behavior (one CSR per motif type); K>1 is
    needed when a single motif type's CSR exceeds GPU memory headroom (e.g.
    AristoV4 where each motif type's CSR is ~25 GB on a 40 GB A100).
    """
    hg = getattr(data, "relation_hypergraph", None)
    if hg is None or isinstance(hg, HypergraphShards):
        return
    assert hg.edge_index.device.type == "cpu", \
        "shard_hypergraph_inplace must run on CPU tensors (call before .to(device))"
    data.relation_hypergraph = HypergraphShards.from_data(hg, arity3_subshards=arity3_subshards)
