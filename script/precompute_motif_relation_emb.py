"""One-shot precompute of MOTIF relation embeddings.

Loads a dataset's test conditioning graph, builds the relation_hypergraph,
runs ``RelHCNet(hg, query=arange(R))`` in chunks, saves the resulting
``[R, R, D]`` lookup table to disk. The eval job can then load only this
artefact, skip the relation_hypergraph cache entirely, and pass the table
as ``precomputed_rel_emb`` to MOTIF.forward.

Three load paths:

* **Cache path** (default, suitable for small datasets like CoDExMedium):
  loads ``data_relation_hypergraph.pt`` via the standard PyG dataset class.
* **Raw path** (``--raw``): bypasses the cache by reading the raw
  ``train/valid/test.txt`` files directly, reconstructing the train
  conditioning graph (with inverse edges), and building the
  relation_hypergraph from scratch via
  ``kgfm.tasks.build_relation_hypergraph``. Avoids the legacy-pickle
  ~2× peak and the 3-collated-copies-in-memory issue. Suitable when the
  build's transient peak fits in the slurm budget alongside the
  shard+compute residual (e.g. CoDExMedium).
* **Hg-cache path** (``--hg-cache <path>``, required for AristoV4):
  loads a relation_hypergraph .pt produced by
  ``script/build_relation_hypergraph.py`` in a separate slurm job. The
  process boundary is the only reliable way to drop the build's
  ~400 GiB transient before shard+compute (CPython/glibc don't return
  freed pages to the OS within a single process).

For AristoV4 the resulting per-split relation_hypergraph still doesn't fit
on the GPU, so ``--shard`` (kgfm.motif_aristo_shim) streams arity-3
motif chunks from CPU memory per layer call.

Output file format::

    {"relation_embeddings": tensor[R, R, D] float32 on CPU,
     "num_relations": int,
     "hidden_dim": int,
     "dataset": str,
     "checkpoint": str}
"""

import argparse
import os
import sys
import time

import torch
from torch_geometric.data import Data

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kgfm import datasets
from kgfm.models import MOTIF
from kgfm.motif_aristo_shim import HypergraphShards, shard_hypergraph_inplace
from kgfm.tasks import build_relation_hypergraph


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   help="Dataset class name in kgfm.datasets (e.g. AristoV4, CoDExMedium)")
    p.add_argument("--ckpt", required=True, help="MOTIF checkpoint .pth path")
    p.add_argument("--output", required=True, help="Output .pt path for the [R,R,D] table")
    p.add_argument("--root", default=os.path.join(REPO, "kg-datasets"),
                   help="kg-datasets root directory")
    p.add_argument("--chunk-size", type=int, default=16,
                   help="Query relation chunk size (matches eval batch size)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--shard", action="store_true",
                   help="Apply shard_hypergraph_inplace before moving to GPU "
                        "(required when the relation_hypergraph doesn't fit on the GPU, "
                        "e.g. AristoV4)")
    p.add_argument("--arity3-subshards", type=int, default=1,
                   help="When --shard is set, split each arity-3 motif type's edges "
                        "into K equal sub-shards before preprocess. K=1 (default) "
                        "preserves legacy 4-chunk behavior. K>1 needed when one motif "
                        "type's CSR exceeds GPU headroom. Sum aggregation is associative, "
                        "so the result is mathematically exact (verified by chunked-equiv).")
    p.add_argument("--shard-cache", default=None,
                   help="Path to a HypergraphShards .pt cache. If the file exists, "
                        "load it directly and skip both --hg-cache loading and the "
                        "shard step (saves ~16 min on AristoV4 at K=4). If the file "
                        "does not exist, the normal load+shard path runs and the "
                        "resulting HypergraphShards is torch.save'd to this path "
                        "before the precompute loop. Useful when iterating after a "
                        "GPU-side failure: the expensive shard work is preserved.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--raw", action="store_true",
                     help="Bypass the relation_hypergraph cache and rebuild from raw "
                          "text files (kg-datasets/<name>/raw/{train,valid,test}.txt). "
                          "Required for small datasets like CoDExMedium when the cache "
                          "fits in memory.")
    src.add_argument("--hg-cache", default=None,
                     help="Path to a relation_hypergraph .pt produced by "
                          "script/build_relation_hypergraph.py. Skips both the "
                          "215 GB legacy cache and the in-process raw-text build. "
                          "Required for AristoV4 (where build's transient peak forces "
                          "the build into its own process to drop allocator residual).")
    p.add_argument("--raw-dir-name", default=None,
                   help="Override the lowercase dataset directory name under --root "
                        "(default: lowercased --dataset).")
    return p.parse_args()


def _read_triplets(path, delimiter, inv_entity, inv_rel):
    """Mirror of TransductiveDataset.load_file (kgfm/datasets.py:288)."""
    triplets = []
    n_ent = len(inv_entity)
    n_rel = len(inv_rel)
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            if delimiter is None:
                u, r, v = line.split()
            else:
                u, r, v = line.strip().split(delimiter)
            if u not in inv_entity:
                inv_entity[u] = n_ent; n_ent += 1
            if v not in inv_entity:
                inv_entity[v] = n_ent; n_ent += 1
            if r not in inv_rel:
                inv_rel[r] = n_rel; n_rel += 1
            triplets.append((inv_entity[u], inv_entity[v], inv_rel[r]))
    return triplets, inv_entity, inv_rel


def load_test_data_raw(root, dataset_name, dir_name, delimiter):
    """Reconstruct test_data from raw text files with no cache load.

    Mirrors TransductiveDataset.process (kgfm/datasets.py:318) for the
    transductive case: the conditioning graph is the train edges (with
    inverses appended), and target_edge_index/_type comes from test.txt.
    """
    raw_dir = os.path.join(root, dir_name, "raw")
    train_file = os.path.join(raw_dir, "train.txt")
    valid_file = os.path.join(raw_dir, "valid.txt")
    test_file = os.path.join(raw_dir, "test.txt")
    assert os.path.isfile(train_file), f"missing {train_file}"
    assert os.path.isfile(test_file),  f"missing {test_file}"

    inv_ent, inv_rel = {}, {}
    train_triplets, inv_ent, inv_rel = _read_triplets(train_file, delimiter, inv_ent, inv_rel)
    # valid is read only to keep the entity/relation vocabularies consistent
    # with the cached version (test_data.num_relations would otherwise differ).
    _,             inv_ent, inv_rel = _read_triplets(valid_file, delimiter, inv_ent, inv_rel)
    test_triplets, inv_ent, inv_rel = _read_triplets(test_file, delimiter, inv_ent, inv_rel)

    num_node = len(inv_ent)
    num_relations = len(inv_rel)

    train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
    train_target_etypes = torch.tensor([t[2] for t in train_triplets], dtype=torch.long)
    test_edges  = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
    test_etypes = torch.tensor([t[2] for t in test_triplets], dtype=torch.long)

    train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
    train_etypes = torch.cat([train_target_etypes, train_target_etypes + num_relations])

    test_data = Data(
        edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
        target_edge_index=test_edges, target_edge_type=test_etypes,
        num_relations=num_relations * 2, device="cpu",
    )
    return test_data


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("FAIL: --device cuda requested but CUDA is unavailable", file=sys.stderr)
        sys.exit(1)
    device = torch.device(args.device)

    if args.shard_cache and os.path.isfile(args.shard_cache):
        print(f"== load shards from {args.shard_cache} ==", flush=True)
        t0 = time.time()
        # weights_only=False so HypergraphShards (custom class) deserializes.
        shards = torch.load(args.shard_cache, map_location="cpu", weights_only=False)
        assert isinstance(shards, HypergraphShards), \
            f"shard cache is not a HypergraphShards: {type(shards).__name__}"
        test_data = Data()
        test_data.relation_hypergraph = shards
        n_shards = sum(1 for s in shards.arity3_csr_cpu if s is not None)
        print(f"  num_nodes={int(shards.num_nodes)}, "
              f"arity-2 edges={shards.arity2.edge_index.shape[1]}, "
              f"arity-3 sub-shards={n_shards} "
              f"(loaded in {time.time()-t0:.1f}s)", flush=True)
    elif args.raw:
        ds_cls = getattr(datasets, args.dataset)
        delimiter = getattr(ds_cls, "delimiter", None)
        # PyG raw_dir is <root>/<name>/raw where `name` is the class attribute
        # (e.g. CoDExMedium.name == "codex-m", AristoV4.name == "aristov4").
        # Lowercased class name (CoDExMedium → "codexmedium") is the *wrong*
        # default — fall back to it only if the class doesn't set `name`.
        dir_name = args.raw_dir_name or getattr(ds_cls, "name", args.dataset.lower())
        print(f"== load {args.dataset} from raw text "
              f"(dir={dir_name}, delimiter={delimiter!r}) ==", flush=True)
        t0 = time.time()
        test_data = load_test_data_raw(args.root, args.dataset, dir_name, delimiter)
        print(f"  num_relations={int(test_data.num_relations)}, "
              f"num_nodes={int(test_data.num_nodes)}, "
              f"conditioning_edges={test_data.edge_index.shape[1]}, "
              f"test_edges={test_data.target_edge_index.shape[1]} "
              f"(loaded in {time.time()-t0:.1f}s)", flush=True)

        print("\n== build relation_hypergraph from raw edges (CPU) ==", flush=True)
        t0 = time.time()
        # build_relation_hypergraph reads graph.device for tensor placement.
        # On CPU the sparse intermediates fit easily; the dense result is the
        # only sizeable allocation (~70 GB on AristoV4).
        test_data.device = "cpu"
        build_relation_hypergraph(test_data, max_arity=3)
        rh = test_data.relation_hypergraph
        print(f"  hg: num_nodes={int(rh.num_nodes)}, "
              f"motif_types={int(rh.num_relations)}, "
              f"edges={rh.edge_index.shape[1]} "
              f"(built in {time.time()-t0:.1f}s)", flush=True)
    elif args.hg_cache:
        print(f"== load relation_hypergraph from {args.hg_cache} ==", flush=True)
        t0 = time.time()
        rh = torch.load(args.hg_cache, map_location="cpu")
        # Stub Data — only relation_hypergraph is read downstream
        # (shard_hypergraph_inplace looks at data.relation_hypergraph; the
        # precompute loop indexes through test_data.relation_hypergraph).
        test_data = Data()
        test_data.relation_hypergraph = rh
        print(f"  hg: num_nodes={int(rh.num_nodes)}, "
              f"motif_types={int(rh.num_relations)}, "
              f"edges={rh.edge_index.shape[1]} "
              f"(loaded in {time.time()-t0:.1f}s)", flush=True)
    else:
        print(f"== load {args.dataset} with relation_hypergraph cache ==", flush=True)
        t0 = time.time()
        ds_cls = getattr(datasets, args.dataset)
        ds = ds_cls(root=args.root, device="cpu", cache_key="relation_hypergraph")
        test_data = ds[2]
        rh = test_data.relation_hypergraph
        print(f"  hg: num_nodes={int(rh.num_nodes)}, "
              f"motif_types={int(rh.num_relations)}, "
              f"edges={rh.edge_index.shape[1]} "
              f"(loaded in {time.time()-t0:.1f}s)", flush=True)

    # In RelHCNet, the relation_hypergraph's nodes are the entity-graph relations;
    # `num_relations` on the hypergraph counts motif types (7). Index the lookup
    # table by entity-graph relation id, i.e. by hypergraph node id.
    R = int(test_data.relation_hypergraph.num_nodes)

    print("\n== build MOTIF and load checkpoint ==", flush=True)
    rel_cfg = dict(input_dim=64, hidden_dims=[64, 64, 64, 64, 64, 64],
                   short_cut=True, use_triton=True, aggregate_func="sum")
    ent_cfg = dict(input_dim=64, hidden_dims=[64, 64, 64, 64, 64, 64],
                   message_func="distmult", aggregate_func="sum",
                   short_cut=True, layer_norm=True, use_triton=True)
    model = MOTIF(rel_model_cfg=rel_cfg, entity_model_cfg=ent_cfg)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd.get("model", sd), strict=True)
    model = model.to(device).eval()

    if isinstance(test_data.relation_hypergraph, HypergraphShards):
        # Came from --shard-cache; skip shard step.
        test_data.relation_hypergraph = test_data.relation_hypergraph.to(device)
    elif args.shard:
        print(f"\n== shard relation_hypergraph by motif type "
              f"(arity3_subshards={args.arity3_subshards}) ==", flush=True)
        t0 = time.time()
        shard_hypergraph_inplace(test_data, arity3_subshards=args.arity3_subshards)
        print(f"  sharded in {time.time()-t0:.1f}s", flush=True)
        if args.shard_cache:
            print(f"\n== save shards to {args.shard_cache} ==", flush=True)
            t0 = time.time()
            os.makedirs(os.path.dirname(os.path.abspath(args.shard_cache)), exist_ok=True)
            torch.save(test_data.relation_hypergraph, args.shard_cache)
            size_gb = os.path.getsize(args.shard_cache) / (1024**3)
            print(f"  saved {size_gb:.1f} GB in {time.time()-t0:.1f}s", flush=True)
        # HypergraphShards.to(device) only moves the small arity-2 portion;
        # arity-3 CSR shards stay on CPU and stream per layer call.
        test_data.relation_hypergraph = test_data.relation_hypergraph.to(device)
    else:
        # Small hypergraph fits on GPU: move directly.
        test_data.relation_hypergraph = test_data.relation_hypergraph.to(device)

    print(f"\n== precompute relation embeddings ({R} queries, chunk={args.chunk_size}) ==",
          flush=True)
    t0 = time.time()
    rows = []
    chunks_total = (R + args.chunk_size - 1) // args.chunk_size
    # Log every chunk so we can ground-truth per-chunk time after the first one.
    with torch.no_grad():
        for start in range(0, R, args.chunk_size):
            stop = min(start + args.chunk_size, R)
            q = torch.arange(start, stop, device=device, dtype=torch.long)
            out = model.relation_model(test_data.relation_hypergraph, q)
            # out: [chunk, R, D]
            rows.append(out.detach().cpu())
            chunks_done = (start // args.chunk_size) + 1
            elapsed = time.time() - t0
            rate = elapsed / chunks_done
            remaining = chunks_total - chunks_done
            eta = rate * remaining
            print(f"  {stop}/{R} queries done | {chunks_done}/{chunks_total} chunks "
                  f"({elapsed:.1f}s elapsed, ~{rate:.1f}s/chunk, "
                  f"ETA {eta/60:.1f} min for {remaining} remaining)", flush=True)
    table = torch.cat(rows, dim=0)
    assert table.shape[0] == R, f"expected first dim={R}, got {table.shape}"
    D = int(table.shape[-1])
    print(f"  table shape: {tuple(table.shape)} ({time.time()-t0:.1f}s total)", flush=True)

    print(f"\n== save to {args.output} ==", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({
        "relation_embeddings": table,
        "num_relations": R,
        "hidden_dim": D,
        "dataset": args.dataset,
        "checkpoint": os.path.abspath(args.ckpt),
    }, args.output)
    print(f"  saved {os.path.getsize(args.output) / (1024**2):.1f} MB", flush=True)


if __name__ == "__main__":
    main()
