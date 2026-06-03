"""Stage-1 of the AristoV4-MOTIF precompute pipeline: build and save the
relation_hypergraph from raw text files, then exit.

Single-process build → shard → compute peaked at ~999 GiB RSS on
AristoV4 because glibc/PyTorch never return ``build_relation_hypergraph``'s
~400 GiB transient pages to the OS, leaving them as a permanent floor
that ``shard_hypergraph_inplace`` then stacks 120 GB of CSR shards on
top of. Splitting build into its own process is the only reliable way
to drop the residual: ``exit(0)`` reclaims the entire address space.

This script is the ``--raw`` path of
``script/precompute_motif_relation_emb.py`` lines 145-174 truncated
after the build, then ``torch.save``-ing the hypergraph Data object.
The companion ``--hg-cache`` mode in the precompute script consumes
the artefact in a fresh process.

Usage::

    python script/build_relation_hypergraph.py \
        --dataset AristoV4 \
        --output  kg-datasets/aristov4/processed/aristov4_test_rh.pt
"""

import argparse
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kgfm import datasets
from kgfm.tasks import build_relation_hypergraph
from precompute_motif_relation_emb import load_test_data_raw


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   help="Dataset class name in kgfm.datasets (e.g. AristoV4, CoDExMedium)")
    p.add_argument("--output", required=True,
                   help="Output .pt path for the relation_hypergraph Data object")
    p.add_argument("--root", default=os.path.join(REPO, "kg-datasets"),
                   help="kg-datasets root directory")
    p.add_argument("--raw-dir-name", default=None,
                   help="Override the lowercase dataset directory name under --root "
                        "(default: dataset class .name attribute, falling back to "
                        "lowercased --dataset).")
    return p.parse_args()


def main():
    args = parse_args()

    ds_cls = getattr(datasets, args.dataset)
    delimiter = getattr(ds_cls, "delimiter", None)
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
    test_data.device = "cpu"
    build_relation_hypergraph(test_data, max_arity=3)
    rh = test_data.relation_hypergraph
    print(f"  hg: num_nodes={int(rh.num_nodes)}, "
          f"motif_types={int(rh.num_relations)}, "
          f"edges={rh.edge_index.shape[1]} "
          f"(built in {time.time()-t0:.1f}s)", flush=True)

    print(f"\n== save to {args.output} ==", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    t0 = time.time()
    torch.save(rh, args.output)
    size_gb = os.path.getsize(args.output) / (1024**3)
    print(f"  saved {size_gb:.2f} GB in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
