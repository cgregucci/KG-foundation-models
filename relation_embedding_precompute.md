# MOTIF relation-embedding precompute

This document describes the offline precompute pipeline that lets MOTIF evaluate
on a dataset whose `relation_hypergraph` is too large to build, load, or fit on
the GPU in the standard pipeline. The motivating example is **AristoV4**, but the
workflow is dataset-agnostic — any large graph that hits the same wall can reuse
it without code changes.

## When you need this

Symptom: building or loading `data.relation_hypergraph` exhausts CPU memory, or
the per-split hypergraph does not fit on the GPU once loaded. On AristoV4 the
legacy `relation_hypergraph.pt` cache is ~215 GB on disk and the per-split
hypergraph does not fit on a 40 GB GPU.

Scope: **MOTIF only**. Ultra and TRIX variants do not need this precompute —
their relation structures (`relation_graph` for Ultra, `relation_adj` for TRIX)
are tiny even on AristoV4.

## What the pipeline does

Three stages, **each run as a separate process**. The process boundaries matter:
the build's transient memory peak (~400 GiB on AristoV4) is only reclaimed by the
OS when the build process exits — CPython / glibc / PyTorch do not return freed
pages within a single long-running process. Running build → shard → compute in one
process on AristoV4 peaks at ~999 GiB RSS; splitting them keeps each stage
bounded.

### Stage 1 — build the hypergraph

```bash
python script/build_relation_hypergraph.py --dataset <Name> --output <name>_test_rh.pt
```

Reads the dataset's raw `train/valid/test.txt` and runs
`kgfm.tasks.build_relation_hypergraph` on CPU. Output: a `<name>_test_rh.pt` Data
object (~70 GB on AristoV4).

### Stage 2 — shard and run the relation model once

```bash
python script/precompute_motif_relation_emb.py \
    --dataset <Name> --ckpt ckpts/motif/motif_3g.pth \
    --hg-cache <name>_test_rh.pt \
    --shard --shard-cache <name>_shards.pt --arity3-subshards K \
    --output <name>_rel_emb.pt
```

Loads the stage-1 hypergraph, shards arity-3 motif edges by motif type via
`kgfm.motif_aristo_shim.shard_hypergraph_inplace`, and runs
`RelHCNet(hg, query=arange(R))` in chunks. Output: a `[R, R, D]` lookup table
saved as a `.pt` blob (`{"relation_embeddings", "num_relations", "hidden_dim",
"dataset", "checkpoint"}`). On AristoV4 this is ~2.6 GB.

- `--shard` is required only when the per-split hypergraph doesn't fit on the GPU.
  If it fits, drop `--shard` / `--shard-cache` and stage 2 collapses to
  `--hg-cache <stage1.pt> --output <table.pt>`.
- `--arity3-subshards K` further splits each arity-3 motif type's CSR into `K`
  sub-shards — increase it if a single motif type still exceeds GPU headroom. Sum
  aggregation is associative, so subsharding is mathematically exact. AristoV4
  uses `K=4`.
- `--shard-cache <path>` is a retry knob: if the file exists it is loaded and the
  shard step is skipped; otherwise the shard result is saved before the precompute
  loop, so a later GPU-side retry goes straight to the precompute.

### Stage 3 — eval against the precomputed table

```bash
python script/run_many.py -c config/motif/transductive/MOTIF_inference.yaml \
    -d <Name> --gpus '[0]' --ckpt ckpts/motif/motif_3g.pth \
    --test-only --rel-emb-cache <name>_rel_emb.pt
```

Reads only the small table. Internally `run_many.py` sets
`cache_key_override="relation_graph"` so the dataset loads its small
`relation_graph` cache (534 MB on AristoV4) instead of the 215 GB
`relation_hypergraph` cache, and `MOTIF.forward` indexes
`precomputed_rel_emb[r_index[:, 0]]` per batch instead of running the relation
model on the hypergraph.

## Adding a new big dataset

No code changes are required — the `run_many.py` gate already accepts any MOTIF
dataset:

```python
use_precomputed_rel_emb = (
    motif_model and args.test_only and args.rel_emb_cache is not None
)
```

To extend to a new dataset `<Name>`:

1. **Confirm the symptom** — the `relation_hypergraph` (not the entity graph) is
   what's blowing up:
   ```bash
   python -c "from kgfm import datasets; ds = datasets.<Name>(root='kg-datasets'); print(ds[2].relation_hypergraph)"
   ```
2. **Run stage 1 → stage 2 → stage 3** (commands above), in order — each consumes
   the previous stage's output file. Start with `--arity3-subshards 1` and
   increase only if the shard step runs out of GPU memory.

## Models other than MOTIF are unaffected

`--rel-emb-cache` is exposed only by `script/run_many.py`, and its gate requires
`cfg.model["class"] == "MOTIF"`, so:

- For Ultra / TRIXEntity / TRIXNoIter, `use_precomputed_rel_emb` is `False`
  regardless of CLI flags. The cache file is never read.
- `precomputed_rel_emb=None` flows unchanged to `model.forward()`. Ultra takes its
  normal `relation_model(...)` branch; TRIXNoIter asserts `precomputed_rel_emb is
  None` and proceeds; TRIXEntity has no precomputed branch at all.
- Passing `--rel-emb-cache <path>` with a non-MOTIF config is a silent no-op.

The other entry points (`script/run.py`, `script/run_relation.py`,
`script/run_many_relation.py`, `script/pretrain.py`, `script/pretrain_relation.py`)
do not declare `--rel-emb-cache` and cannot trigger the precompute path.

## Reference files

- `script/build_relation_hypergraph.py` — stage 1
- `script/precompute_motif_relation_emb.py` — stage 2 (load paths `--raw` /
  `--hg-cache` / default cache; sharding flags `--shard`, `--arity3-subshards`,
  `--shard-cache`)
- `script/run_many.py` — stage 3 gate, `cache_key_override`, rel-emb load
- `kgfm/motif_aristo_shim.py` — `shard_hypergraph_inplace`, `HypergraphShards`
