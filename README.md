# Half a Link can Be Enough to Predict a Whole Link: Understanding Generalization in Knowledge Graph Foundation Models

Official implementation of `Half a Link can Be Enough to Predict a Whole Link: Understanding Generalization in Knowledge Graph Foundation Models`

If you use it, please cite:

`Gregucci, C., Theeb, O., Hernandez, D., Vergari, A. and Staab, S., 2026. Half a Link can Be Enough to Predict a Whole Link: Understanding Generalization in Knowledge Graph Foundation Models. arXiv preprint arXiv:2606.18001.`

A single deduplicated package hosting three knowledge-graph foundation models —
**ULTRA**, **MOTIF**, and **TRIX** — behind one set of entry-point scripts, plus a
shared visibility-splits evaluator (SQSA / SQUA / UQSA / UQUA) that works for all
of them.

Upstream sources: ULTRA + MOTIF from `HxyScotthuang/MOTIF`; TRIX from
`yuchengz99/TRIX`. TRIX's entity- and relation-side classes are renamed here to
`TRIXEntity` / `TRIXRelation`.

## Models

| YAML `model.class` | Python class                             | Task            |
|--------------------|------------------------------------------|-----------------|
| `Ultra`            | `kgfm.models.ultra.Ultra`                | entity          |
| `MOTIF`            | `kgfm.models.motif.MOTIF`                | entity          |
| `TRIXEntity`       | `kgfm.models.trix_entity.TRIXEntity`     | entity          |
| `TRIXRelation`     | `kgfm.models.trix_relation.TRIXRelation` | relation        |
| `TRIXNoIter`       | `kgfm.models.trix_noiter.TRIXNoIter`     | entity (variant)|

`TRIXNoIter` is TRIX with the iterative entity↔relation update removed (see
`trix_noiter.md`). ULTRA additionally supports a random-frozen-backbone training
mode (see `ultra_randfrozen.md`).

Pretrained checkpoints in `ckpts/`: `ultra_3g.pth`, `motif_3g.pth`,
`trix_entity_prediction.pth`, `trix_relation_prediction.pth`, `trix_noiter_3g.pth`,
`ultra_3g_randfrozen.pth`.

## Environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # requirements-frozen.txt pins exact versions for python 3.9 and CUDA 12.1
```

The C++/CUDA extension in `kgfm/rspmm/` is JIT-compiled by
`torch.utils.cpp_extension` on first import — the first run is slow, subsequent
runs reuse the cached build.

## Common commands

```bash
# Single-dataset entity-side eval (Ultra | MOTIF | TRIXEntity | TRIXNoIter)
python script/run.py -c config/motif/transductive/MOTIF_inference.yaml \
    --gpus '[0]' --ckpt ckpts/motif_3g.pth --dataset FB15k237

# Multi-dataset entity-side eval, optionally with visibility splits
python script/run_many.py \
    -c config/motif/transductive/MOTIF_inference.yaml \
    -d FB15k237,WN18RR \
    --gpus '[0]' --ckpt ckpts/motif_3g.pth \
    --visibility-splits --vis-csv runs/motif_visibility.csv \
    --test-only

# TRIX relation-side (separate entry points; visibility-splits N/A)
python script/run_relation.py        -c config/trix/run_relation_transductive.yaml ...
python script/run_many_relation.py   ...

# Pretraining (multi-graph)
python script/pretrain.py            -c config/ultra/transductive/ULTRA_pretrain_3g.yaml ...
python script/pretrain_relation.py   -c config/trix/pretrain_relation.yaml ...
```

## Fine-tuning

Fine-tuning continues training a pretrained checkpoint on the target dataset:
`python script/run_many.py -c <config> -d <dataset> --finetune --ckpt <ckpt>`.
The per-dataset schedule is a **fixed dict keyed by dataset, applied identically
to all three models** (ULTRA / MOTIF / TRIX) — only `batch_size` and
`num_negative` come from each model's own config; the optimizer is AdamW @ 5e-4
everywhere. Each entry is `(num_epoch, batch_per_epoch)`, where `'null'` means
one full pass over the training set:

```python
default_finetuning_config = {
    "CoDExSmall": (1, 4000), "CoDExMedium": (1, 4000), "CoDExLarge": (1, 2000),
    "FB15k237": (1, 'null'), "WN18RR": (1, 'null'),
    "YAGO310": (1, 2000), "DBpedia100k": (1, 1000), "AristoV4": (1, 2000),
    "ConceptNet100k": (1, 2000), "ATOMIC": (1, 200),
    "NELL995": (1, 'null'), "Hetionet": (1, 4000),
    "WDsinger": (3, 'null'), "FB15k237_10": (1, 'null'),
    "FB15k237_20": (1, 'null'), "FB15k237_50": (1, 1000), "NELL23k": (3, 'null'),
    "FB15k237Inductive": (1, 'null'), "WN18RRInductive": (1, 'null'),
    "NELLInductive": (3, 'null'),
    "ILPC2022SmallInductive": (1, 1000), "ILPC2022LargeInductive": (1, 1000),
    "NLIngram": (3, 'null'), "FBIngram": (3, 'null'), "WKIngram": (3, 'null'),
    "WikiTopicsMT1": (3, 'null'), "WikiTopicsMT2": (3, 'null'),
    "WikiTopicsMT3": (3, 'null'), "WikiTopicsMT4": (3, 'null'),
    "Metafam": (3, 'null'), "FBNELL": (3, 'null'),
    "HM": (1, 100),
}
```

Override per run with `--ft-epochs N` / `--ft-bpe N`.

**Note** — on datasets with a fixed `batch_per_epoch` (e.g. `CoDExSmall` = 4000),
the *epoch count* is shared but the triples seen per epoch is
`batch_per_epoch × batch_size`, so MOTIF (bs 16) and TRIX (bs 32) see more data
per epoch than ULTRA (bs 4). On `'null'` datasets all three see one full pass.

**Zero-shot fallback (off by default).** With `--zero-shot-fallback`, the
pretrained model's zero-shot valid score is used as the best-epoch baseline: if
no fine-tune epoch beats it, the original pretrained checkpoint is reloaded
(fine-tuning can never report below zero-shot). This is upstream TRIX's behavior
— **pass `--zero-shot-fallback` to reproduce the TRIX fine-tuning results.**
Without it (the default), the best fine-tune epoch is always kept, so reported
numbers can legitimately regress vs zero-shot.

## How it fits together

**Relation structures are baked at preprocess time.** ULTRA's `relation_graph`,
MOTIF's `relation_hypergraph`, and TRIX's `relation_adj` are written to
`kg-datasets/<name>/processed/data.pt` by the `pre_transform` in
`kgfm/datasets.py`; none is rebuilt at runtime on a fresh cache.

**Visibility splits (SQSA / SQUA / UQSA / UQUA).** Entity-side only. For each
test triple `(h, r, t)`, bucket by whether `(h, r)` and `(r, t)` were seen in the
model's conditioning graph at test time. Implementation in `kgfm/visibility.py`;
runtime hook is `script/run.py:test_visibility`, invoked from
`script/run_many.py --visibility-splits`. One row per `(dataset, split)` with
`split ∈ {Orig, SQSA, SQUA, UQSA, UQUA}`.

**Relation-graph induction analysis.** For each test triple, measure how many
edges the model's relation graph would gain if the test triple were added to the
conditioning graph (`n_added`), bucketed by visibility quadrant. ULTRA-side uses
`kgfm/tasks.build_relation_graph` (`kgfm/relgraph_inference.py`); TRIX-side uses
`relation_adj` (`kgfm/relgraph_inference_trix.py`). Run via
`script/analyze_relgraph_inference{,_trix}.py -d <dataset> [--max-rows N]`, which
writes a per-bucket summary CSV plus a per-triple TSV.

**Large-graph MOTIF (AristoV4).** For graphs whose relation hypergraph does not
fit on the GPU, MOTIF supports a streaming/sharding path plus a precompute
pipeline (`script/precompute_motif_relation_emb.py`,
`script/build_relation_hypergraph.py`, `kgfm/motif_aristo_shim.py`). See
`relation_embedding_precompute.md`.

## Layout

```
kgfm/                 package (models/, layers, datasets, tasks, util, visibility, rspmm/)
script/               entry points
config/               per-model YAML (ultra/ motif/ trix/ trix_noiter/)
ckpts/                pretrained checkpoints
kg-datasets/          cached preprocessed datasets (kg-datasets/<name>/processed/)
runs/                 per-run CSV output
```

Every run creates a fresh working directory
`<output_dir>/<model_class>/<dataset_class>[/<version>]/<timestamp>/` and `cd`s
into it. When running multi-dataset sweeps, pass absolute paths for `--ckpt` /
`--vis-csv` since the runner changes directory before consuming them.
