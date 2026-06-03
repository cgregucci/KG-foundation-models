# TRIXNoIter — TRIX without iterative entity-relation updates

## Motivation

TRIX (`kgfm/models/trix_entity.py`) interleaves its two NBFNets inside a single forward pass:

```
RelNet layer 0 → entity_model_1 (mid-pass) → RelNet layers 1-2 → entity_model_2 (score)
```

Upstream TRIX (`yuchengz99/TRIX`) calls this `# iterative updates: relation 1 - entity 1 - relation 1 - entity 3`. The mid-pass at `i==0` (`trix_entity.py:131-132`) is what makes TRIX's relation graph *entity-aware*: entity_model_1's NBFNet output over the entity graph is projected into per-relation `node_representations` that feed the convs at layers 1 and 2.

Without that mid-pass, `node_representations` stays as `torch.ones`, the convs' `project_relations=True` projection collapses to a relation-agnostic transform, and TRIX degenerates to ULTRA-shape behavior.

**TRIXNoIter** removes the iteration: same components as TRIX, same entity-aware relation graph, but each sub-network runs **once**, sequentially.
