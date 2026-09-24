# ULTRA random-frozen-backbone — only the score-function MLP is trained

## What this experiment is

ULTRA pretrained from scratch on FB15k237 + WN18RR + CoDExMedium, but with the
backbone **frozen at random init** and only the entity-side score function
(`entity_model.mlp`) updated.

Reference configs (only difference vs. plain `ULTRA_pretrain_3g.yaml` is the
three frozen-backbone fields):

```
config/ultra/transductive/ULTRA_random_frozen_backbone_pretrain_3g.yaml
config/ultra/{transductive,inductive}/ULTRA_random_frozen_backbone_inference.yaml
```

## What is trained vs frozen

`trainable_module_prefixes: ["entity_model.mlp"]` matches exactly four state-dict
tensors — the two `Linear` layers of the readout that turns per-entity NBFNet
features into a scalar score:

| state-dict key                    | shape       | params |
|-----------------------------------|-------------|-------:|
| `entity_model.mlp.0.weight`       | (128, 128)  | 16 384 |
| `entity_model.mlp.0.bias`         | (128,)      |    128 |
| `entity_model.mlp.2.weight`       | (1, 128)    |    128 |
| `entity_model.mlp.2.bias`         | (1,)        |      1 |
|                                   |             | **16 641** |

That is **9.86 % of the model's 168 705 parameters**. The remaining 90.14 %
(both NBFNets, all `GeneralizedRelationalConv` layers, all `LayerNorm`s, the
relation embeddings) stay at their random init for the entire run.

PyTorch's `nn.LayerNorm` is initialized to gain `1.0` and bias `0.0`. After any
gradient update those drift. Comparing the same tensor in the fully-trained
`ckpts/ultra/ultra_3g.pth` vs. the random-frozen ckpt shows the diagnostic cleanly:

| param                                          | `ultra_3g.pth` (trained) | `ultra_3g_randfrozen.pth` |
|------------------------------------------------|--------------------------|---------------------------|
| `relation_model.layers.0.layer_norm.weight`    | mean 0.807, std 0.154    | **mean 1.0000, std 0.0000** |
| `relation_model.layers.0.layer_norm.bias`      | mean -0.138, std 0.100   | **mean 0.0000, std 0.0000** |
| `relation_model.layers.5.layer_norm.weight`    | mean 0.788, std 0.113    | **mean 1.0000, std 0.0000** |
| `relation_model.layers.0.linear.weight` (std)  | 0.068                    | 0.051 ≈ Kaiming init       |
| `entity_model.mlp.0.weight` (std)              | 0.064                    | 0.071 (the only trained)   |

LayerNorms sitting at exactly init values across all six relation_model layers
(and the same in entity_model) — and 0/82 state-dict keys identical between the
two ckpts overall — confirm the freeze: the backbone did not move at all,
only `entity_model.mlp.*` did.

The pretrain log shows the same:

```
Trainability configured. Unfrozen prefixes: ['entity_model.mlp']
Trainable parameters: 16641 / 168705
```

logged before optimizer construction, so the optimizer was built only over
those 16 641 params.

## Why the result isn't random-chance

10-epoch random-frozen pretrain → ULTRA_random_frozen_backbone_inference eval:

| Benchmark        | random-frozen MRR | trained ULTRA MRR (published) |
|------------------|-------------------|-------------------------------|
| FB15k237         | 0.226             | ≈ 0.36                        |
| WN18RR           | 0.391             | ≈ 0.48                        |
| HM (indigo, ind) | 0.331             | ≈ 0.40                        |

Random guessing on FB15k237 is ≈ 1/14 541 ≈ 6.9 × 10⁻⁵. The 0.226 the frozen
model reaches — about 63 % of the fully trained ULTRA.

## How to reproduce / verify

```bash
# 1. pretrain (already saved as ckpts/ultra/ultra_3g_randfrozen.pth)
python script/pretrain.py -c config/ultra/transductive/ULTRA_random_frozen_backbone_pretrain_3g.yaml --gpus '[0]'

# 2. eval (transductive / inductive)
python script/run.py -c config/ultra/transductive/ULTRA_random_frozen_backbone_inference.yaml --gpus '[0]' --ckpt ckpts/ultra/ultra_3g_randfrozen.pth --dataset FB15k237
python script/run.py -c config/ultra/inductive/ULTRA_random_frozen_backbone_inference.yaml    --gpus '[0]' --ckpt ckpts/ultra/ultra_3g_randfrozen.pth --dataset FB15k237Inductive --version v1

```