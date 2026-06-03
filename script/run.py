"""Single-dataset entity-prediction entry point.

Dispatches Ultra / MOTIF / TRIXEntity / TRIXNoIter by `cfg.model["class"]`.
TRIXEntity and TRIXNoIter additionally need `data.relation_adj` precomputed
(see kgfm.tasks.build_relation_adj).
The `test_visibility` helper computes the four-quadrant SQSA/SQUA/UQSA/UQUA
metrics; it is invoked from script/run_many.py with --visibility-splits.
"""

import os
import sys
import math
import pprint
from itertools import islice
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kgfm import tasks, util
from kgfm.models import Ultra, MOTIF, TRIXEntity, TRIXNoIter
from kgfm.visibility import compute_quadrant_metrics


separator = ">" * 30
line = "-" * 30

# Tail-only benchmarks: head direction is not evaluated upstream.
TAIL_ONLY_BENCHMARKS = {"FB15k237_10", "FB15k237_20", "FB15k237_50"}


def build_model(cfg):
    cls = cfg.model["class"]
    if cls == "Ultra":
        return Ultra(rel_model_cfg=cfg.model.relation_model,
                     entity_model_cfg=cfg.model.entity_model)
    elif cls == "MOTIF":
        return MOTIF(rel_model_cfg=cfg.model.relation_model,
                     entity_model_cfg=cfg.model.entity_model)
    elif cls == "TRIXEntity":
        return TRIXEntity(rel_model_cfg=cfg.model.relation_model,
                          entity_model_1_cfg=cfg.model.entity_model_1,
                          entity_model_2_cfg=cfg.model.entity_model_2)
    elif cls == "TRIXNoIter":
        return TRIXNoIter(rel_model_cfg=cfg.model.relation_model,
                          entity_model_1_cfg=cfg.model.entity_model_mini,
                          entity_model_cfg=cfg.model.entity_model)
    raise ValueError(f"Unknown model class for entity prediction: {cls}")


def ensure_relation_adj(splits):
    """TRIX-family models read `data.relation_adj`; MOTIF's pre_transform doesn't build it."""
    for d in splits:
        if not hasattr(d, "relation_adj") or d.relation_adj is None:
            tasks.build_relation_adj(d)


def train_and_validate(cfg, model, train_data, valid_data, device, logger, filtered_data=None, batch_per_epoch=None,
                       zero_shot_fallback=False):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler)

    batch_per_epoch = batch_per_epoch or len(train_loader)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(util.trainable_parameters(model), **cfg.optimizer)
    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    step = math.ceil(cfg.train.num_epoch / 10)
    if batch_per_epoch == "null":
        batch_per_epoch = None
    # Optional zero-shot baseline on valid as a candidate for best-epoch
    # selection. This is upstream TRIX's behavior (src/run_entity.py:53): if
    # no finetune epoch beats the pretrained model's zero-shot valid score,
    # best_epoch stays -1 and the pretrained ckpt is reloaded at the end of
    # the function (analogous to upstream lines 340-342).
    # OFF by default here; enable with run_many.py --zero-shot-fallback to
    # reproduce the TRIX fine-tuning setting. When off, best_result is seeded
    # at -inf so any finetune epoch wins and the pretrained ckpt is never
    # reloaded — "pure" finetune numbers that can legitimately regress vs
    # zero-shot.
    if zero_shot_fallback:
        best_result = test(cfg, model, valid_data, filtered_data=filtered_data, device=device, logger=logger)
    else:
        best_result = float("-inf")
    best_epoch = -1
    batch_id = 0
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            for batch in islice(train_loader, batch_per_epoch):
                batch = tasks.negative_sampling(train_data, batch, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative)
                pred = parallel_model(train_data, batch)
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)
                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        result = test(cfg, model, valid_data, filtered_data=filtered_data, device=device, logger=logger)
        if result > best_result:
            best_result = result
            best_epoch = epoch

    if best_epoch != -1:
        if rank == 0:
            logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
        state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
        model.load_state_dict(state["model"])
    elif zero_shot_fallback:
        if rank == 0:
            logger.warning("Zero-shot val was best — reloading original pretrained checkpoint")
        if "checkpoint" in cfg and cfg.checkpoint is not None:
            state = torch.load(os.path.expanduser(cfg.checkpoint), map_location=device)
            model.load_state_dict(state["model"])
    # else: no fallback, no improvement — keep the in-memory model as is.
    # This is the default path (fallback off): best_result starts at -inf so
    # best_epoch >= 1 always, meaning we never actually land here in practice.
    util.synchronize()


@torch.no_grad()
def test(cfg, model, test_data, device, logger, filtered_data=None, return_metrics=False,
         precomputed_rel_emb=None):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    model.eval()
    rankings = []
    num_negatives = []
    tail_rankings, num_tail_negs = [], []
    for batch in test_loader:
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        t_pred = model(test_data, t_batch, precomputed_rel_emb=precomputed_rel_emb)
        h_pred = model(test_data, h_batch, precomputed_rel_emb=precomputed_rel_emb)

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, _ = batch.t()
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

        tail_rankings += [t_ranking]
        num_tail_negs += [num_t_negative]

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)

    tail_ranking = torch.cat(tail_rankings)
    num_tail_neg = torch.cat(num_tail_negs)
    all_size_t = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_t[rank] = len(tail_ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_t, op=dist.ReduceOp.SUM)

    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative

    cum_size_t = all_size_t.cumsum(0)
    all_ranking_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_ranking_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = tail_ranking
    all_num_negative_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_num_negative_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = num_tail_neg
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_ranking_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative_t, op=dist.ReduceOp.SUM)

    metrics = {}
    if rank == 0:
        for metric in cfg.task.metric:
            if "-tail" in metric:
                _metric_name, direction = metric.split("-")
                if direction != "tail":
                    raise ValueError("Only tail metric is supported in this mode")
                _ranking = all_ranking_t
                _num_neg = all_num_negative_t
            else:
                _ranking = all_ranking
                _num_neg = all_num_negative
                _metric_name = metric

            if _metric_name == "mr":
                score = _ranking.float().mean()
            elif _metric_name == "mrr":
                score = (1 / _ranking.float()).mean()
            elif _metric_name.startswith("hits@"):
                values = _metric_name[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    fp_rate = (_ranking - 1).float() / _num_neg
                    score = 0
                    for i in range(threshold):
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (_ranking <= threshold).float().mean()
            logger.warning("%s: %g" % (metric, score))
            metrics[metric] = score
    mrr = (1 / all_ranking.float()).mean()

    return mrr if not return_metrics else metrics


@torch.no_grad()
def test_visibility(cfg, model, test_data, device, logger,
                    filtered_data=None, tail_labels=None, head_labels=None,
                    precomputed_rel_emb=None, return_per_triple=False):
    """Run filtered evaluation and return per-quadrant SQSA/SQUA/UQSA/UQUA metrics.

    Ported verbatim from MOTIF-main/script/run.py:test_visibility — model-agnostic
    (works for Ultra / MOTIF / TRIXEntity / TRIXNoIter since `compute_ranking` shape is the same).

    When `return_per_triple=True`, additionally returns `(global_tail,
    global_head)` — the 1-based filtered ranks aligned to `test_data`'s
    `target_edge_index` order (head empty for tail-only benchmarks). That
    alignment only holds single-process, so this requires `world_size == 1`.
    """
    world_size = util.get_world_size()
    rank = util.get_rank()

    if return_per_triple and world_size != 1:
        raise RuntimeError(
            "return_per_triple requires world_size == 1 so that global_tail/"
            f"global_head stay aligned to target_edge_index order (got {world_size})")

    tail_only = str(cfg.dataset["class"]) in TAIL_ONLY_BENCHMARKS

    test_triplets = torch.cat([test_data.target_edge_index,
                               test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank,
                                            shuffle=False)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size,
                                        sampler=sampler)

    model.eval()
    all_tail_ranks = []
    all_head_ranks = []
    for batch in test_loader:
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        t_pred = model(test_data, t_batch, precomputed_rel_emb=precomputed_rel_emb)

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, _ = batch.t()
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        all_tail_ranks.append(t_ranking)

        if not tail_only:
            h_pred = model(test_data, h_batch, precomputed_rel_emb=precomputed_rel_emb)
            h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
            all_head_ranks.append(h_ranking)

    tail_ranking = torch.cat(all_tail_ranks)

    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(tail_ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)

    cum_size = all_size.cumsum(0)
    total = all_size.sum().item()
    start = (cum_size[rank] - all_size[rank]).item()
    end = cum_size[rank].item()

    global_tail = torch.zeros(total, dtype=torch.long, device=device)
    global_tail[start:end] = tail_ranking
    if world_size > 1:
        dist.all_reduce(global_tail, op=dist.ReduceOp.SUM)

    if tail_only:
        global_head = torch.empty(0, dtype=torch.long, device=device)
    else:
        head_ranking = torch.cat(all_head_ranks)
        global_head = torch.zeros(total, dtype=torch.long, device=device)
        global_head[start:end] = head_ranking
        if world_size > 1:
            dist.all_reduce(global_head, op=dist.ReduceOp.SUM)

    results = {}
    if rank == 0:
        tail_labels_d = tail_labels.to(device)
        if tail_only:
            head_labels_d = torch.empty(0, dtype=tail_labels.dtype, device=device)
        else:
            head_labels_d = head_labels.to(device)
        results = compute_quadrant_metrics(global_tail, global_head,
                                           tail_labels_d, head_labels_d)

        for split_name, m in results.items():
            logger.warning(
                "%s  mrr: %.4f  h@1: %.4f  h@3: %.4f  h@10: %.4f  "
                "(n_tail=%d  n_head=%d  n_combined=%d)",
                split_name.ljust(5), m["mrr"], m["hits@1"], m["hits@3"],
                m["hits@10"], m["n_tail"], m["n_head"], m["n_combined"])

    if return_per_triple:
        return results, (global_tail, global_head)
    return results


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    task_name = cfg.task["name"]
    device = util.get_device(cfg)
    # AristoV4-MOTIF: load on CPU so the 215 GB relation_hypergraph doesn't go
    # straight to GPU. Sharded by motif type below before .to(device).
    aristov4_motif = (cfg.dataset["class"] == "AristoV4" and cfg.model["class"] == "MOTIF")
    dataset = util.build_dataset(cfg, device="cpu" if aristov4_motif else device)

    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    if aristov4_motif:
        from kgfm.motif_aristo_shim import shard_hypergraph_inplace
        for d in (train_data, valid_data, test_data):
            shard_hypergraph_inplace(d)
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

    if cfg.model["class"] in {"TRIXEntity", "TRIXNoIter"}:
        ensure_relation_adj([train_data, valid_data, test_data])

    model = build_model(cfg)

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    model = model.to(device)

    if cfg.model.get("freeze_except_trainable_prefixes", False):
        prefixes = util.configure_trainability(
            model, cfg.model.get("trainable_module_prefixes", [])
        )
        if util.get_rank() == 0:
            logger.warning(f"Trainability configured. Unfrozen prefixes: {prefixes}")
            util.log_trainability_summary(model, logger_obj=logger)

    # Dispatch on dataset class — TRIX configs use task.name="InductiveInference"
    # for transductive datasets too, which would otherwise misroute the filter.
    dataset_class = str(cfg.dataset["class"])
    is_inductive_family = (
        ("Inductive" in dataset_class)
        or ("ILPC" in dataset_class)
        or ("Ingram" in dataset_class)
        or ("WikiTopics" in dataset_class)
        or ("HM" in dataset_class)
        or ("Metafam" in dataset_class)
        or ("FBNELL" in dataset_class)
    )
    if is_inductive_family:
        if ("ILPC" in dataset_class) or ("Ingram" in dataset_class):
            full_inference_edges = torch.cat([valid_data.edge_index, valid_data.target_edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([valid_data.edge_type, valid_data.target_edge_type, test_data.target_edge_type])
            test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)
            val_filtered_data = test_filtered_data
        else:
            full_inference_edges = torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([test_data.edge_type, test_data.target_edge_type])
            test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)
            val_filtered_data = Data(
                edge_index=torch.cat([train_data.edge_index, valid_data.target_edge_index], dim=1),
                edge_type=torch.cat([train_data.edge_type, valid_data.target_edge_type])
            )
    else:
        filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=dataset[0].num_nodes)
        val_filtered_data = test_filtered_data = filtered_data

    val_filtered_data = val_filtered_data.to(device)
    test_filtered_data = test_filtered_data.to(device)

    train_and_validate(cfg, model, train_data, valid_data, filtered_data=val_filtered_data, device=device, batch_per_epoch=cfg.train.batch_per_epoch, logger=logger)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    test(cfg, model, valid_data, filtered_data=val_filtered_data, device=device, logger=logger)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test(cfg, model, test_data, filtered_data=test_filtered_data, device=device, logger=logger)
