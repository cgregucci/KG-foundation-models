"""Multi-dataset entity-prediction runner with optional visibility splits.

Adapted from MOTIF-main/script/run_many.py with --visibility-splits / --vis-csv
/ --test-only flags. Dispatches Ultra / MOTIF / TRIXEntity / TRIXNoIter by
cfg.model["class"].
"""

import os
import sys
import csv
import time
import pprint
import argparse
import random
import copy
import torch
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kgfm import util
from kgfm.models import Ultra, MOTIF, TRIXEntity, TRIXNoIter
from script.run import (train_and_validate, test, test_visibility, build_model,
                         ensure_relation_adj)
from kgfm.visibility import (
    build_seen_sets_from_doubled_graph,
    classify_test_triples,
    QUADRANT_NAMES,
)


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

default_train_config = {
    "CoDExSmall": (10, 4000), "CoDExMedium": (10, 8000), "CoDExLarge": (10, 4000),
    "FB15k237": (10, 4000), "WN18RR": (10, 2000), "YAGO310": (10, 4000),
    "DBpedia100k": (10, 2000), "AristoV4": (10, 2000),
    "ConceptNet100k": (10, 2000), "ATOMIC": (10, 1000),
    "NELL995": (10, 2000), "Hetionet": (10, 4000),
    "WDsinger": (10, 2000), "FB15k237_10": (10, 2000),
    "FB15k237_20": (10, 2000), "FB15k237_50": (10, 4000), "NELL23k": (10, 4000),
    "FB15k237Inductive": (10, 'null'), "WN18RRInductive": (10, 'null'),
    "NELLInductive": (10, 'null'),
    "ILPC2022SmallInductive": (10, 'null'), "ILPC2022LargeInductive": (10, 1000),
    "NLIngram": (10, 'null'), "FBIngram": (10, 'null'), "WKIngram": (10, 'null'),
    "WikiTopicsMT1": (10, 'null'), "WikiTopicsMT2": (10, 'null'),
    "WikiTopicsMT3": (10, 'null'), "WikiTopicsMT4": (10, 'null'),
    "Metafam": (10, 'null'), "FBNELL": (10, 'null'),
    "HM": (10, 1000),
}


separator = ">" * 30
line = "-" * 30


def set_seed(seed):
    random.seed(seed + util.get_rank())
    torch.manual_seed(seed + util.get_rank())
    torch.cuda.manual_seed(seed + util.get_rank())
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    seeds = [1024, 42, 1337, 512, 256]

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-d", "--datasets", help="target datasets",
                        default='FB15k237Inductive:v1,NELLInductive:v4', type=str, required=True)
    parser.add_argument("-reps", "--repeats", help="number of times to repeat each exp", default=1, type=int)
    parser.add_argument("-ft", "--finetune", help="finetune the checkpoint on the specified datasets", action='store_true')
    parser.add_argument("--ft-epochs", type=int, default=None,
                        help="Override per-dataset finetune epochs from default_finetuning_config "
                             "with this uniform value (e.g. 3 to match upstream TRIX's --epochs 3).")
    parser.add_argument("--ft-bpe", type=str, default=None,
                        help="Override per-dataset finetune batch_per_epoch with this uniform value "
                             "(e.g. 1000 to match upstream TRIX's --bpe 1000; 'null' for full epoch).")
    parser.add_argument("--zero-shot-fallback", action='store_true',
                        help="Enable upstream TRIX's zero-shot val baseline + pretrained reload (OFF by "
                             "default). Required to reproduce the TRIX fine-tuning setting; without it, "
                             "fine-tune numbers can legitimately regress vs zero-shot.")
    parser.add_argument("-tr", "--train", help="train the model from scratch", action='store_true')
    parser.add_argument("--visibility-splits", action='store_true',
                        help="Run SQSA/SQUA/UQSA/UQUA visibility evaluation")
    parser.add_argument("--vis-csv", type=str, default=None,
                        help="Path for visibility-splits CSV (default: auto-generated)")
    parser.add_argument("--test-only", action='store_true',
                        help="Skip training and validation, run test evaluation only")
    parser.add_argument("--rel-emb-cache", type=str, default=None,
                        help="Path to a precomputed [R, R, D] relation embedding "
                             "table (produced by script/precompute_motif_relation_emb.py). "
                             "When set with AristoV4-MOTIF + --test-only, the eval "
                             "skips the 215 GB relation_hypergraph cache entirely.")
    parser.add_argument("--vis-out-dir", type=str, default=None,
                        help="Directory for per-benchmark visibility CSVs "
                             "(<dir>/<dataset>[_<version>].csv). Overrides --vis-csv.")
    parser.add_argument("--vis-combined-csv", type=str, default=None,
                        help="Also append every (dataset,split) row to this single "
                             "unified CSV (in addition to per-benchmark files). Safe "
                             "here: one sbatch = one process = sequential dataset loop.")
    parser.add_argument("--dump-per-triple", type=str, default=None,
                        help="Directory for per-benchmark per-triple rank dumps "
                             "(<dir>/<dataset>[_<version>].csv with columns "
                             "direction,quadrant,rank,gold_degree). Implies "
                             "--visibility-splits. Single-GPU only.")
    args, unparsed = parser.parse_known_args()

    if args.dump_per_triple:
        args.visibility_splits = True

    datasets = args.datasets.split(",")
    path = os.path.dirname(os.path.expanduser(__file__))
    results_dir = os.path.join(path, "..", "runs")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv")

    if args.visibility_splits:
        if args.vis_out_dir:
            # Per-benchmark CSVs: one file per (dataset, version), computed
            # inside the dataset loop. vis_csv_file is set there.
            os.makedirs(args.vis_out_dir, exist_ok=True)
            vis_csv_file = None
        elif args.vis_csv:
            vis_csv_file = args.vis_csv
        else:
            vars_tmp = util.detect_variables(args.config)
            parser_tmp = argparse.ArgumentParser()
            for var in vars_tmp:
                parser_tmp.add_argument("--%s" % var)
            vars_tmp = parser_tmp.parse_known_args(unparsed)[0]
            ckpt_name = os.path.basename(getattr(vars_tmp, 'ckpt', '') or '').replace('.pth', '') or 'no_ckpt'
            vis_csv_file = os.path.join(results_dir,
                                        f"visibility_{ckpt_name}_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv")
        if vis_csv_file is not None:
            vis_csv_dir = os.path.dirname(vis_csv_file)
            if vis_csv_dir and not os.path.exists(vis_csv_dir):
                os.makedirs(vis_csv_dir, exist_ok=True)
        if args.vis_combined_csv:
            combined_dir = os.path.dirname(args.vis_combined_csv)
            if combined_dir and not os.path.exists(combined_dir):
                os.makedirs(combined_dir, exist_ok=True)
        if args.dump_per_triple:
            os.makedirs(args.dump_per_triple, exist_ok=True)

    for graph in datasets:
        ds, version = graph.split(":") if ":" in graph else (graph, None)
        for i in range(args.repeats):
            seed = seeds[i] if i < len(seeds) else random.randint(0, 10000)
            print(f"Running on {graph}, iteration {i+1} / {args.repeats}, seed: {seed}")

            vars = util.detect_variables(args.config)
            parser = argparse.ArgumentParser()
            for var in vars:
                parser.add_argument("--%s" % var)
            vars = parser.parse_known_args(unparsed)[0]
            vars = {k: util.literal_eval(v) for k, v in vars._get_kwargs()}

            if args.finetune:
                epochs, batch_per_epoch = default_finetuning_config[ds]
                if args.ft_epochs is not None:
                    epochs = args.ft_epochs
                if args.ft_bpe is not None:
                    batch_per_epoch = util.literal_eval(args.ft_bpe)
            elif args.train:
                epochs, batch_per_epoch = default_train_config[ds]
            else:
                epochs, batch_per_epoch = 0, 'null'
            vars['epochs'] = epochs
            vars['bpe'] = batch_per_epoch
            vars['dataset'] = ds
            if version is not None:
                vars['version'] = version
            cfg = util.load_config(args.config, context=vars)

            root_dir = os.path.expanduser(cfg.output_dir)
            os.makedirs(root_dir, exist_ok=True)
            os.chdir(root_dir)
            working_dir = util.create_working_directory(cfg)
            set_seed(seed)

            logger = util.get_root_logger()
            if util.get_rank() == 0:
                logger.warning("Random seed: %d" % seed)
                logger.warning("Config file: %s" % args.config)
                logger.warning(pprint.pformat(cfg))

            task_name = cfg.task["name"]
            device = util.get_device(cfg)
            # AristoV4-MOTIF: load on CPU so the 215 GB relation_hypergraph doesn't
            # go straight to GPU. Sharded by motif type below before .to(device).
            motif_model = cfg.model["class"] == "MOTIF"
            aristov4_motif = motif_model and cfg.dataset["class"] == "AristoV4"
            # When --rel-emb-cache is provided we skip the relation_hypergraph
            # cache entirely (AristoV4-MOTIF: 215 GB → 534 MB) and feed the
            # precomputed [R, R, D] table to MOTIF.forward via precomputed_rel_emb.
            # Gating is MOTIF-generic — non-AristoV4 use is for validating that
            # the precomputed path matches the legacy hypergraph path.
            use_precomputed_rel_emb = (
                motif_model and args.test_only and args.rel_emb_cache is not None
            )
            cache_key_override = "relation_graph" if use_precomputed_rel_emb else None
            shard_aristov4 = aristov4_motif and not use_precomputed_rel_emb
            dataset = util.build_dataset(
                cfg,
                device="cpu" if shard_aristov4 else device,
                cache_key_override=cache_key_override,
            )

            train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
            if shard_aristov4:
                from kgfm.motif_aristo_shim import shard_hypergraph_inplace
                for d in (train_data, valid_data, test_data):
                    shard_hypergraph_inplace(d)
            train_data = train_data.to(device)
            valid_data = valid_data.to(device)
            test_data = test_data.to(device)

            precomputed_rel_emb = None
            if use_precomputed_rel_emb:
                if util.get_rank() == 0:
                    logger.warning(f"loading precomputed relation embeddings: {args.rel_emb_cache}")
                blob = torch.load(args.rel_emb_cache, map_location=device)
                precomputed_rel_emb = blob["relation_embeddings"].to(device)
                R_table = int(blob["num_relations"])
                R_data = int(test_data.num_relations)
                if R_table != R_data:
                    raise ValueError(
                        f"precomputed table has num_relations={R_table} but "
                        f"test_data has num_relations={R_data}; mismatch likely "
                        f"means the table was built for a different dataset/checkpoint"
                    )

            if cfg.model["class"] in {"TRIXEntity", "TRIXNoIter"}:
                ensure_relation_adj([train_data, valid_data, test_data])

            if "fast_test" in cfg.train:
                num_val_edges = cfg.train.fast_test
                if util.get_rank() == 0:
                    logger.warning(f"Fast evaluation on {num_val_edges} samples in validation")
                graph_short = copy.deepcopy(valid_data)
                mask = torch.randperm(graph_short.target_edge_index.shape[1])[:num_val_edges]
                graph_short.target_edge_index = graph_short.target_edge_index[:, mask]
                graph_short.target_edge_type = graph_short.target_edge_type[mask]
                short_valid = graph_short.to(device)

            model = build_model(cfg)

            if "checkpoint" in cfg and cfg.checkpoint is not None:
                state = torch.load(os.path.expanduser(cfg.checkpoint), map_location="cpu")
                model.load_state_dict(state["model"])

            model = model.to(device)

            if cfg.model.get("freeze_except_trainable_prefixes", False):
                prefixes = util.configure_trainability(
                    model, cfg.model.get("trainable_module_prefixes", [])
                )
                if util.get_rank() == 0:
                    logger.warning(f"Trainability configured. Unfrozen prefixes: {prefixes}")
                    util.log_trainability_summary(model, logger_obj=logger)

            # Dispatch the filter-graph construction on the dataset class
            # name rather than `cfg.task.name`. TRIX configs declare
            # `task.name: InductiveInference` even for transductive
            # datasets like FB15k237, which would otherwise route us into
            # the inductive filter branch and silently apply the wrong
            # filtered ranking (≈4 MRR points off on FB15k237).
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

            if not args.test_only:
                train_and_validate(cfg, model, train_data,
                                   valid_data if "fast_test" not in cfg.train else short_valid,
                                   filtered_data=val_filtered_data, batch_per_epoch=batch_per_epoch,
                                   device=device, logger=logger,
                                   zero_shot_fallback=args.zero_shot_fallback)
                if util.get_rank() == 0:
                    logger.warning(separator)
                    logger.warning("Evaluate on valid")
                test(cfg, model, valid_data, filtered_data=val_filtered_data, device=device, logger=logger,
                     precomputed_rel_emb=precomputed_rel_emb)
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Evaluate on test")
            metrics = test(cfg, model, test_data, filtered_data=test_filtered_data,
                           return_metrics=True, device=device, logger=logger,
                           precomputed_rel_emb=precomputed_rel_emb)

            metrics = {k: v.item() for k, v in metrics.items()}
            metrics['dataset'] = graph
            with open(results_file, "a", newline='') as csv_file:
                fieldnames = ['dataset'] + list(metrics.keys())[:-1]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=',')
                if csv_file.tell() == 0:
                    writer.writeheader()
                writer.writerow(metrics)

            # --- visibility splits evaluation ---
            if args.visibility_splits:
                num_base_rel = test_data.num_relations // 2

                seen_q, seen_a = build_seen_sets_from_doubled_graph(
                    test_data.edge_index, test_data.edge_type)

                tail_labels, head_labels = classify_test_triples(
                    test_data.target_edge_index,
                    test_data.target_edge_type,
                    seen_q, seen_a, num_base_rel)

                if util.get_rank() == 0:
                    logger.warning(separator)
                    logger.warning("Visibility splits on test")
                vis_out = test_visibility(
                    cfg, model, test_data,
                    filtered_data=test_filtered_data,
                    device=device, logger=logger,
                    tail_labels=tail_labels,
                    head_labels=head_labels,
                    precomputed_rel_emb=precomputed_rel_emb,
                    return_per_triple=bool(args.dump_per_triple))
                if args.dump_per_triple:
                    vis_results, (global_tail, global_head) = vis_out
                else:
                    vis_results = vis_out

                # --- per-triple rank dump (single-GPU; aligned to
                # target_edge_index order, same as tail/head_labels) ---
                if args.dump_per_triple and util.get_rank() == 0 and vis_results:
                    base = ds if version is None else f"{ds}_{version}"
                    deg = torch.bincount(
                        test_data.edge_index[0],
                        minlength=test_data.num_nodes).cpu()
                    gold_tail = test_data.target_edge_index[1].cpu()
                    gold_head = test_data.target_edge_index[0].cpu()
                    tl = tail_labels.cpu()
                    hl = head_labels.cpu()
                    gt = global_tail.cpu()
                    gh = global_head.cpu()
                    tail_only = gh.numel() == 0
                    dump_path = os.path.join(args.dump_per_triple, f"{base}.csv")
                    with open(dump_path, "w", newline='') as df:
                        dw = csv.writer(df)
                        dw.writerow(["direction", "quadrant", "rank", "gold_degree"])
                        for i in range(gt.numel()):
                            dw.writerow(["tail", QUADRANT_NAMES[int(tl[i])],
                                         int(gt[i]), int(deg[gold_tail[i]])])
                        if not tail_only:
                            for i in range(gh.numel()):
                                dw.writerow(["head", QUADRANT_NAMES[int(hl[i])],
                                             int(gh[i]), int(deg[gold_head[i]])])
                    logger.warning(f"per-triple dump → {dump_path} "
                                   f"({gt.numel()} tail + {gh.numel()} head rows)")

                if util.get_rank() == 0 and vis_results:
                    if args.vis_out_dir:
                        base = ds if version is None else f"{ds}_{version}"
                        vis_csv_file = os.path.join(args.vis_out_dir, f"{base}.csv")
                    ckpt_name = cfg.checkpoint if ("checkpoint" in cfg and cfg.checkpoint) else "none"
                    vis_fieldnames = [
                        "checkpoint", "benchmark_group", "dataset", "version",
                        "split", "mrr", "hits@1", "hits@3", "hits@10",
                        "n_tail", "n_head", "n_combined",
                    ]
                    benchmark_group = "inductive" if is_inductive_family else "transductive"
                    ds_version = version if version is not None else ""
                    vis_rows = [
                        {
                            "checkpoint": ckpt_name,
                            "benchmark_group": benchmark_group,
                            "dataset": ds,
                            "version": ds_version,
                            "split": split_name,
                            "mrr": f"{vis_results[split_name]['mrr']:.6f}",
                            "hits@1": f"{vis_results[split_name]['hits@1']:.6f}",
                            "hits@3": f"{vis_results[split_name]['hits@3']:.6f}",
                            "hits@10": f"{vis_results[split_name]['hits@10']:.6f}",
                            "n_tail": vis_results[split_name]["n_tail"],
                            "n_head": vis_results[split_name]["n_head"],
                            "n_combined": vis_results[split_name]["n_combined"],
                        }
                        for split_name in ["Orig"] + QUADRANT_NAMES
                    ]
                    # Per-benchmark file (resilient to partial SLURM failures)
                    # plus, optionally, one unified CSV for the whole sweep.
                    vis_targets = [vis_csv_file]
                    if args.vis_combined_csv:
                        vis_targets.append(args.vis_combined_csv)
                    for target in vis_targets:
                        with open(target, "a", newline='') as vf:
                            writer = csv.DictWriter(vf, fieldnames=vis_fieldnames, delimiter=',')
                            if vf.tell() == 0:
                                writer.writeheader()
                            for row in vis_rows:
                                writer.writerow(row)
