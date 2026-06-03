"""Multi-dataset TRIX relation-prediction runner.

Mirrors script/run_many.py but uses the relation-side eval from
script/run_relation.py. No visibility-splits integration: the visibility
classifier is entity-side (it labels test triples by whether `(h, r)` and
`(r, t)` were seen) and does not have a relation-prediction analog.
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
from kgfm import tasks, util
from kgfm.models import TRIXRelation
from script.run_relation import train_and_validate, test


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
    parser.add_argument("-d", "--datasets", help="target datasets", type=str, required=True)
    parser.add_argument("-reps", "--repeats", default=1, type=int)
    parser.add_argument("--test-only", action='store_true')
    args, unparsed = parser.parse_known_args()

    datasets = args.datasets.split(",")
    path = os.path.dirname(os.path.expanduser(__file__))
    results_dir = os.path.join(path, "..", "runs")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir,
                                f"trix_relation_results_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv")

    for graph in datasets:
        ds, version = graph.split(":") if ":" in graph else (graph, None)
        for i in range(args.repeats):
            seed = seeds[i] if i < len(seeds) else random.randint(0, 10000)
            print(f"Running on {graph}, iteration {i+1} / {args.repeats}, seed: {seed}")

            vars = util.detect_variables(args.config)
            tmp_parser = argparse.ArgumentParser()
            for var in vars:
                tmp_parser.add_argument("--%s" % var)
            vars = tmp_parser.parse_known_args(unparsed)[0]
            vars = {k: util.literal_eval(v) for k, v in vars._get_kwargs()}

            vars.setdefault('epochs', 0)
            vars.setdefault('bpe', 'null')
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
                logger.warning("Config: %s\n%s", args.config, pprint.pformat(cfg))

            task_name = cfg.task["name"]
            device = util.get_device(cfg)
            dataset = util.build_dataset(cfg, device=device)

            train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
            train_data = train_data.to(device)
            valid_data = valid_data.to(device)
            test_data = test_data.to(device)

            for d in (train_data, valid_data, test_data):
                if not hasattr(d, "relation_adj") or d.relation_adj is None:
                    tasks.build_relation_adj(d)

            model = TRIXRelation(
                rel_model_cfg=cfg.model.relation_model,
                entity_model_cfg=cfg.model.entity_model,
                trix_cfg=cfg.model.trix,
            )
            if "checkpoint" in cfg and cfg.checkpoint is not None:
                state = torch.load(os.path.expanduser(cfg.checkpoint), map_location="cpu")
                model.load_state_dict(state["model"])
            model = model.to(device)

            dataset_class = str(cfg.dataset["class"])
            is_inductive_family = (
                ("Inductive" in dataset_class) or ("ILPC" in dataset_class)
                or ("Ingram" in dataset_class) or ("WikiTopics" in dataset_class)
                or ("HM" in dataset_class) or ("Metafam" in dataset_class)
                or ("FBNELL" in dataset_class)
            )
            if is_inductive_family:
                full_inference_edges = torch.cat(
                    [test_data.edge_index, test_data.target_edge_index], dim=1)
                full_inference_etypes = torch.cat(
                    [test_data.edge_type, test_data.target_edge_type])
                test_filtered_data = Data(edge_index=full_inference_edges,
                                          edge_type=full_inference_etypes,
                                          num_nodes=test_data.num_nodes)
                val_filtered_data = Data(
                    edge_index=torch.cat([train_data.edge_index, valid_data.target_edge_index], dim=1),
                    edge_type=torch.cat([train_data.edge_type, valid_data.target_edge_type]))
            else:
                filtered_data = Data(edge_index=dataset._data.target_edge_index,
                                     edge_type=dataset._data.target_edge_type,
                                     num_nodes=dataset[0].num_nodes)
                val_filtered_data = test_filtered_data = filtered_data

            val_filtered_data = val_filtered_data.to(device)
            test_filtered_data = test_filtered_data.to(device)

            if not args.test_only:
                train_and_validate(cfg, model, train_data, valid_data,
                                   filtered_data=val_filtered_data, device=device,
                                   batch_per_epoch=cfg.train.batch_per_epoch, logger=logger)
                if util.get_rank() == 0:
                    logger.warning(separator + "\nEvaluate on valid")
                test(cfg, model, valid_data, filtered_data=val_filtered_data,
                     device=device, logger=logger)
            if util.get_rank() == 0:
                logger.warning(separator + "\nEvaluate on test")
            metrics = test(cfg, model, test_data, filtered_data=test_filtered_data,
                           return_metrics=True, device=device, logger=logger)

            metrics = {k: (v.item() if hasattr(v, "item") else v) for k, v in metrics.items()}
            metrics['dataset'] = graph
            with open(results_file, "a", newline='') as csv_file:
                fieldnames = ['dataset'] + [k for k in metrics if k != 'dataset']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=',')
                if csv_file.tell() == 0:
                    writer.writeheader()
                writer.writerow(metrics)
