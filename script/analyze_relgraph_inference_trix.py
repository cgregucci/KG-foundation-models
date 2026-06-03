#!/usr/bin/env python3
"""Per-triple TRIX-flavored relation-graph induction analysis.

Same flow as `script/analyze_relgraph_inference.py` but uses the TRIX
relation structure (`data.relation_adj`-style 4-tuples
`(r1, r2, entity, type)`) via `kgfm.relgraph_inference_trix`. Outputs:

  - runs/relgraph_visibility_trix_<name>.csv (combined summary, 11 cols
    matching the ULTRA schema)
  - runs/relgraph_inference_trix/<dataset>/per_triple_nadded.tsv

CPU-only, fork-multiprocessing, no model load.
"""

import argparse
import csv
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable, List, Tuple

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kgfm import datasets as datasets_module
from kgfm.relgraph_inference_trix import (
    Row,
    aggregate_per_bucket,
    compute_nadded_rows,
)
from kgfm.visibility import (
    QUADRANT_NAMES,
    build_seen_sets_from_doubled_graph,
    classify_test_triples,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "kg-datasets"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"

SUMMARY_FIELDS = [
    "benchmark",
    "split",
    "n_total",
    "r0_size",
    "n_full",
    "n_with_added",
    "n_added_total",
    "pct_with_added",
    "nadded_mean",
    "nadded_median",
    "nadded_std",
]

PER_TRIPLE_FIELDS = [
    "h", "t", "r", "direction", "bucket", "nadded", "r0_size", "r1_size",
]


def parse_entries(arg: str) -> List[Tuple[str, str]]:
    if not arg:
        return []
    out: List[Tuple[str, str]] = []
    for raw in arg.split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" in item:
            name, version = item.split(":", 1)
            out.append((name.strip(), version.strip()))
        else:
            out.append((item, None))
    return out


def resolve_class(name: str):
    cls = getattr(datasets_module, name, None)
    if cls is None or not callable(cls):
        raise KeyError(f"unknown dataset constructor: {name}")
    return cls


def build_dataset_cpu(name: str, version, data_root: Path):
    cls = resolve_class(name)
    kwargs = {"root": str(data_root), "device": "cpu"}
    if version is not None:
        kwargs["version"] = version
    return cls(**kwargs)


def split_index(name: str) -> int:
    return {"train": 0, "valid": 1, "test": 2}[name]


def _fmt_float(x: float) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return ""
    return f"{x:.6f}"


def write_per_triple_tsv(rows: Iterable[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(PER_TRIPLE_FIELDS) + "\n")
        for r in rows:
            f.write(
                f"{r.h}\t{r.t}\t{r.r}\t{r.direction}\t{r.bucket}\t"
                f"{r.nadded}\t{r.r0_size}\t{r.r1_size}\n"
            )


def append_summary_csv(csv_path: Path, benchmark: str, summary: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_existed = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if not file_existed:
            writer.writeheader()
        for split in ["Orig", *QUADRANT_NAMES]:
            entry = summary[split]
            r0 = entry.get("r0_size")
            writer.writerow({
                "benchmark": benchmark,
                "split": split,
                "n_total": entry["n_total"],
                "r0_size": "" if r0 is None else r0,
                "n_full": entry["n_full"],
                "n_with_added": entry["n_with_added"],
                "n_added_total": entry["n_added_total"],
                "pct_with_added": _fmt_float(entry["pct_with_added"]),
                "nadded_mean": _fmt_float(entry["nadded_mean"]),
                "nadded_median": _fmt_float(entry["nadded_median"]),
                "nadded_std": _fmt_float(entry["nadded_std"]),
            })


def print_summary(benchmark: str, summary: dict) -> None:
    r0 = summary["Orig"].get("r0_size")
    print(f"\n  {benchmark}    |R0|={r0}")
    print(f"  {'split':>6}  {'n_total':>8}  {'n_full':>8}  {'n_w_add':>8}  "
          f"{'n_add_t':>10}  {'%w_add':>8}  {'nadd_mean':>10}  "
          f"{'nadd_med':>9}  {'nadd_std':>9}")
    for split in ["Orig", *QUADRANT_NAMES]:
        e = summary[split]
        print(f"  {split:>6}  {e['n_total']:>8}  {e['n_full']:>8}  "
              f"{e['n_with_added']:>8}  {e['n_added_total']:>10}  "
              f"{e['pct_with_added']:>8.4f}  {e['nadded_mean']:>10.4f}  "
              f"{e['nadded_median']:>9.4f}  {e['nadded_std']:>9.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--datasets", required=True,
                        help="comma-separated Name[:version] entries")
    parser.add_argument("--root", default=str(DEFAULT_DATA_ROOT),
                        help="dataset root (default: %(default)s)")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--out-dir",
                        default=str(DEFAULT_RUNS_DIR / "relgraph_inference_trix"),
                        help="directory for per-triple TSVs (default: %(default)s)")
    parser.add_argument("--csv",
                        default=str(DEFAULT_RUNS_DIR / "relgraph_visibility_trix.csv"),
                        help="combined per-bucket summary CSV path (default: %(default)s)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="debug cap on test triples per dataset")
    args = parser.parse_args()

    entries = parse_entries(args.datasets)
    if not entries:
        parser.error("at least one dataset must be provided via -d")

    data_root = Path(args.root)
    out_dir = Path(args.out_dir)
    csv_path = Path(args.csv)
    split_idx = split_index(args.split)

    print(f"Summary CSV: {csv_path}")
    print(f"Per-triple TSV dir: {out_dir}")
    print(f"Split: {args.split}    Workers: {args.num_workers}    Mode: TRIX")
    print(f"Benchmarks ({len(entries)}): "
          + ", ".join(f"{n}:{v}" if v else n for n, v in entries))

    n_ok = 0
    failures: List[Tuple[str, str]] = []

    for name, version in entries:
        label = f"{name}:{version}" if version else name
        t0 = time.time()
        try:
            ds = build_dataset_cpu(name, version, data_root)
            data = ds[split_idx]

            num_base_rel = int(data.num_relations) // 2
            seen_q, seen_a = build_seen_sets_from_doubled_graph(
                data.edge_index, data.edge_type)
            tail_labels, head_labels = classify_test_triples(
                data.target_edge_index, data.target_edge_type,
                seen_q, seen_a, num_base_rel)

            rows, r0_size = compute_nadded_rows(
                data, tail_labels, head_labels,
                num_workers=args.num_workers,
                max_rows=args.max_rows,
            )

            summary = aggregate_per_bucket(rows, r0_size=r0_size)

            tsv_dir = out_dir / label.replace(":", "_")
            write_per_triple_tsv(rows, tsv_dir / "per_triple_nadded.tsv")

            append_summary_csv(csv_path, label, summary)
            print_summary(label, summary)
            n_ok += 1

        except Exception as e:
            print(f"  FAIL  {label}: {e.__class__.__name__}: {e}",
                  file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            failures.append((label, str(e)))
            continue

        finally:
            dt = time.time() - t0
            print(f"  [{label}] elapsed: {dt:.1f}s")

    print(f"\n===== {n_ok} / {len(entries)} benchmarks processed =====")
    if failures:
        print(f"Failures ({len(failures)}):")
        for label, err in failures:
            print(f"  - {label}: {err}")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
