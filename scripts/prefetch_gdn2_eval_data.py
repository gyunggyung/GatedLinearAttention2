#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("NUMEXPR_MAX_THREADS", "256")

from datasets import load_dataset
from lm_eval.tasks.ruler import niah_utils
from transformers import AutoTokenizer


DATASETS = [
    ("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1"),
    ("EleutherAI/lambada_openai", "default"),
    ("baber/piqa", None),
    ("Rowan/hellaswag", None),
    ("allenai/winogrande", "winogrande_xl"),
    ("allenai/ai2_arc", "ARC-Easy"),
    ("allenai/ai2_arc", "ARC-Challenge"),
    ("allenai/openbookqa", "main"),
    ("aps/super_glue", "boolq"),
    ("hazyresearch/based-swde-v2", "default"),
    ("hazyresearch/based-fda", "default"),
    ("hazyresearch/based-squad", "default"),
    ("mandarjoshi/trivia_qa", "rc.nocontext"),
    ("google-research-datasets/nq_open", None),
    ("EleutherAI/drop", None),
]

RULER_TASKS = {
    "niah_single_1": niah_utils.niah_single_1,
    "niah_single_2": niah_utils.niah_single_2,
    "niah_single_3": niah_utils.niah_single_3,
    "niah_multikey_1": niah_utils.niah_multikey_1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch datasets used by the GDN-2 paper evaluation suite.")
    parser.add_argument("--tokenizer_name", default="TinyLlama/TinyLlama_v1.1")
    parser.add_argument("--ruler_lengths", default="1024,2048,4096,8192")
    parser.add_argument("--ruler_limit", type=int, default=5, help="Generate a small sample now to download essays/tokenizer; full RULER is generated during eval.")
    parser.add_argument("--output", default="runs/eval/gdn2_paper/data_prefetch_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lengths = [int(item) for item in args.ruler_lengths.split(",") if item.strip()]
    manifest: list[dict] = []

    for dataset_path, dataset_name in DATASETS:
        entry = {"dataset_path": dataset_path, "dataset_name": dataset_name, "status": "pending"}
        try:
            kwargs = {"path": dataset_path}
            if dataset_name is not None:
                kwargs["name"] = dataset_name
            ds = load_dataset(**kwargs)
            entry["status"] = "ok"
            entry["splits"] = list(ds.keys())
            entry["rows"] = {split: len(ds[split]) for split in ds.keys() if hasattr(ds[split], "__len__")}
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
        manifest.append(entry)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, trust_remote_code=True)
    ruler_manifest: list[dict] = []
    for task_name, fn in RULER_TASKS.items():
        entry = {"task": task_name, "lengths": lengths, "status": "pending"}
        try:
            ds = fn(tokenizer=args.tokenizer_name, pretrained=args.tokenizer_name, max_seq_lengths=lengths)["test"]
            count = len(ds)
            samples = list(ds.select(range(min(args.ruler_limit, count)))) if args.ruler_limit > 0 else []
            token_lengths = [len(tokenizer(sample["input"]).input_ids) for sample in samples]
            entry["status"] = "ok"
            entry["rows"] = count
            entry["sample_token_lengths"] = token_lengths
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
        ruler_manifest.append(entry)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump({"datasets": manifest, "ruler": ruler_manifest}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
