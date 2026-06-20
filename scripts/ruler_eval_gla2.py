#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMEXPR_MAX_THREADS", "256")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lm_eval.tasks.ruler import common_utils, niah_utils  # noqa: E402
from lm_eval.tasks.ruler.common_utils import string_match_all  # noqa: E402
from lm_eval_gla2 import GatedLinearAttention2LM  # noqa: E402


TASKS = {
    "niah_single_1": niah_utils.niah_single_1,
    "niah_single_2": niah_utils.niah_single_2,
    "niah_single_3": niah_utils.niah_single_3,
    "niah_multikey_1": niah_utils.niah_multikey_1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct RULER NIAH evaluation for GatedLinearAttention2.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1")
    parser.add_argument("--lengths", default="1024,2048,4096,8192")
    parser.add_argument("--model_name", default="gdn2_kla_1.3B")
    parser.add_argument("--tokenizer_name", default="TinyLlama/TinyLlama_v1.1")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--limit", type=int, default=0, help="0 means full generated dataset")
    parser.add_argument("--max_gen_toks", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    lengths = [int(item) for item in args.lengths.split(",") if item.strip()]
    common_utils.DEFAULT_SEQ_LENGTHS[:] = lengths

    lm = GatedLinearAttention2LM(
        checkpoint=args.checkpoint,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        max_length=max(args.max_length, max(lengths)),
        device=args.device,
        dtype=args.dtype,
    )

    summary: dict[str, dict[str, float | int]] = {}
    examples: dict[str, list[dict[str, str]]] = {}
    for task_name in tasks:
        if task_name not in TASKS:
            raise ValueError(f"Unsupported RULER task: {task_name}")
        dataset = TASKS[task_name](
            tokenizer=args.tokenizer_name,
            pretrained=args.tokenizer_name,
            max_seq_lengths=lengths,
        )["test"]
        rows = list(dataset)
        if args.limit and args.limit > 0:
            rows = rows[: args.limit]

        by_length: dict[int, list[float]] = {length: [] for length in lengths}
        task_examples: list[dict[str, str]] = []
        for row in rows:
            prompt = row["input"].strip()
            gen_prefix = row.get("gen_prefix", "").strip()
            if gen_prefix:
                prompt = f"{prompt} {gen_prefix}"
            pred = lm.generate_until(
                [
                    type(
                        "Req",
                        (),
                        {"args": (prompt, {"until": [], "max_gen_toks": args.max_gen_toks, "do_sample": False})},
                    )()
                ]
            )[0]
            score = string_match_all([pred], [row["outputs"]])
            by_length[int(row["max_length"])].append(float(score))
            if len(task_examples) < 5:
                task_examples.append(
                    {
                        "length": str(row["max_length"]),
                        "prediction": pred[:300],
                        "answers": ", ".join(row["outputs"]),
                    }
                )

        summary[task_name] = {}
        for length, scores in by_length.items():
            summary[task_name][str(length)] = sum(scores) / len(scores) if scores else -1.0
            summary[task_name][f"{length}_n"] = len(scores)
        examples[task_name] = task_examples

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "tokenizer": args.tokenizer_name,
                "lengths": lengths,
                "limit": args.limit,
                "results": summary,
                "examples": examples,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
