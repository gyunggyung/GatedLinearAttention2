#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric(results: dict, task: str, preferred: list[str]) -> float | None:
    task_results = results.get("results", {}).get(task, {})
    for key in preferred:
        if key in task_results:
            value = task_results[key]
            if isinstance(value, (int, float)):
                return float(value)
    return None


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}" if value <= 1.5 else f"{value:.2f}"


def ppl(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def parse_standard(path: Path) -> dict[str, float | None]:
    data = load_json(path) or {}
    return {
        "wiki_word_ppl": metric(data, "wikitext", ["word_perplexity,none", "word_perplexity"]),
        "lambada_ppl": metric(data, "lambada_openai", ["perplexity,none", "perplexity"]),
        "lambada_acc": metric(data, "lambada_openai", ["acc,none", "acc"]),
        "piqa_acc": metric(data, "piqa", ["acc,none", "acc"]),
        "hellaswag_acc_norm": metric(data, "hellaswag", ["acc_norm,none", "acc_norm"]),
        "winogrande_acc": metric(data, "winogrande", ["acc,none", "acc"]),
        "arc_easy_acc": metric(data, "arc_easy", ["acc,none", "acc"]),
        "arc_challenge_acc": metric(data, "arc_challenge", ["acc,none", "acc"]),
        "openbookqa_acc": metric(data, "openbookqa", ["acc,none", "acc"]),
        "social_iqa_acc": metric(data, "social_iqa", ["acc,none", "acc"]),
        "boolq_acc": metric(data, "boolq", ["acc,none", "acc"]),
    }


def parse_real(path: Path) -> dict[str, float | None]:
    data = load_json(path) or {}
    return {
        "swde": metric(data, "swde", ["contains,none", "contains"]),
        "squad": metric(data, "squad_completion", ["contains,none", "contains"]),
        "fda": metric(data, "fda", ["contains,none", "contains"]),
        "triviaqa": metric(data, "triviaqa", ["exact_match,remove_whitespace", "exact_match,none", "exact_match"]),
        "nq": metric(data, "nq_open", ["exact_match,remove_whitespace", "exact_match,none", "exact_match"]),
        "drop": metric(data, "drop", ["em,none", "em", "f1,none", "f1"]),
    }


def parse_ruler(path: Path) -> dict[str, float | None]:
    data = load_json(path) or {}
    out: dict[str, float | None] = {}
    for task, values in data.get("results", {}).items():
        for length in ["1024", "2048", "4096", "8192"]:
            if length in values:
                out[f"{task}_{int(length)//1024}k"] = values[length] * 100
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GDN-2 paper evaluation outputs.")
    parser.add_argument("--results_dir", type=Path, default=REPO_ROOT / "runs" / "eval" / "gdn2_paper")
    parser.add_argument("--baseline", type=Path, default=REPO_ROOT / "eval" / "gdn2_paper_baselines.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "runs" / "eval" / "gdn2_paper" / "GDN2_PAPER_EVAL_RESULTS.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_json(args.baseline) or {}
    checkpoint_dirs = sorted([p for p in args.results_dir.iterdir() if p.is_dir() and p.name != "logs"])

    lines: list[str] = []
    lines.append("# GatedLinearAttention2 GDN-2 Paper Evaluation Results")
    lines.append("")
    lines.append("This report is generated from local evaluation JSON files and compares against GDN-2 paper Tables 2, 3, and 4.")
    lines.append(f"Generated at: {datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')}.")
    lines.append("")
    lines.append("`n/a` means the matching JSON or metric is not available yet. Social IQA is marked `n/a` in the accelerated local run because the installed `datasets==4.8.5` blocks the legacy `allenai/social_i_qa` dataset script used by this lm-eval task.")
    lines.append("")
    lines.append("## Baseline Targets")
    lines.append("")
    rec_t2 = baseline.get("table2_language_modeling_and_commonsense", {}).get("recurrent", {}).get("Gated DeltaNet-2", [])
    rec_t4 = baseline.get("table4_real_world_retrieval", {}).get("recurrent", {}).get("Gated DeltaNet-2", [])
    lines.append(f"- Paper recurrent GDN-2 Table 2 avg acc: {rec_t2[-1] if rec_t2 else 'n/a'}")
    lines.append(f"- Paper recurrent GDN-2 Table 4 recall avg: {rec_t4[-1] if rec_t4 else 'n/a'}")
    lines.append("- RULER Table 3 target is the recurrent GDN-2 row in `eval/gdn2_paper_baselines.json`.")
    lines.append("")

    for ckpt in checkpoint_dirs:
        standard = parse_standard(ckpt / "standard_lm_eval.json")
        real = parse_real(ckpt / "real_world_lm_eval.json")
        ruler = parse_ruler(ckpt / "ruler_table3.json")
        lines.append(f"## Checkpoint {ckpt.name}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for key in ["wiki_word_ppl", "lambada_ppl"]:
            lines.append(f"| {key} | {ppl(standard.get(key))} |")
        for key in ["lambada_acc", "piqa_acc", "hellaswag_acc_norm", "winogrande_acc", "arc_easy_acc", "arc_challenge_acc", "openbookqa_acc", "social_iqa_acc", "boolq_acc"]:
            lines.append(f"| {key} | {pct(standard.get(key))} |")
        lines.append("")
        lines.append("| RULER Metric | Value |")
        lines.append("|---|---:|")
        for key in sorted(ruler):
            lines.append(f"| {key} | {pct(ruler.get(key))} |")
        lines.append("")
        lines.append("| Real-world Retrieval | Value |")
        lines.append("|---|---:|")
        vals = []
        for key in ["swde", "squad", "fda", "triviaqa", "nq", "drop"]:
            vals.append(real.get(key))
            lines.append(f"| {key} | {pct(real.get(key))} |")
        valid = [v for v in vals if v is not None]
        lines.append(f"| avg | {pct(sum(valid) / len(valid) if valid else None)} |")
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
