#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "outputs" / "tsz128x4k_10B_gdn2_kla_1.3B_fineweb_edu_10bt"


@dataclass
class Job:
    name: str
    checkpoint_name: str
    checkpoint_path: Path
    output_path: Path
    command: list[str]


def checkpoint_sort_key(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)B", str(path))
    if match:
        return int(match.group(1))
    if path.name == "final-model-ckpt.pth":
        return 10_000
    if path.name == "latest-model-ckpt.pth":
        return 9_999
    return 0


def discover_checkpoints(out_dir: Path, include_latest: bool = False, include_final: bool = True) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for path in sorted((out_dir / "hf_checkpoints").glob("checkpoint-*B/model-ckpt.pth"), key=checkpoint_sort_key):
        name = path.parent.name.replace("checkpoint-", "")
        found.append((name, path))
    if include_final and (out_dir / "final-model-ckpt.pth").exists():
        found.append(("final", out_dir / "final-model-ckpt.pth"))
    if include_latest and (out_dir / "latest-model-ckpt.pth").exists():
        found.append(("latest", out_dir / "latest-model-ckpt.pth"))
    return found


def build_jobs(args: argparse.Namespace, checkpoints: list[tuple[str, Path]]) -> list[Job]:
    results_dir = Path(args.results_dir)
    jobs: list[Job] = []

    standard_tasks = "wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,social_iqa,boolq"
    real_tasks = "swde,squad_completion,fda,triviaqa,nq_open,drop"
    ruler_tasks = "niah_single_1,niah_single_2,niah_single_3,niah_multikey_1"

    for checkpoint_name, checkpoint_path in checkpoints:
        base = results_dir / checkpoint_name
        common = [
            "--checkpoint",
            str(checkpoint_path),
            "--model_name",
            args.model_name,
            "--tokenizer_name",
            args.tokenizer_name,
            "--dtype",
            args.dtype,
        ]
        jobs.append(
            Job(
                name=f"{checkpoint_name}:standard",
                checkpoint_name=checkpoint_name,
                checkpoint_path=checkpoint_path,
                output_path=base / "standard_lm_eval.json",
                command=[
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "lm_eval_gla2.py"),
                    *common,
                    "--tasks",
                    standard_tasks,
                    "--max_length",
                    "4096",
                    "--output",
                    str(base / "standard_lm_eval.json"),
                    "--bootstrap_iters",
                    str(args.bootstrap_iters),
                    *([] if args.limit is None else ["--limit", str(args.limit)]),
                ],
            )
        )
        jobs.append(
            Job(
                name=f"{checkpoint_name}:ruler_table3",
                checkpoint_name=checkpoint_name,
                checkpoint_path=checkpoint_path,
                output_path=base / "ruler_table3.json",
                command=[
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "ruler_eval_gla2.py"),
                    *common,
                    "--tasks",
                    ruler_tasks,
                    "--lengths",
                    "1024,2048,4096,8192",
                    "--max_length",
                    "8192",
                    "--output",
                    str(base / "ruler_table3.json"),
                    *([] if args.ruler_limit <= 0 else ["--limit", str(args.ruler_limit)]),
                ],
            )
        )
        jobs.append(
            Job(
                name=f"{checkpoint_name}:real_world_retrieval",
                checkpoint_name=checkpoint_name,
                checkpoint_path=checkpoint_path,
                output_path=base / "real_world_lm_eval.json",
                command=[
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "lm_eval_gla2.py"),
                    *common,
                    "--tasks",
                    real_tasks,
                    "--max_length",
                    "2048",
                    "--output",
                    str(base / "real_world_lm_eval.json"),
                    "--bootstrap_iters",
                    str(args.bootstrap_iters),
                    *([] if args.limit is None else ["--limit", str(args.limit)]),
                ],
            )
        )
    return jobs


def run_jobs(args: argparse.Namespace, jobs: list[Job]) -> int:
    logs_dir = Path(args.results_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pending = [job for job in jobs if args.overwrite or not job.output_path.exists()]
    running: dict[int, tuple[subprocess.Popen, Job, object]] = {}
    completed: list[dict] = []
    failed = 0

    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    while pending or running:
        for gpu in gpu_ids:
            gpu_int = int(gpu)
            if gpu_int in running or not pending:
                continue
            job = pending.pop(0)
            job.output_path.parent.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / f"{job.checkpoint_name}_{job.name.split(':')[-1]}_gpu{gpu}.log"
            log_f = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env.setdefault("NUMEXPR_MAX_THREADS", "256")
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            print(f"[launch] gpu={gpu} job={job.name}", flush=True)
            proc = subprocess.Popen(
                job.command,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[gpu_int] = (proc, job, log_f)

        time.sleep(args.poll_seconds)
        for gpu_int, (proc, job, log_f) in list(running.items()):
            ret = proc.poll()
            if ret is None:
                continue
            log_f.close()
            status = {"job": job.name, "checkpoint": job.checkpoint_name, "returncode": ret, "output": str(job.output_path)}
            completed.append(status)
            print(f"[done] gpu={gpu_int} rc={ret} job={job.name}", flush=True)
            if ret != 0:
                failed += 1
            del running[gpu_int]

        manifest = Path(args.results_dir) / "job_manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", encoding="utf-8") as f:
            json.dump({"pending": [job.name for job in pending], "running": [job.name for _, job, _ in running.values()], "completed": completed}, f, indent=2)

    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GDN-2 paper evaluation suite across checkpoints and GPUs.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--results_dir", type=Path, default=REPO_ROOT / "runs" / "eval" / "gdn2_paper")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--model_name", default="gdn2_kla_1.3B")
    parser.add_argument("--tokenizer_name", default="TinyLlama/TinyLlama_v1.1")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--limit", type=float, default=None, help="Optional lm-eval sample limit for debugging")
    parser.add_argument("--ruler_limit", type=int, default=0, help="Optional direct RULER sample limit for debugging")
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--poll_seconds", type=float, default=10.0)
    parser.add_argument("--include_latest", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require_10b", action="store_true", help="Fail unless checkpoint-10B exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = discover_checkpoints(args.out_dir, include_latest=args.include_latest)
    if args.require_10b and not any(name == "10B" for name, _ in checkpoints):
        raise SystemExit(f"checkpoint-10B not found under {args.out_dir / 'hf_checkpoints'}")
    if not checkpoints:
        raise SystemExit(f"No checkpoints found under {args.out_dir}")
    jobs = build_jobs(args, checkpoints)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    with (Path(args.results_dir) / "planned_jobs.json").open("w", encoding="utf-8") as f:
        json.dump(
            [{"name": job.name, "checkpoint": str(job.checkpoint_path), "output": str(job.output_path), "command": job.command} for job in jobs],
            f,
            indent=2,
            ensure_ascii=False,
        )
    raise SystemExit(run_jobs(args, jobs))


if __name__ == "__main__":
    main()
