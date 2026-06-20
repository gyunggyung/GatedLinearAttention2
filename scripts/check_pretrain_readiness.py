#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def parse_train_config(name: str) -> tuple[int, int, int]:
    match = re.search(r"tsz(\d+)x(\d+)k_(\d+)B", name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Unsupported train config: {name}")
    seq_batch = int(match.group(1))
    seq_len = int(match.group(2)) * 1024
    token_budget = int(match.group(3)) * 1_000_000_000
    return seq_batch, seq_len, token_budget


def default_micro_batch_size(sequence_length: int, model_name: str) -> int:
    if sequence_length >= 65536:
        base = 1
    elif sequence_length >= 32768:
        base = 2
    elif sequence_length >= 16384:
        base = 4
    else:
        base = 8
    if "1.3B" in model_name:
        base //= 2
    return max(1, base)


def format_duration(seconds: float) -> str:
    hours = seconds / 3600
    days = seconds / 86400
    return f"{days:.2f} days / {hours:.1f} hours"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check GDN-2 FineWeb-Edu 100BT pretraining readiness.")
    parser.add_argument("--train-config", default="tsz128x4k_100B")
    parser.add_argument("--model-name", default="gdn2_1.3B")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data/fineweb-edu/data"))
    parser.add_argument("--micro-batch-size", type=int, default=0)
    parser.add_argument("--global-batch-tokens", type=int, default=524288)
    parser.add_argument("--devices", type=int, default=0, help="Device count for schedule math; 0 means detected GPUs.")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--throughput", type=float, default=250000.0, help="Assumed total tokens/sec for ETA.")
    args = parser.parse_args()

    deps = {
        "torch": has_module("torch"),
        "lightning_or_lightning_fabric": has_module("lightning") or has_module("lightning_fabric"),
        "pytorch_lightning": has_module("pytorch_lightning"),
        "datasets": has_module("datasets"),
        "transformers": has_module("transformers"),
        "fla": has_module("fla"),
        "wandb": has_module("wandb"),
        "torchdata": has_module("torchdata"),
        "flash_attn_optional_for_pure_gdn": has_module("flash_attn"),
    }

    print("Dependency status")
    for name, ok in deps.items():
        print(f"  {name}: {'OK' if ok else 'MISSING'}")

    gpu_count = 0
    if deps["torch"]:
        import torch

        gpu_count = torch.cuda.device_count()
        print("\nGPU status")
        print(f"  cuda_available: {torch.cuda.is_available()}")
        print(f"  gpu_count: {gpu_count}")
        for index in range(gpu_count):
            props = torch.cuda.get_device_properties(index)
            print(f"  gpu_{index}: {props.name}, {props.total_memory / 1024**3:.1f} GiB")

    data_root = Path(args.data_dir)
    files = sorted(data_root.glob("*.parquet")) + sorted(data_root.glob("**/*.parquet"))
    files = sorted(set(files))
    size = sum(path.stat().st_size for path in files)
    print("\nData status")
    print(f"  data_dir: {data_root}")
    print(f"  parquet_files: {len(files)}")
    print(f"  size_decimal_gb: {size / 1e9:.2f}")

    seq_batch, seq_len, token_budget = parse_train_config(args.train_config)
    devices = args.devices or max(1, gpu_count)
    micro_batch = args.micro_batch_size or default_micro_batch_size(seq_len, args.model_name)
    tokens_per_micro = micro_batch * seq_len * devices * args.nodes
    grad_accum = max(1, math.ceil(args.global_batch_tokens / tokens_per_micro))
    effective_global_batch = tokens_per_micro * grad_accum
    optimizer_steps = math.ceil(token_budget / effective_global_batch)
    micro_iters = optimizer_steps * grad_accum
    trained_tokens = micro_iters * tokens_per_micro

    print("\nTraining schedule")
    print(f"  train_config: {args.train_config}")
    print(f"  sequence_length: {seq_len}")
    print(f"  sequence_batch_from_config: {seq_batch}")
    print(f"  micro_batch_size_per_gpu: {micro_batch}")
    print(f"  gradient_accumulation_steps: {grad_accum}")
    print(f"  effective_global_batch_tokens: {effective_global_batch:,}")
    print(f"  optimizer_steps: {optimizer_steps:,}")
    print(f"  micro_iterations: {micro_iters:,}")
    print(f"  stop_tokens: {trained_tokens:,}")

    print("\nETA")
    for throughput in [100000, 200000, args.throughput, 300000, 500000]:
        print(f"  {throughput:,.0f} tokens/s: {format_duration(token_budget / throughput)}")

    blockers = [
        name for name in ["torch", "lightning_or_lightning_fabric", "pytorch_lightning", "datasets", "transformers", "fla", "wandb"]
        if not deps[name]
    ]
    print("\nReadiness")
    if blockers:
        print(f"  BLOCKED: missing {', '.join(blockers)}")
    elif len(files) != 140:
        print("  BLOCKED: FineWeb-Edu sample/100BT parquet file count is not 140")
    elif gpu_count < 8:
        print("  WARNING: fewer than 8 GPUs detected")
    elif not deps["torchdata"]:
        print("  READY WITH WARNING: torchdata is missing, so dataloader resume state is not available")
    else:
        print("  READY")


if __name__ == "__main__":
    main()
