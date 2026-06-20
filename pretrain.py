# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE
import argparse
import datetime
import glob
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from distutils.dir_util import copy_tree
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

os.environ.setdefault("NUMEXPR_MAX_THREADS", "256")

try:
    import lightning as L
    from lightning.fabric.strategies import FSDPStrategy
except ImportError:
    import lightning_fabric as L
    from lightning_fabric.strategies import FSDPStrategy
import torch
import torch.multiprocessing as mp
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))
from lit_gpt.model import GPT, Block, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.utils import chunked_cross_entropy, num_parameters
from lit_gpt import FusedCrossEntropyLoss
from data import get_stateful_stream_tok_dataset

_TRAIN_START_TIME = time.time()
GITHUB_REPO_URL = "https://github.com/gyunggyung/Gated_Linear_Attention2"
MODEL_CARD_USAGE_MD = """## How To Use

This is a causal language model: given a text prefix, it predicts the next token
and can continue the text autoregressively. It was pretrained on FineWeb-Edu and
is not instruction-tuned, RLHF-tuned, or chat-aligned.

The checkpoint is a PyTorch `.pth` checkpoint, not a
`transformers.AutoModelForCausalLM` checkpoint. Use the standalone runtime below
to load it.

Install and clone:

```bash
git clone https://github.com/gyunggyung/Gated_Linear_Attention2
cd Gated_Linear_Attention2
pip install -e .
```

Minimal text-generation example:

```python
import torch

from gated_linear_attention2 import GatedLinearAttention2ForCausalLM, load_tokenizer
from gated_linear_attention2.generation import generate

repo_id = "gyung/Gated_Linear_Attention2"
checkpoint_file = "checkpoints/checkpoint-01B/model-ckpt.pth"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is recommended for this 1.3B checkpoint; CPU will be very slow.")

device = "cuda"
dtype = torch.bfloat16

model = GatedLinearAttention2ForCausalLM.from_hf(
    repo_id=repo_id,
    checkpoint=checkpoint_file,
    device=device,
    dtype=dtype,
)
tokenizer = load_tokenizer(repo_id, subfolder="tokenizer")

prompt = "Artificial intelligence can help education by"
print(generate(model, tokenizer, prompt, max_new_tokens=80, temperature=0.8, top_k=50))
```

For next-token scoring instead of generation, run one forward pass and inspect
the final-position logits:

```python
prompt = "The capital of France is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    logits = model(input_ids)[:, -1, :]
next_token_id = int(torch.argmax(logits, dim=-1)[0])
print(tokenizer.decode([next_token_id]))
```

The standalone runtime uses a recurrent state cache during generation, so decode
memory does not grow with generated token length like a Transformer KV cache.
"""

os.environ["TRITON_CACHE_MANAGER"] = "cache:ParallelFileCacheManager"


def token_budget_from_config(name: str) -> int:
    config_match = re.search(r"tsz\d+x\d+k_(\d+(?:\.\d+)?)B", name, re.IGNORECASE)
    if config_match is not None:
        return int(float(config_match.group(1)) * 1_000_000_000)
    upper_name = name.upper()
    if "100B" in upper_name:
        return int(1e11)
    if "50B" in upper_name:
        return int(5e10)
    if "30B" in upper_name:
        return int(3e10)
    if "20B" in upper_name:
        return int(2e10)
    if "15B" in upper_name:
        return int(1.5e10)
    if "10B" in upper_name:
        return int(1e10)
    raise ValueError(f"Unknown training token budget in train_config/exp_name: {name}")


def parse_sequence_batch(train_config: str) -> tuple[int, int]:
    match = re.search(r"tsz(\d+)x(\d+)k", train_config.lower())
    if match is None:
        raise ValueError(
            f"Cannot parse train_config={train_config!r}. Expected a form like 'tsz128x4k_100B'."
        )
    return int(match.group(1)), int(match.group(2)) * 1024


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


def build_strategy(args):
    world_size = devices * args.nodes
    if world_size <= 1:
        return "auto"
    if args.interactive_job or args.nodes == 1:
        return FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full")
    return FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full", sharding_strategy="HYBRID_SHARD")


def make_wandb_logger(args):
    mode = "disabled" if args.debug else args.wandb_mode
    group = "debug" if args.debug else args.exp_group
    return WandbLogger(
        project=args.wandb_project,
        mode=mode,
        name=args.exp_name,
        id=args.exp_name,
        save_dir=args.wandb_dir,
        dir=args.wandb_dir,
        version=args.exp_name,
        group=group,
    )


def load_env_file(path: str) -> None:
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def resolve_hf_repo_id(args) -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF upload is enabled, but HF_TOKEN is not set in the environment or .env file.")
    from huggingface_hub import HfApi

    user = HfApi().whoami(token=token)["name"]
    if args.hf_repo_id:
        repo_id = args.hf_repo_id.strip()
        return repo_id if "/" in repo_id else f"{user}/{repo_id}"
    repo_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.exp_name).strip("-").lower()
    return f"{user}/{repo_name}"


def unwrap_model(model):
    return getattr(model, "module", model)


def write_hf_milestone_metadata(args, model, folder: Path, milestone_index: int, trained_tokens: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    model_config_obj = getattr(unwrap_model(model), "config", None) or getattr(model, "config", None)
    model_config = model_config_obj.__dict__.copy() if model_config_obj is not None else {}
    metadata = {
        "model_name": args.model_name,
        "experiment_name": args.exp_name,
        "train_config": args.train_config,
        "trained_tokens": trained_tokens,
        "milestone_index": milestone_index,
        "tokenizer_name": args.tokenizer_name,
        "tokenizer_path": args.tokenizer_path,
        "data_source": "HuggingFaceFW/fineweb-edu sample/100BT local parquet",
        "sequence_length": model_config.get("block_size"),
        "global_batch_tokens": args.global_batch_tokens,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "data_shuffle_seed": args.data_shuffle_seed,
        "data_shuffle_buffer": args.data_shuffle_buffer,
        "checkpoint_format": "LitGPT/Fabric .pth, not a Transformers AutoModel checkpoint",
        "model_config": model_config,
    }
    (folder / "training_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    readme = f"""---
license: apache-2.0
library_name: pytorch
tags:
- linear-attention
- recurrent
- gated-deltanet
- gdn2
- kaczmarz
datasets:
- HuggingFaceFW/fineweb-edu
---

# Gated_Linear_Attention2

This repository stores milestone checkpoints for `{args.model_name}` from the
`{args.exp_name}` run.

## What This Model Is

`gdn2_kla_1.3B` is a recurrent-only linear attention experiment. It starts from
Gated DeltaNet-2 and folds a Kaczmarz-style key-norm-normalized update step into
the separate erase and write gates:

```math
\\lambda_t = \\frac{{\\eta_t}}{{\\|k_t\\|_2^2 + \\epsilon}}
```

```math
S_t =
\\left(I - k_t(\\lambda_t b_t \\odot k_t)^\\top\\right)D_tS_{{t-1}}
+
k_t(\\lambda_t w_t \\odot v_t)^\\top
```

It is not a standard Transformers checkpoint and does not use softmax attention
or SWA layers.

## Code

- GitHub: {GITHUB_REPO_URL}

## License

The model weights in this Hugging Face repository are released under Apache-2.0.

The standalone inference runtime linked above is also Apache-2.0. It does not
import `lit_gpt`, `fla`, or the NVIDIA GatedDeltaNet-2 Triton kernels. The
training code used during experimentation may contain NVIDIA GatedDeltaNet-2
derived components under `Nvidia Source Code License-NC`, but this Hugging Face
model repository is intended to be used with the standalone Apache-2.0 runtime.

## Training Setup

- Base architecture: recurrent-only GDN-2, 1.3B scale
- Candidate: Kaczmarz-normalized GDN-2 gates
- Training data source: FineWeb-Edu `sample/100BT` local parquet
- Token budget for this run: 10B
- Current milestone: {trained_tokens:,} tokens
- Sequence length: {model_config.get("block_size")} tokens
- Global batch tokens: {args.global_batch_tokens:,}
- Tokenizer: `{args.tokenizer_name}`
- Data shuffle seed: `{args.data_shuffle_seed}`
- Data shuffle buffer: `{args.data_shuffle_buffer}`

## Checkpoint Format

Each `checkpoints/checkpoint-XXB/` folder contains:

- `model-ckpt.pth`: PyTorch model-only checkpoint
- `training_metadata.json`: run metadata and model config
- `README.md`: this model card snapshot

This is not loadable with `transformers.AutoModelForCausalLM.from_pretrained`.

{MODEL_CARD_USAGE_MD}

## Evaluation Plan

Compare against the plain `gdn2_1.3B` baseline on the GDN-2 paper tasks:

- WikiText and LAMBADA perplexity
- LAMBADA and commonsense zero-shot accuracy
- RULER S-NIAH and MK-NIAH
- Real-world retrieval tasks: SWDE, SQuAD, FDA, TriviaQA, NQ, DROP

The 10B run is an ablation, not a claim that it replaces the published 100B
GDN-2 model.
"""
    (folder / "README.md").write_text(readme, encoding="utf-8")


def launch_hf_upload(args, folder: Path, path_in_repo: str, log_path: Path) -> None:
    if not args.hf_upload:
        return
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "upload_hf_checkpoint.py"),
        "--folder",
        str(folder),
        "--repo-id",
        args.hf_repo_id_resolved,
        "--path-in-repo",
        path_in_repo,
        "--env-file",
        args.env_file,
    ]
    if args.hf_private:
        cmd.append("--private")
    stdout = open(log_path, "a", encoding="utf-8")
    stderr = subprocess.STDOUT
    if args.hf_upload_blocking:
        subprocess.run(cmd, check=True, stdout=stdout, stderr=stderr)
        stdout.close()
    else:
        subprocess.Popen(cmd, stdout=stdout, stderr=stderr, start_new_session=True)


def load_tokenizer(args, fabric, config: Config):
    tokenizer_source = args.tokenizer_path if args.tokenizer_path and Path(args.tokenizer_path).exists() else args.tokenizer_name
    if fabric.global_rank == 0:
        fabric.print(f"Loading tokenizer from {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        fallback_token = tokenizer.eos_token or tokenizer.unk_token
        if fallback_token is None:
            raise ValueError("Tokenizer has no pad/eos/unk token; set a tokenizer with a usable padding token.")
        tokenizer.pad_token = fallback_token
    tokenizer.model_max_length = 999999999
    if len(tokenizer) > config.padded_vocab_size:
        raise ValueError(
            f"Tokenizer size {len(tokenizer)} exceeds model padded vocab size {config.padded_vocab_size}."
        )
    return tokenizer


def validation_enabled(args) -> bool:
    if args.val_type.lower() in {"", "none", "off", "disabled"}:
        return False
    return bool(args.val_data_dir_raw or args.val_data_dir)


def format_duration(seconds: float) -> str:
    days = seconds / 86400
    hours = seconds / 3600
    if days >= 1:
        return f"{days:.2f} days ({hours:.1f} hours)"
    return f"{hours:.2f} hours"


def print_run_schedule(args, fabric, config: Config):
    tokens_per_micro_iter_global = args.micro_batch_size * config.block_size * fabric.world_size
    optimizer_steps = math.ceil(args.max_tokens / args.global_batch_tokens)
    micro_iters = optimizer_steps * args.gradient_accumulation_steps
    fabric.print("##### Token Schedule #####")
    fabric.print(f"Sequence length: {config.block_size}")
    fabric.print(f"Tokens per global micro-iteration: {tokens_per_micro_iter_global:,}")
    fabric.print(f"Target global batch tokens: {args.target_global_batch_tokens:,}")
    fabric.print(f"Effective global batch tokens: {args.global_batch_tokens:,}")
    fabric.print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    fabric.print(f"Estimated micro-iterations: {micro_iters:,}")
    fabric.print(f"Estimated optimizer steps: {optimizer_steps:,}")
    fabric.print(f"Estimated trained tokens at stop: {micro_iters * tokens_per_micro_iter_global:,}")
    fabric.print(f"Activation checkpointing: {config.activation_checkpointing}")
    if args.expected_tokens_per_sec > 0:
        fabric.print(
            f"ETA at {args.expected_tokens_per_sec:,.0f} tokens/s: "
            f"{format_duration(args.max_tokens / args.expected_tokens_per_sec)}"
        )


def main(args):
    load_env_file(args.env_file)
    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.wandb_dir, exist_ok=True)
    wandb_logger = make_wandb_logger(args)

    strategy = build_strategy(args)
    precision = "bf16-mixed" if torch.cuda.is_available() else "32-true"
    fabric = L.Fabric(
        devices=devices,
        num_nodes=args.nodes,
        strategy=strategy,
        precision=precision,
        loggers=[wandb_logger],
    )
    fabric.launch()
    # fix seed in the very beginning
    fabric.seed_everything(args.seed)  # same seed for every process to init model (FSDP)
    fabric.print("##### Infra Details #####")
    fabric.print(f"Number of Nodes: {args.nodes}")
    fabric.print(f"Number of GPUs: {fabric.world_size}")
    fabric.print("##### Training Details #####")
    fabric.print(f"Maximum number of training tokens: {args.max_tokens}")
    fabric.print(f"Maximum training time: {args.actual_train_time/60.0} min")
    fabric.print(f"Micro batch size: {args.micro_batch_size}")
    fabric.print(f"Batch size: {args.batch_size}")
         
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME], device=fabric.device, dtype=torch.int64)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    if fabric.global_rank == 0:
        if args.hf_upload:
            args.hf_repo_id_resolved = resolve_hf_repo_id(args)
            fabric.print(f"HF upload enabled: {args.hf_repo_id_resolved}")
        else:
            args.hf_repo_id_resolved = ""
        fabric.print(args)
    fabric.logger.log_hyperparams(args)
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=args.log_iter_interval)

    if os.path.exists(args.out_dir):
        args.resume = True
        print('Resuming from {}'.format(args.out_dir))
    else:
        if fabric.global_rank == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            target_litgpt_save_dir = os.path.join(args.out_dir, 'lit_gpt')
            target_bash_scripts_save_dir = os.path.join(args.out_dir, 'bash_scripts')
            target_pretrain_file = os.path.join(args.out_dir, 'pretrain.py')
            os.makedirs(target_litgpt_save_dir, exist_ok=True)
            os.makedirs(target_bash_scripts_save_dir, exist_ok=True)
            if not args.debug:
                copy_tree(str(REPO_ROOT / 'lit_gpt'), target_litgpt_save_dir)
                copy_tree(str(REPO_ROOT / "scripts"), target_bash_scripts_save_dir)
                shutil.copyfile(str(REPO_ROOT / "pretrain.py"), target_pretrain_file)
    if fabric.world_size > 1:
        fabric.barrier()
    
    config = Config.from_name(
        args.model_name,
        block_size=args.sequence_length,
        activation_checkpointing=args.activation_checkpointing,
    )
    if fabric.global_rank == 0:
        print_run_schedule(args, fabric, config)
    if args.use_stream_tok:
        tokenizer = load_tokenizer(args, fabric, config)
        train_data_path = args.train_data_dir_raw or args.train_data_dir
        if not train_data_path:
            raise ValueError("--train_data_dir_raw or --train_data_dir is required when --use_stream_tok is enabled.")
        train_dataloader = get_stateful_stream_tok_dataset(
            corpus_name=args.corpus_name,
            path=train_data_path,
            split='train',
            tokenizer=tokenizer,
            block_size=config.block_size + 1,
            rank=fabric.global_rank,
            world_size=fabric.world_size,
            batch_size=args.micro_batch_size,
            num_workers=args.train_num_workers,
            shuffle_seed=args.data_shuffle_seed,
            shuffle_buffer_size=args.data_shuffle_buffer,
        )
        val_dataloader = None
        if validation_enabled(args):
            val_dataloader = get_stateful_stream_tok_dataset(
                corpus_name=args.corpus_name,
                path=args.val_data_dir_raw or args.val_data_dir,
                split=args.val_type,
                tokenizer=tokenizer,
                block_size=16384 + 1,
                rank=fabric.global_rank,
                world_size=fabric.world_size,
                batch_size=max(1, args.micro_batch_size // 2),
                num_workers=args.val_num_workers,
                shuffle_seed=args.data_shuffle_seed,
                shuffle_buffer_size=0,
            )

    else:
        train_dataloader, val_dataloader = create_dataloaders(
        batch_size=args.micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        seed=args.seed,
        )
        if val_dataloader is None:
            train_dataloader = fabric.setup_dataloaders(train_dataloader)
        else:
            train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    if not validation_enabled(args):
        val_dataloader = None
        
    if fabric.global_rank == 0:
        fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(model._init_weights ,n_layer=config.n_layer))
    
    if fabric.global_rank == 0:
        fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
        # we ignore the embedding & lm head parameter, which is standard 
        fabric.print(f"Total parameters {num_parameters(model.transformer.h):,}")
        fabric.print(model)
    
    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        fused=torch.cuda.is_available(),
    )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {
        "model": model,
        "optimizer": optimizer,
        "hparams": args.hparams,
        "iter_num": 0,
        "step_count": 0,
        "trained_tokens_total": 0,
        "next_hf_upload_tokens": args.hf_upload_interval_tokens if args.hf_upload else 0,
    }

    if args.resume:
        resume = os.path.join(args.out_dir, "latest-model-ckpt.pth")
        try:
            if fabric.global_rank == 0:
                fabric.print(f"Resuming training from {resume}")
            load_state = state.copy()
            load_state.pop("trained_tokens_total", None)
            load_state.pop("next_hf_upload_tokens", None)
            checkpoint_remainder = fabric.load(resume, load_state)
            state.update(load_state)
            if isinstance(checkpoint_remainder, dict):
                for metadata_key in ("trained_tokens_total", "next_hf_upload_tokens"):
                    if metadata_key in checkpoint_remainder:
                        state[metadata_key] = checkpoint_remainder[metadata_key]
            if args.resume_trained_tokens > 0:
                state["trained_tokens_total"] = args.resume_trained_tokens
            fabric.print(f"Successfully resumed from {resume}")
        except Exception as exc:
            fabric.print(f"Failed to resume from {resume}: {type(exc).__name__}: {exc}")
            raise
    train_time = time.perf_counter()
    train(args, _TRAIN_START_TIME, fabric, state, train_dataloader, val_dataloader, monitor, args.resume)
    if fabric.global_rank == 0:
        fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        if fabric.global_rank == 0:
            fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(args, _TRAIN_START_TIME, fabric, state, train_dataloader, val_dataloader, monitor, resume):
    
    model = state["model"]
    optimizer = state["optimizer"]

    total_lengths = 0
    total_t0 = time.perf_counter()    
    tokens_per_iter = args.micro_batch_size * model.config.block_size
    tokens_per_global_micro_iter = tokens_per_iter * fabric.world_size
    tokens_per_optimizer_step = tokens_per_global_micro_iter * args.gradient_accumulation_steps
    max_optimizer_steps = math.ceil(args.max_tokens / tokens_per_optimizer_step)
    max_iters = max_optimizer_steps * args.gradient_accumulation_steps
    warmup_optimizer_steps = max(1, math.ceil(args.warmup_tokens / tokens_per_optimizer_step))
    warmup_iters = warmup_optimizer_steps * args.gradient_accumulation_steps
    initial_iter = state["iter_num"]
    if "trained_tokens_total" not in state or state["trained_tokens_total"] <= 0:
        state["trained_tokens_total"] = state["iter_num"] * tokens_per_global_micro_iter
    curr_iter = 0
    loss_func = FusedCrossEntropyLoss()    
    if args.hf_upload and state.get("next_hf_upload_tokens", 0) <= 0:
        state["next_hf_upload_tokens"] = args.hf_upload_interval_tokens

    if resume:
        if args.use_stream_tok:
            if not hasattr(train_dataloader, "load_state_dict"):
                raise RuntimeError("Resuming streaming data state requires torchdata StatefulDataLoader.")
            try:
                if fabric.world_size <= 1:
                    data_state_path = os.path.join(args.out_dir,"latest-data-state-ckpt.pth")
                else:
                    data_state_path = os.path.join(args.out_dir,f"latest-data-states-rank-{fabric.global_rank}-ckpt.pth")
                train_dataloader.load_state_dict(torch.load(data_state_path, map_location="cpu"))
                if fabric.global_rank == 0:
                    fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
                resume = False
            except:
                fabric.print(f"Failed to resume dataloader from {args.out_dir}")
                raise KeyError("Failed to resume dataloader.. Please retrain from scratch.")

    tokens = 0
    train_t0 = time.perf_counter()
    
    if args.eval_before_training and val_dataloader is not None:
        fabric.print("Do validation before training:")
        val_loss = validate(args, fabric, model, val_dataloader, None)
        for i in range(args.num_extrapol):
            if fabric.global_rank == 0:
                fabric.print(f"step {state['iter_num']} {i+1} x: val loss {val_loss[i]:.4f}")
    
    def save_checkpoint(final=False):
        name = 'latest' if not final else 'final'
        checkpoint_path = os.path.join(args.out_dir,f"{name}-model-ckpt.pth")
        fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
        if not final:
            fabric.save(checkpoint_path, state)
        else:
            # we are not interested in the optimizer state for the final checkpoint 
            state['optimizer'] = None
            fabric.save(checkpoint_path, state)

        if args.use_stream_tok and not final and hasattr(train_dataloader, "state_dict"):
            if fabric.world_size <= 1:
                checkpoint_path = os.path.join(args.out_dir,f"latest-data-state-ckpt.pth")
            else:
                checkpoint_path = os.path.join(args.out_dir,f"latest-data-states-rank-{fabric.global_rank}-ckpt.pth")
            torch.save(train_dataloader.state_dict(), checkpoint_path)                
            fabric.print(f"Dataloader state checkpoint saved")
        elif args.use_stream_tok and not final:
            fabric.print("Dataloader state checkpoint skipped because torchdata StatefulDataLoader is unavailable")

    def save_hf_milestone(milestone_tokens: int):
        milestone_index = max(1, round(milestone_tokens / 1_000_000_000))
        folder = Path(args.out_dir) / "hf_checkpoints" / f"checkpoint-{milestone_index:02d}B"
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint_path = folder / "model-ckpt.pth"
        upload_state = {
            "model": model,
            "hparams": args.hparams,
            "iter_num": state["iter_num"],
            "step_count": state["step_count"],
            "trained_tokens": milestone_tokens,
        }
        fabric.print(f"Saving HF model-only milestone to {str(checkpoint_path)!r}")
        fabric.save(str(checkpoint_path), upload_state)
        if fabric.global_rank == 0:
            write_hf_milestone_metadata(args, model, folder, milestone_index, milestone_tokens)
        if fabric.world_size > 1:
            fabric.barrier()
        if fabric.global_rank == 0:
            log_path = Path(args.out_dir) / "hf_upload.log"
            launch_hf_upload(
                args,
                folder,
                path_in_repo=f"checkpoints/checkpoint-{milestone_index:02d}B",
                log_path=log_path,
            )

    for train_data in train_dataloader:
        # per gpu
        tokens += model.config.block_size * args.micro_batch_size
        if resume and not args.use_stream_tok:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                if fabric.global_rank == 0:
                    fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))

        if state["trained_tokens_total"] >= args.max_tokens:
            break
    
        iter_t0 = time.perf_counter()
        if args.use_stream_tok:
            input_ids = train_data['input_ids'][:, 0 : model.config.block_size].contiguous().to(fabric.device)
            targets = train_data['labels'][:, 1 : model.config.block_size + 1].contiguous().to(fabric.device)
        else:
            input_ids = train_data[:, 0 : model.config.block_size].contiguous()
            targets = train_data[:, 1 : model.config.block_size + 1].contiguous()

        lr_iter = min(max_iters, int(state["trained_tokens_total"] / tokens_per_optimizer_step) * args.gradient_accumulation_steps)
        lr = get_lr(args, lr_iter, warmup_iters, max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        is_accumulating = (state["iter_num"] + 1) % args.gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = loss_func(logits, targets)
            # Check if loss is NaN
            if torch.isnan(loss):
                # Create a debug directory if it doesn't exist
                debug_dir = "./logs/debug"
                os.makedirs(debug_dir, exist_ok=True)
                
                # Save the relevant tensors
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                torch.save({
                    'input_ids': input_ids,
                    'logits': logits,
                    'targets': targets,
                    'loss': loss
                }, os.path.join(debug_dir, f'nan_tensors_{timestamp}.pt'))
                
                print(f"NaN loss detected! Tensors saved to {debug_dir}/nan_tensors_{timestamp}.pt")
            fabric.backward(loss / args.gradient_accumulation_steps)


        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        state["iter_num"] += 1
        state["trained_tokens_total"] += tokens_per_global_micro_iter
        trained_tokens_exact = state["trained_tokens_total"]
        # input_id: B L 
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        if fabric.global_rank == 0 and state["iter_num"] % 10 == 0:
            total_tokens = trained_tokens_exact / 1e9
            peak_memory = 0.0
            if fabric.device.type == "cuda":
                peak_memory = torch.cuda.memory_stats(fabric.device)["allocated_bytes.all.peak"] / 1e9
            elapsed = max(t1 - train_t0, 1e-9)
            global_tokens_per_second = tokens * fabric.world_size / elapsed
            remaining_hours = max(args.max_tokens - trained_tokens_exact, 0) / max(global_tokens_per_second, 1e-9) / 3600
            fabric.print(
                    f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                    f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                    f" remaining time: {remaining_hours:.2f} hours. "
                    f" or {remaining_hours / 24:.2f} days. "
                    f" total training throughput {tokens / (t1 - train_t0) / 1e3:.2f}K tokens/s per GPU."
                    f" total trained tokens: {total_tokens} B tokens"
                    f" peak memory allocate {peak_memory:.2f} GB"
                )           
            
        estimated_flops = 1
        monitor.on_train_batch_end(
            state["iter_num"] * args.micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss = loss.item()
        )        

        # Exiting based on duration. 
        # credits: https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/e52bdabbde3c6895aceb76c1bced295c2646121f/megatron/training.py#L985-L998
        if not is_accumulating and args.actual_train_time:
            train_time = (time.time() - _TRAIN_START_TIME)
            # start monitoring sync
            done_cuda = torch.tensor([train_time > args.actual_train_time], device=fabric.device, dtype=torch.int)
            # force synchronization.
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                fabric.print(f"Training time {train_time/60.0} min, Reach time limit. Exiting ...")
                save_checkpoint()
                sys.exit()

        if not is_accumulating and state["step_count"] % args.save_step_interval == 0:
            save_checkpoint()

        if args.hf_upload and not is_accumulating:
            while (
                state["next_hf_upload_tokens"] > 0
                and trained_tokens_exact >= state["next_hf_upload_tokens"]
                and state["next_hf_upload_tokens"] <= args.max_tokens
            ):
                milestone_tokens = state["next_hf_upload_tokens"]
                save_hf_milestone(milestone_tokens)
                state["next_hf_upload_tokens"] += args.hf_upload_interval_tokens

        # First save ckpt then do eval in case ckpt is not saved in time
        if val_dataloader is not None and not is_accumulating and state["step_count"] % args.eval_step_interval == 0:            
            t0 = time.perf_counter()
            val_loss = validate(args, fabric, model, val_dataloader, args.eval_iters)
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            for i in range(args.num_extrapol):
                if fabric.global_rank == 0:
                    fabric.print(f"step {state['iter_num']} {i+1} x: val loss {val_loss[i]:.4f}, val time: {t1 * 1000:.2f}ms")        
                    fabric.log_dict({"metric/val_loss@"+str(i+1)+"x": val_loss[i].item()}, state["step_count"])
                    fabric.log_dict({"metric/val_ppl@"+str(i+1)+"x": math.exp(val_loss[i].item())}, state["step_count"])

            fabric.barrier()
    
    save_checkpoint(final=True)

# each gpu will run validation on the entire val_dataset to avoid headache.
@torch.no_grad()
def validate(args, fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, eval_iters=100) -> torch.Tensor:
    if fabric.global_rank == 0:
        fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(args.num_extrapol, device=fabric.device, dtype=torch.float64)
    num_sample = 0
    eval_lengths = [4096, 8192, 12288, 16384][:args.num_extrapol]
    for k, val_data in enumerate(val_dataloader):
        if eval_iters is not None and k >= eval_iters:
            break
        num_sample += 1
        for i, length in enumerate(eval_lengths):
            if args.use_stream_tok:
                input_ids = val_data['input_ids'][:, 0:length].contiguous().to(fabric.device)
                targets = val_data['labels'][:, 1:length + 1].contiguous().to(fabric.device)
            else:
                input_ids = val_data[:, 0 : length].contiguous()
                targets = val_data[:, 1 : length + 1].contiguous()
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            # running average to avoid overflow
            losses[i] += (loss.item() - losses[i]) / num_sample
    fabric.print(f"Validation loss: {losses}")
    model.train()
    return losses


def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        #filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        filenames = sorted(glob.glob(os.path.join(data_dir,f"{prefix}*")))
        random.seed(seed)
        random.shuffle(filenames)
        if split != "train":
            n_chunks = - (8 // -nodes) # ceil division
        else:
            n_chunks = 8
        dataset = PackedDataset(
            filenames,
            n_chunks=n_chunks,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed+fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train"
    )
    val_dataloader = (
        create_dataloader(
            batch_size=- (batch_size // -2), # ceil division
            block_size=  16384 + 1, #num_extrapol * block_size + 1, # val 4* extrapolation
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with linear warmup)
def get_lr(args, it: int, warmup_iters: int, max_iters: int) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    devices = int(os.getenv("DEVICES", str(torch.cuda.device_count() or 1)))
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='LLM Training')
    group = parser.add_argument_group('hyperparameters')
    group.add_argument('--output_root', default=str(REPO_ROOT / "runs"), type=str, help='output root directory')
    group.add_argument('--wandb_dir', default='', type=str, help='wandb directory')
    group.add_argument('--wandb_project', default='gated-deltanet-2', type=str, help='wandb project name')
    group.add_argument('--wandb_mode', default=os.getenv("WANDB_MODE", "disabled"), choices=["online", "offline", "disabled"], help='wandb logging mode')
    group.add_argument('--train_data_dir', default=str(REPO_ROOT / "data/fineweb-edu/data"), type=str, help='training data directory')
    group.add_argument('--corpus_name', default='fineweb-edu', type=str, help='corpus name')
    group.add_argument('--train_data_dir_raw', default=str(REPO_ROOT / "data/fineweb-edu/data"), type=str, help='training data directory (raw file for stream tok)')
    group.add_argument('--val_data_dir', default='', type=str, help='validation data directory')
    group.add_argument('--val_data_dir_raw', default='', type=str, help='validation data directory (raw file for stream tok)')
    group.add_argument('--model_name', default='gdn2_1.3B', type=str, help='model name')
    group.add_argument('--exp_name', default='gdn2_1.3B_fineweb_edu_100bt', type=str, help='experiment name')
    group.add_argument('--exp_group', default='fineweb_edu_100bt', type=str, help='experiment group name')
    group.add_argument('--train_config', default='tsz128x4k_100B', type=str, help='training config')
    group.add_argument('--resume', action='store_true', default=False, help='resume flag')
    group.add_argument('--debug', action='store_true', default=False, help='debug flag')
    group.add_argument('--interactive_job', action='store_true', default=False, help='debug flag')
    group.add_argument('--use_stream_tok', action=argparse.BooleanOptionalAction, default=True, help='stream and tokenize raw parquet files')
    group.add_argument('--tokenizer_name', type=str, default='TinyLlama/TinyLlama_v1.1')
    group.add_argument('--tokenizer_path', type=str, default='')
    group.add_argument('--learning_rate', type=float, default=4e-4, help='learning rate')
    group.add_argument('--total_evals', type=int, default=400, help='total number of evals')
    group.add_argument('--eval_iters', type=int, default=15, help='number of evaluation iterations')
    group.add_argument('--log_step_interval', type=int, default=10, help='log_step_interval')
    group.add_argument('--save_step_interval', type=int, default=1000, help='save_step_interval')
    group.add_argument('--eval_step_interval', type=int, default=1000, help='eval_step_interval')
    group.add_argument('--seed', type=int, default=3407, help='seed')
    group.add_argument('--num_extrapol', type=int, default=4, help='num_extrapol')
    group.add_argument('--weight_decay', type=float, default=1e-1, help='weight decay')
    group.add_argument('--beta1', type=float, default=0.9, help='beta1')
    group.add_argument('--beta2', type=float, default=0.95, help='beta2')
    group.add_argument('--grad_clip', type=float, default=1.0, help='gradient clip')
    group.add_argument('--val_type', default='none', type=str, help='use none to skip validation, or val_sampled for a parquet validation set')
    group.add_argument('--eval_before_training', action='store_true', default=False, help='do validation before the training starts')
    group.add_argument('--nnodes', type=int, default=None, help='number of nodes')
    group.add_argument('--train_num_workers', type=int, default=8)
    group.add_argument('--val_num_workers', type=int, default=1)
    group.add_argument('--actual_train_time', type=int, default=0, help='optional wall-clock limit in minutes; 0 means train until max_tokens')
    group.add_argument('--micro_batch_size', type=int, default=0, help='micro batch size')
    group.add_argument('--global_batch_tokens', type=int, default=0, help='target tokens per optimizer step; 0 infers it from train_config')
    group.add_argument('--max_tokens_override', type=int, default=0, help='override token budget parsed from train_config')
    group.add_argument('--activation_checkpointing', choices=['auto', 'on', 'off'], default='auto', help='activation checkpointing mode')
    group.add_argument('--expected_tokens_per_sec', type=float, default=0.0, help='optional ETA throughput assumption')
    group.add_argument('--data_shuffle_seed', type=int, default=3407, help='seed for deterministic streaming dataset shuffle')
    group.add_argument('--data_shuffle_buffer', type=int, default=0, help='HF streaming shuffle buffer; 0 disables dataset-level shuffle')
    group.add_argument('--env_file', default=str(REPO_ROOT / ".env"), type=str, help='env file to load secrets such as HF_TOKEN')
    group.add_argument('--hf_upload', action=argparse.BooleanOptionalAction, default=False, help='upload model-only milestones to Hugging Face Hub')
    group.add_argument('--hf_repo_id', default=os.getenv("HF_REPO_ID", ""), type=str, help='HF repo id; if empty, infer from token owner and exp_name')
    group.add_argument('--hf_upload_interval_tokens', type=int, default=1_000_000_000, help='upload every N trained tokens')
    group.add_argument('--hf_private', action=argparse.BooleanOptionalAction, default=False, help='create/use a private HF repo')
    group.add_argument('--hf_upload_blocking', action=argparse.BooleanOptionalAction, default=False, help='block training while uploading milestones')
    group.add_argument('--resume_trained_tokens', type=int, default=int(os.getenv("RESUME_TRAINED_TOKENS", "0")), help='override trained token counter after loading a resume checkpoint')

    args = parser.parse_args()
    name = f"{args.train_config}_{args.exp_name}".strip("_")
    args.output_root = str(Path(args.output_root).expanduser().resolve())
    args.out_dir = str(Path(args.output_root) / 'outputs' / name)
    if not args.wandb_dir:
        args.wandb_dir = str(Path(args.output_root) / 'wandb' / name)
    else:
        args.wandb_dir = str(Path(args.wandb_dir).expanduser().resolve())

    train_data_config = [("train_slim", 1.0)]
    val_data_config = [("validation", 1.0)]

    nodes = args.nnodes or int(os.getenv("SLURM_NNODES", "1"))
    args.nodes = nodes

    sequence_batch, sequence_length = parse_sequence_batch(args.train_config)
    args.sequence_length = sequence_length
    if args.activation_checkpointing == 'auto':
        args.activation_checkpointing = sequence_length >= 32768
    else:
        args.activation_checkpointing = args.activation_checkpointing == 'on'
    model_config = Config.from_name(
        args.model_name,
        block_size=sequence_length,
        activation_checkpointing=args.activation_checkpointing,
    )

    max_tokens = args.max_tokens_override or token_budget_from_config(name)
    micro_batch_size = default_micro_batch_size(sequence_length, args.model_name)
    if args.micro_batch_size == 0:
        args.micro_batch_size = micro_batch_size

    args.min_lr = args.learning_rate / 10
    target_global_batch_tokens = args.global_batch_tokens or sequence_batch * sequence_length
    tokens_per_micro_step = args.micro_batch_size * sequence_length * devices * nodes
    gradient_accumulation_steps = max(1, math.ceil(target_global_batch_tokens / tokens_per_micro_step))
    effective_global_batch_tokens = tokens_per_micro_step * gradient_accumulation_steps
    args.batch_size = args.micro_batch_size * gradient_accumulation_steps
    args.target_global_batch_tokens = target_global_batch_tokens
    args.global_batch_tokens = effective_global_batch_tokens
    log_iter_interval = args.log_step_interval * gradient_accumulation_steps
    args.gradient_accumulation_steps = gradient_accumulation_steps

    args.actual_train_time = args.actual_train_time * 60 # convert to seconds
    args.hf_repo_id_resolved = args.hf_repo_id
    args.warmup_tokens = int(1e9) if max_tokens == int(1e11) else int(max_tokens * 0.01)
    args.max_tokens = max_tokens
    args.log_iter_interval= log_iter_interval
    hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
    args.hparams = hparams
    main(args)
