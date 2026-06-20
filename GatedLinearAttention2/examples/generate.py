#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gated_linear_attention2 import GatedLinearAttention2Config, GatedLinearAttention2ForCausalLM
from gated_linear_attention2.generation import generate, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with GatedLinearAttention2.")
    parser.add_argument("--repo-id", default="gyung/Gated_Linear_Attention2")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint-01B/model-ckpt.pth")
    parser.add_argument("--tokenizer-repo", default="")
    parser.add_argument("--tokenizer-subfolder", default="tokenizer")
    parser.add_argument("--tokenizer-fallback", default="TinyLlama/TinyLlama_v1.1")
    parser.add_argument("--prompt", default="Artificial intelligence can help education by")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    tokenizer_repo = args.tokenizer_repo or args.repo_id
    config = GatedLinearAttention2Config.gdn2_kla_1_3b(
        tokenizer_name=tokenizer_repo,
        tokenizer_subfolder=args.tokenizer_subfolder,
    )
    model = GatedLinearAttention2ForCausalLM.from_hf(
        repo_id=args.repo_id,
        checkpoint=args.checkpoint,
        config=config,
        device=args.device,
        dtype=dtype,
    )
    tokenizer = load_tokenizer(
        repo_id=tokenizer_repo,
        subfolder=args.tokenizer_subfolder,
        fallback=args.tokenizer_fallback,
    )

    text = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)


if __name__ == "__main__":
    main()
