from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .model import GatedLinearAttention2ForCausalLM


def load_tokenizer(
    repo_id: str = "gyung/Gated_Linear_Attention2",
    subfolder: str = "tokenizer",
    fallback: str = "TinyLlama/TinyLlama_v1.1",
):
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(fallback, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


@torch.no_grad()
def generate(
    model: GatedLinearAttention2ForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 50,
    eos_token_id: int | None = None,
) -> str:
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    logits, cache = model(input_ids, use_cache=True, return_cache=True)
    generated = input_ids
    next_logits = logits[:, -1, :]
    eos_id = tokenizer.eos_token_id if eos_token_id is None else eos_token_id

    for _ in range(max_new_tokens):
        if temperature <= 0:
            next_id = next_logits.argmax(dim=-1, keepdim=True)
        else:
            logits_step = next_logits / temperature
            if top_k > 0:
                values, _ = torch.topk(logits_step, k=min(top_k, logits_step.size(-1)))
                logits_step = logits_step.masked_fill(logits_step < values[:, [-1]], -float("inf"))
            probs = F.softmax(logits_step, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_id], dim=-1)
        if eos_id is not None and int(next_id[0]) == int(eos_id):
            break
        logits, cache = model(next_id, cache=cache, use_cache=True, return_cache=True)
        next_logits = logits[:, -1, :]

    return tokenizer.decode(generated[0], skip_special_tokens=True)
