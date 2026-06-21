#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMEXPR_MAX_THREADS", "256")

import torch
import torch.nn.functional as F
from lm_eval import evaluator, utils
from lm_eval.api.model import LM
from lm_eval.tasks import TaskManager
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lit_gpt.model import Config, GPT  # noqa: E402


def parse_dtype(name: str) -> torch.dtype:
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in aliases:
        raise ValueError(f"Unsupported dtype: {name}")
    return aliases[name]


def normalize_state_dict(raw: Any) -> dict[str, torch.Tensor]:
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    if not isinstance(state, dict):
        raise TypeError("checkpoint must be a state_dict or a dict containing 'model'")

    normalized: dict[str, torch.Tensor] = {}
    prefixes = ("_forward_module.", "module.", "_orig_mod.", "model.")
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        normalized[new_key] = value
    return normalized


class GatedLinearAttention2LM(LM):
    def __init__(
        self,
        checkpoint: str,
        model_name: str = "gdn2_kla_1.3B",
        tokenizer_name: str = "TinyLlama/TinyLlama_v1.1",
        max_length: int = 4096,
        device: str = "cuda",
        dtype: str = "bf16",
        strict: bool = True,
        eval_batch_size: int = 1,
    ) -> None:
        super().__init__()
        self._device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self._dtype = parse_dtype(dtype)
        self.max_length = int(max_length)
        self.batch_size = max(1, int(eval_batch_size))
        self.logits_cache = False
        self.backend = "causal"
        self.checkpoint = str(checkpoint)
        self._tokenizer_name = tokenizer_name

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.prefix_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        if self.prefix_token_id is None:
            raise RuntimeError("Tokenizer must provide a BOS or EOS token for rolling perplexity.")

        config = Config.from_name(model_name)
        config.activation_checkpointing = False
        model = GPT(config)
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = normalize_state_dict(raw)
        incompatible = model.load_state_dict(state, strict=strict)
        if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            print(
                "Non-strict load:",
                {"missing": incompatible.missing_keys, "unexpected": incompatible.unexpected_keys},
                file=sys.stderr,
            )
        model.to(device=self._device, dtype=self._dtype)
        model.eval()
        self.model = model

    @property
    def eot_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id or self.prefix_token_id)

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    @property
    def device(self) -> torch.device:
        return self._device

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        left_truncate_len: int | None = None,
        **_: Any,
    ) -> list[int]:
        if add_special_tokens is None:
            add_special_tokens = False
        tokens = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            tokens = tokens[-left_truncate_len:]
        return tokens

    def tok_decode(self, tokens, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    @torch.no_grad()
    def _model_logits(self, input_ids: list[int]) -> torch.Tensor:
        if not input_ids:
            input_ids = [self.prefix_token_id]
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length :]
        tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self._dtype,
            enabled=self.device.type == "cuda" and self._dtype in (torch.bfloat16, torch.float16),
        ):
            return self.model(tensor)

    @torch.no_grad()
    def _model_logits_batch(self, input_ids_batch: list[list[int]]) -> tuple[torch.Tensor, list[int]]:
        sanitized: list[list[int]] = []
        lengths: list[int] = []
        for input_ids in input_ids_batch:
            if not input_ids:
                input_ids = [self.prefix_token_id]
            if len(input_ids) > self.max_length:
                input_ids = input_ids[-self.max_length :]
            sanitized.append(input_ids)
            lengths.append(len(input_ids))

        max_len = max(lengths)
        pad_id = int(self.tokenizer.pad_token_id or self.eot_token_id)
        tensor = torch.full((len(sanitized), max_len), pad_id, dtype=torch.long, device=self.device)
        for row, input_ids in enumerate(sanitized):
            tensor[row, : len(input_ids)] = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self._dtype,
            enabled=self.device.type == "cuda" and self._dtype in (torch.bfloat16, torch.float16),
        ):
            return self.model(tensor), lengths

    def _score_token_ids(self, context_enc: list[int], continuation_enc: list[int]) -> tuple[float, bool]:
        if not continuation_enc:
            return 0.0, True
        if not context_enc:
            context_enc = [self.prefix_token_id]
        full = context_enc + continuation_enc
        sliced = full[-(self.max_length + 1) :]
        if len(sliced) < 2:
            return 0.0, True

        input_ids = sliced[:-1]
        targets = sliced[1:]
        first_scored = max(0, len(context_enc) - (len(full) - len(sliced)) - 1)
        scored_targets = targets[first_scored:]
        if not scored_targets:
            return 0.0, True

        logits = self._model_logits(input_ids)[0, -len(targets) :, :]
        logits = logits[first_scored:, : self.tokenizer.vocab_size].float()
        target_tensor = torch.tensor(scored_targets, dtype=torch.long, device=logits.device)
        log_probs = F.log_softmax(logits, dim=-1)
        token_logprobs = log_probs.gather(-1, target_tensor[:, None]).squeeze(-1)
        greedy = torch.argmax(logits, dim=-1)
        return float(token_logprobs.sum().item()), bool(torch.equal(greedy, target_tensor))

    def _prepare_score(self, context_enc: list[int], continuation_enc: list[int]) -> dict[str, Any]:
        if not continuation_enc:
            return {"empty": True}
        if not context_enc:
            context_enc = [self.prefix_token_id]
        full = context_enc + continuation_enc
        sliced = full[-(self.max_length + 1) :]
        if len(sliced) < 2:
            return {"empty": True}

        input_ids = sliced[:-1]
        targets = sliced[1:]
        first_scored = max(0, len(context_enc) - (len(full) - len(sliced)) - 1)
        scored_targets = targets[first_scored:]
        if not scored_targets:
            return {"empty": True}
        return {
            "empty": False,
            "input_ids": input_ids,
            "targets": targets,
            "first_scored": first_scored,
            "scored_targets": scored_targets,
        }

    def _score_prepared_batch(self, prepared_batch: list[dict[str, Any]]) -> list[tuple[float, bool]]:
        active = [item for item in prepared_batch if not item["empty"]]
        if not active:
            return [(0.0, True) for _ in prepared_batch]

        logits_batch, _ = self._model_logits_batch([item["input_ids"] for item in active])
        active_results: list[tuple[float, bool]] = []
        for row, item in enumerate(active):
            targets = item["targets"]
            first_scored = item["first_scored"]
            scored_targets = item["scored_targets"]
            logits = logits_batch[row, : len(item["input_ids"]), : self.tokenizer.vocab_size]
            logits = logits[-len(targets) :, :].float()
            logits = logits[first_scored:, :]
            target_tensor = torch.tensor(scored_targets, dtype=torch.long, device=logits.device)
            log_probs = F.log_softmax(logits, dim=-1)
            token_logprobs = log_probs.gather(-1, target_tensor[:, None]).squeeze(-1)
            greedy = torch.argmax(logits, dim=-1)
            active_results.append((float(token_logprobs.sum().item()), bool(torch.equal(greedy, target_tensor))))

        out: list[tuple[float, bool]] = []
        active_idx = 0
        for item in prepared_batch:
            if item["empty"]:
                out.append((0.0, True))
            else:
                out.append(active_results[active_idx])
                active_idx += 1
        return out

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        prepared: list[dict[str, Any]] = []
        for request in requests:
            context, continuation = request.args
            prepared.append(self._prepare_score(self.tok_encode(context), self.tok_encode(continuation)))
        for start in range(0, len(prepared), self.batch_size):
            results.extend(self._score_prepared_batch(prepared[start : start + self.batch_size]))
        return results

    def loglikelihood_rolling(self, requests) -> list[float]:
        results: list[float] = []
        for request in requests:
            (text,) = request.args
            windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(text),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            total = 0.0
            for window in windows:
                if len(window) == 3:
                    _, context_enc, continuation_enc = window
                elif len(window) == 2:
                    context_enc, continuation_enc = window
                else:
                    raise ValueError(f"Unexpected rolling window shape: {len(window)}")
                score, _ = self._score_token_ids(context_enc, continuation_enc)
                total += score
            results.append(total)
        return results

    @torch.no_grad()
    def generate_until(self, requests) -> list[str]:
        outputs: list[str] = []
        for batch_start in range(0, len(requests), self.batch_size):
            batch = requests[batch_start : batch_start + self.batch_size]
            contexts: list[list[int]] = []
            untils: list[list[str]] = []
            max_gen_toks = 0
            for request in batch:
                context, gen_kwargs = request.args
                until = gen_kwargs.get("until", []) or []
                if isinstance(until, str):
                    until = [until]
                untils.append(list(until))
                max_gen_toks = max(max_gen_toks, int(gen_kwargs.get("max_gen_toks", gen_kwargs.get("max_new_tokens", 32))))
                contexts.append(self.tok_encode(context)[-self.max_length :])

            generated: list[list[int]] = [[] for _ in batch]
            finished = [False for _ in batch]
            texts = ["" for _ in batch]
            for _ in range(max_gen_toks):
                active_indices = [idx for idx, done in enumerate(finished) if not done]
                if not active_indices:
                    break
                input_batch = [(contexts[idx] + generated[idx])[-self.max_length :] for idx in active_indices]
                logits_batch, lengths = self._model_logits_batch(input_batch)
                for row, idx in enumerate(active_indices):
                    logits = logits_batch[row, lengths[row] - 1, : self.tokenizer.vocab_size]
                    next_id = int(torch.argmax(logits.float(), dim=-1).item())
                    generated[idx].append(next_id)
                    text = self.tok_decode(generated[idx])
                    if untils[idx] and any(stop in text for stop in untils[idx]):
                        for stop in untils[idx]:
                            if stop in text:
                                text = text.split(stop)[0]
                                break
                        texts[idx] = text
                        finished[idx] = True
                    elif next_id == self.eot_token_id:
                        texts[idx] = text
                        finished[idx] = True
                    else:
                        texts[idx] = text

            outputs.extend(texts)
        return outputs

    @torch.no_grad()
    def generate_until_slow(self, requests) -> list[str]:
        outputs: list[str] = []
        for request in requests:
            context, gen_kwargs = request.args
            until = gen_kwargs.get("until", []) or []
            if isinstance(until, str):
                until = [until]
            max_gen_toks = int(gen_kwargs.get("max_gen_toks", gen_kwargs.get("max_new_tokens", 32)))
            input_ids = self.tok_encode(context)[-self.max_length :]
            generated: list[int] = []
            for _ in range(max_gen_toks):
                logits = self._model_logits(input_ids + generated)[0, -1, : self.tokenizer.vocab_size]
                next_id = int(torch.argmax(logits.float(), dim=-1).item())
                generated.append(next_id)
                text = self.tok_decode(generated)
                if until and any(stop in text for stop in until):
                    for stop in until:
                        if stop in text:
                            text = text.split(stop)[0]
                            break
                    break
                if next_id == self.eot_token_id:
                    break
            outputs.append(self.tok_decode(generated))
        return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lm-eval on a GatedLinearAttention2 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", required=True, help="Comma-separated lm-eval task names")
    parser.add_argument("--model_name", default="gdn2_kla_1.3B")
    parser.add_argument("--tokenizer_name", default="TinyLlama/TinyLlama_v1.1")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--ruler_lengths", default="", help="Comma-separated max sequence lengths for RULER tasks")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    metadata: dict[str, Any] = {"tokenizer": args.tokenizer_name}
    if args.ruler_lengths:
        metadata["max_seq_lengths"] = [int(item) for item in args.ruler_lengths.split(",") if item.strip()]

    lm = GatedLinearAttention2LM(
        checkpoint=args.checkpoint,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length,
        device=args.device,
        dtype=args.dtype,
        strict=args.strict,
        eval_batch_size=args.batch_size,
    )
    task_manager = TaskManager(metadata=metadata)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
        task_manager=task_manager,
        log_samples=False,
        metadata={
            "checkpoint": args.checkpoint,
            "max_length": args.max_length,
            "tokenizer": args.tokenizer_name,
            "ruler_lengths": metadata.get("max_seq_lengths"),
        },
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
