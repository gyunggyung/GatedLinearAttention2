from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from .config import GatedLinearAttention2Config
from .layers import CausalDepthwiseConv1d, LLaMAMLP, RMSNorm, RMSNormSwishGate


class GatedLinearAttention2Mixer(nn.Module):
    def __init__(self, config: GatedLinearAttention2Config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.num_heads = config.num_heads
        self.num_v_heads = config.num_v_heads or config.num_heads
        self.key_dim = config.key_dim
        self.value_dim = config.value_dim
        self.use_short_conv = config.use_short_conv
        self.allow_neg_eigval = config.allow_neg_eigval
        self.use_qk_l2norm = config.use_qk_l2norm
        self.use_kaczmarz_step = config.use_kaczmarz_step
        self.kaczmarz_eps = config.kaczmarz_eps

        self.q_proj = nn.Linear(config.n_embd, self.key_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.key_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.value_dim, bias=False)

        if self.use_short_conv:
            self.q_conv1d = CausalDepthwiseConv1d(self.key_dim, config.conv_size, config.conv_bias, activation=True)
            self.k_conv1d = CausalDepthwiseConv1d(self.key_dim, config.conv_size, config.conv_bias, activation=True)
            self.v_conv1d = CausalDepthwiseConv1d(self.value_dim, config.conv_size, config.conv_bias, activation=True)

        self.f_proj = nn.Sequential(
            nn.Linear(config.n_embd, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        self.b_proj = nn.Linear(config.n_embd, self.key_dim, bias=False)
        self.w_proj = nn.Linear(config.n_embd, self.value_dim, bias=False)
        if self.use_kaczmarz_step:
            self.eta_proj = nn.Linear(config.n_embd, config.num_heads, bias=True)

        self.A_log = nn.Parameter(torch.zeros(config.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(self.key_dim, dtype=torch.float32))

        self.g_proj = nn.Sequential(
            nn.Linear(config.n_embd, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = RMSNormSwishGate(self.head_v_dim, eps=config.norm_eps)
        self.o_proj = nn.Linear(self.value_dim, config.n_embd, bias=False)

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        conv_cache: dict[str, torch.Tensor] | None = None
        if self.use_short_conv:
            q, q_cache = self.q_conv1d(q, None if cache is None else cache.get("q_conv"), use_cache)
            k, k_cache = self.k_conv1d(k, None if cache is None else cache.get("k_conv"), use_cache)
            v, v_cache = self.v_conv1d(v, None if cache is None else cache.get("v_conv"), use_cache)
            if use_cache:
                assert q_cache is not None and k_cache is not None and v_cache is not None
                conv_cache = {"q_conv": q_cache, "k_conv": k_cache, "v_conv": v_cache}
        else:
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)
        return q, k, v, conv_cache

    def _split_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        batch, time, _ = x.shape
        return x.view(batch, time, -1, head_dim)

    def _maybe_expand_key_heads(self, *items: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.num_v_heads <= self.num_heads:
            return items
        groups = self.num_v_heads // self.num_heads
        return tuple(item.repeat_interleave(groups, dim=2) for item in items)

    def _scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time, heads, key_dim = q.shape
        value_dim = v.size(-1)
        state_dtype = torch.float32
        if initial_state is None:
            state = torch.zeros(batch, heads, key_dim, value_dim, device=q.device, dtype=state_dtype)
        else:
            state = initial_state.to(device=q.device, dtype=state_dtype)

        outputs: list[torch.Tensor] = []
        for index in range(time):
            q_t = q[:, index].float()
            k_t = k[:, index].float()
            v_t = v[:, index].float()
            g_t = g[:, index].float()
            b_t = b[:, index].float()
            w_t = w[:, index].float()

            decayed = state * torch.exp(g_t).unsqueeze(-1)
            erase_vector = b_t * k_t
            erased_value = torch.einsum("bhd,bhdv->bhv", erase_vector, decayed)
            write_value = w_t * v_t
            state = decayed - k_t.unsqueeze(-1) * erased_value.unsqueeze(-2)
            state = state + k_t.unsqueeze(-1) * write_value.unsqueeze(-2)
            outputs.append(torch.einsum("bhd,bhdv->bhv", q_t, state).to(dtype=q.dtype))

        return torch.stack(outputs, dim=1), state.detach()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: dict[str, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        q, k, v, conv_cache = self._project_qkv(hidden_states, cache, use_cache)

        g = (
            -self.A_log.float().exp().repeat_interleave(self.head_k_dim)
            * F.softplus(self.f_proj(hidden_states).float() + self.dt_bias)
        )
        b = self.b_proj(hidden_states).sigmoid()
        w = self.w_proj(hidden_states).sigmoid()

        eta = None
        k_norm_sq = None
        if self.use_kaczmarz_step:
            k_heads = self._split_heads(k.float(), self.head_k_dim)
            k_norm_sq = k_heads.square().sum(dim=-1, keepdim=True).clamp_min(self.kaczmarz_eps)
            eta = self.eta_proj(hidden_states).sigmoid().unsqueeze(-1)

        q = self._split_heads(q, self.head_k_dim)
        k = self._split_heads(k, self.head_k_dim)
        g = self._split_heads(g, self.head_k_dim)
        b = self._split_heads(b, self.head_k_dim)
        v = self._split_heads(v, self.head_v_dim)
        w = self._split_heads(w, self.head_v_dim)

        if self.use_qk_l2norm:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)

        q, k, g, b = self._maybe_expand_key_heads(q, k, g, b)
        if eta is not None and k_norm_sq is not None:
            eta, k_norm_sq = self._maybe_expand_key_heads(eta, k_norm_sq)
            step = eta / k_norm_sq
            b = (b.float() * step).clamp(0.0, 1.0).to(dtype=b.dtype)
            w = (w.float() * step).clamp(0.0, 1.0).to(dtype=w.dtype)

        if self.allow_neg_eigval:
            b = b * 2.0

        initial_state = None if cache is None else cache.get("state")
        recurrent, final_state = self._scan(q, k, v, g, b, w, initial_state)
        gate = self._split_heads(self.g_proj(hidden_states), self.head_v_dim)
        out = self.o_norm(recurrent, gate)
        out = out.reshape(out.size(0), out.size(1), self.value_dim)
        out = self.o_proj(out)

        next_cache = None
        if use_cache:
            next_cache = {} if conv_cache is None else conv_cache
            next_cache["state"] = final_state
        return out, next_cache


class GLABlock(nn.Module):
    def __init__(self, config: GatedLinearAttention2Config) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = GatedLinearAttention2Mixer(config)
        self.norm_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = LLaMAMLP(config.n_embd, config.intermediate_size, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        attn_out, next_cache = self.attn(self.norm_1(x), cache=cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm_2(x))
        return x, next_cache


class GatedLinearAttention2ForCausalLM(nn.Module):
    def __init__(self, config: GatedLinearAttention2Config | None = None) -> None:
        super().__init__()
        self.config = config or GatedLinearAttention2Config.gdn2_kla_1_3b()
        assert self.config.padded_vocab_size is not None
        self.lm_head = nn.Linear(self.config.n_embd, self.config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(self.config.padded_vocab_size, self.config.n_embd),
                "h": nn.ModuleList([GLABlock(self.config) for _ in range(self.config.n_layer)]),
                "ln_f": RMSNorm(self.config.n_embd, eps=self.config.norm_eps),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: list[dict[str, torch.Tensor] | None] | None = None,
        use_cache: bool = False,
        return_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor] | None]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        x = self.transformer["wte"](input_ids)
        next_cache: list[dict[str, torch.Tensor] | None] = []
        if cache is None:
            cache = [None] * len(self.transformer["h"])
        if len(cache) != len(self.transformer["h"]):
            raise ValueError("cache length does not match layer count")

        for layer, layer_cache in zip(self.transformer["h"], cache):
            x, new_layer_cache = layer(x, cache=layer_cache, use_cache=use_cache or return_cache)
            next_cache.append(new_layer_cache)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)
        if return_cache:
            return logits, next_cache
        return logits

    @staticmethod
    def _normalize_state_dict(raw: Any) -> dict[str, torch.Tensor]:
        state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        if not isinstance(state, dict):
            raise TypeError("checkpoint must be a state_dict or contain a 'model' state_dict")

        normalized: dict[str, torch.Tensor] = {}
        prefixes = ("module.", "_orig_mod.", "model.")
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

    def load_litgpt_checkpoint(self, checkpoint_path: str | Path, strict: bool = True) -> Any:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = self._normalize_state_dict(raw)
        return self.load_state_dict(state, strict=strict)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: GatedLinearAttention2Config | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        strict: bool = True,
    ) -> "GatedLinearAttention2ForCausalLM":
        model = cls(config)
        model.load_litgpt_checkpoint(checkpoint_path, strict=strict)
        model.to(device=device, dtype=dtype)
        model.eval()
        return model

    @classmethod
    def from_hf(
        cls,
        repo_id: str = "gyung/Gated_Linear_Attention2",
        checkpoint: str = "checkpoints/checkpoint-01B/model-ckpt.pth",
        config: GatedLinearAttention2Config | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        strict: bool = True,
    ) -> "GatedLinearAttention2ForCausalLM":
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename=checkpoint)
        return cls.from_checkpoint(checkpoint_path, config=config, device=device, dtype=dtype, strict=strict)
