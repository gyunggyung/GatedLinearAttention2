from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + self.eps)
        return (y.to(dtype) * self.weight.to(dtype))


class RMSNormSwishGate(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + self.eps)
        y = y.to(dtype) * self.weight.to(dtype)
        return y * F.silu(gate)


class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, bias: bool = False) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class LLaMAMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False) -> None:
        super().__init__()
        self.swiglu = SwiGLU(hidden_size, intermediate_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 4, bias: bool = False, activation: bool = True) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.ndim != 3:
            raise ValueError("expected x with shape [batch, time, channels]")
        if x.size(-1) != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {x.size(-1)}")

        xt = x.transpose(1, 2)
        if cache is None:
            padded = F.pad(xt, (self.kernel_size - 1, 0))
        else:
            padded = torch.cat([cache.to(dtype=xt.dtype, device=xt.device), xt], dim=-1)

        y = F.conv1d(padded, self.weight.to(dtype=xt.dtype), self.bias, groups=self.channels)
        y = y.transpose(1, 2)
        if self.activation:
            y = F.silu(y)

        new_cache = None
        if use_cache:
            new_cache = padded[:, :, -(self.kernel_size - 1) :].detach()
        return y, new_cache
