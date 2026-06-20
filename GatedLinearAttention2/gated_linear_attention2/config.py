from __future__ import annotations

from dataclasses import dataclass


def _find_multiple(value: int, multiple: int) -> int:
    if value % multiple == 0:
        return value
    return value + multiple - value % multiple


@dataclass
class GatedLinearAttention2Config:
    vocab_size: int = 32000
    padding_multiple: int = 64
    padded_vocab_size: int | None = None
    block_size: int = 4096
    n_layer: int = 18
    n_embd: int = 2304
    intermediate_size: int = 6208
    norm_eps: float = 1e-5
    bias: bool = False

    num_heads: int = 16
    num_v_heads: int | None = None
    head_dim: int = 128
    expand_v: float = 1.0
    conv_size: int = 4
    conv_bias: bool = False
    use_short_conv: bool = True
    allow_neg_eigval: bool = False
    use_qk_l2norm: bool = False
    use_kaczmarz_step: bool = True
    kaczmarz_eps: float = 1e-6

    tokenizer_name: str = "gyung/Gated_Linear_Attention2"
    tokenizer_subfolder: str = "tokenizer"

    def __post_init__(self) -> None:
        if self.padded_vocab_size is None:
            self.padded_vocab_size = _find_multiple(self.vocab_size, self.padding_multiple)
        if self.num_v_heads is None:
            self.num_v_heads = self.num_heads
        if self.n_embd <= 0 or self.n_layer <= 0:
            raise ValueError("n_embd and n_layer must be positive.")
        if self.head_dim <= 0 or self.num_heads <= 0:
            raise ValueError("head_dim and num_heads must be positive.")
        value_dim = int(self.num_v_heads * self.head_dim * self.expand_v)
        if value_dim <= 0:
            raise ValueError("value dimension must be positive.")

    @property
    def head_k_dim(self) -> int:
        return self.head_dim

    @property
    def head_v_dim(self) -> int:
        return int(self.head_dim * self.expand_v)

    @property
    def key_dim(self) -> int:
        return self.num_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        assert self.num_v_heads is not None
        return self.num_v_heads * self.head_v_dim

    @classmethod
    def gdn2_kla_1_3b(cls, **kwargs) -> "GatedLinearAttention2Config":
        values = dict(
            vocab_size=32000,
            padding_multiple=64,
            block_size=4096,
            n_layer=18,
            n_embd=2304,
            intermediate_size=6208,
            norm_eps=1e-5,
            bias=False,
            num_heads=16,
            num_v_heads=16,
            head_dim=128,
            expand_v=1.0,
            conv_size=4,
            conv_bias=False,
            use_short_conv=True,
            allow_neg_eigval=False,
            use_qk_l2norm=False,
            use_kaczmarz_step=True,
            kaczmarz_eps=1e-6,
            tokenizer_name="gyung/Gated_Linear_Attention2",
            tokenizer_subfolder="tokenizer",
        )
        values.update(kwargs)
        return cls(**values)
