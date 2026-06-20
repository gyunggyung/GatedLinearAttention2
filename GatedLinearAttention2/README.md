# GatedLinearAttention2 Runtime

Independent PyTorch runtime for the `Gated_Linear_Attention2` weights.

This folder is intentionally separate from the training code. It does not import
`lit_gpt`, `fla`, or the NVIDIA GatedDeltaNet-2 Triton kernels. The implementation
is a clean PyTorch reference runtime for inference and checkpoint loading.

## License Split

- Runtime code in this folder: Apache-2.0.
- Hugging Face model weights: Apache-2.0.
- Original training repository code outside this folder may still contain
  NVIDIA-derived code under `Nvidia Source Code License-NC`.

Commercial use should depend on this independent runtime or another compatible
implementation, not the NVIDIA-derived training runtime.

## Install

From this folder:

```bash
pip install -e .
pip install transformers huggingface_hub
```

The tokenizer used by the current training run is uploaded inside the model repo:

```text
gyung/Gated_Linear_Attention2/tokenizer
```

It is the tokenizer used during training, saved into the model repository so the
runtime does not need a second model repo at inference time. The original source
is `TinyLlama/TinyLlama_v1.1`.

## Generate Text

```bash
python examples/generate.py \
  --repo-id gyung/Gated_Linear_Attention2 \
  --checkpoint checkpoints/checkpoint-01B/model-ckpt.pth \
  --prompt "Artificial intelligence can help education by" \
  --max-new-tokens 80
```

The checkpoint is a LitGPT/Fabric `.pth` file stored on Hugging Face. It is not a
`transformers.AutoModelForCausalLM` checkpoint.

## Python API

```python
import torch
from gated_linear_attention2 import GatedLinearAttention2ForCausalLM, load_tokenizer

model = GatedLinearAttention2ForCausalLM.from_hf(
    repo_id="gyung/Gated_Linear_Attention2",
    checkpoint="checkpoints/checkpoint-01B/model-ckpt.pth",
    device="cuda",
    dtype=torch.bfloat16,
)
tokenizer = load_tokenizer("gyung/Gated_Linear_Attention2", subfolder="tokenizer")

ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    logits = model(ids)
next_id = int(logits[:, -1].argmax(dim=-1)[0])
print(tokenizer.decode([next_id]))
```

For autoregressive decoding, use the helper:

```python
from gated_linear_attention2.generation import generate

out = generate(model, tokenizer, "Once upon a time", max_new_tokens=64)
print(out)
```

## Context And Cache Behavior

The recurrent mixer keeps a fixed-size state per layer instead of a growing KV
cache. During decoding, cache memory is constant with respect to generated token
length:

```text
state per layer ~= batch * heads * key_dim * value_dim
```

For the 1.3B config:

```text
18 layers, 16 heads, 128 key dim, 128 value dim
```

The reference runtime supports any input length that fits memory and time. The
released checkpoint was trained at `4096` tokens, so quality beyond that length
is an extrapolation and must be evaluated separately.

## Performance

This is a correctness-first eager PyTorch runtime. It is useful for Apache-2.0
loading, generation, debugging, and clean downstream integration. It is not yet
as fast as the training Triton kernels. The next production step is to replace
the scan in `gated_linear_attention2/model.py` with an independently written
CUDA/Triton kernel under Apache-2.0 while preserving the same public API.
