#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

GITHUB_REPO_URL = "https://github.com/gyunggyung/Gated_Linear_Attention2"
LICENSE_BODY = """The model weights in this Hugging Face repository are released under Apache-2.0.

The standalone inference runtime linked above is also Apache-2.0. It does not
import `lit_gpt`, `fla`, or the NVIDIA GatedDeltaNet-2 Triton kernels. The
training code used during experimentation may contain NVIDIA GatedDeltaNet-2
derived components under `Nvidia Source Code License-NC`, but this Hugging Face
model repository is intended to be used with the standalone Apache-2.0 runtime."""

USAGE_BODY = """This is a causal language model: given a text prefix, it predicts the next token
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
memory does not grow with generated token length like a Transformer KV cache."""


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
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def normalize_license_header(text: str) -> str:
    text = text.replace(
        "license: other\nlicense_name: Nvidia Source Code License-NC\n",
        "license: apache-2.0\n",
    )
    text = text.replace("license: other\n", "license: apache-2.0\n")
    text = text.replace("- litgpt\n", "")
    text = text.replace("LitGPT/Fabric model-only checkpoint", "PyTorch model-only checkpoint")
    text = text.replace("LitGPT/Fabric PyTorch checkpoint", "PyTorch `.pth` checkpoint")
    return text


def upsert_section(text: str, heading: str, body: str, before_heading: str | None = None) -> str:
    section = f"{heading}\n\n{body.strip()}\n\n"
    pattern = rf"{re.escape(heading)}\n\n.*?(?=\n## |\Z)"
    if re.search(pattern, text, flags=re.S):
        return re.sub(pattern, section.rstrip() + "\n", text, count=1, flags=re.S)
    if before_heading and before_heading in text:
        return text.replace(before_heading, section + before_heading, 1)
    return text.rstrip() + "\n\n" + section.rstrip() + "\n"


def remove_section(text: str, heading: str) -> str:
    pattern = rf"\n?{re.escape(heading)}\n\n.*?(?=\n## |\Z)"
    return re.sub(pattern, "\n", text, count=1, flags=re.S).replace("\n\n\n", "\n\n")


def ensure_model_card_details(readme: Path) -> None:
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    text = normalize_license_header(text)
    text = remove_section(text, "## Loading Sketch")
    text = upsert_section(text, "## Code", f"- GitHub: {GITHUB_REPO_URL}", "## Training Setup")
    text = upsert_section(text, "## License", LICENSE_BODY, "## Training Setup")
    text = upsert_section(text, "## How To Use", USAGE_BODY, "## Evaluation Plan")
    readme.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a model checkpoint folder to Hugging Face Hub.")
    parser.add_argument("--folder", required=True, help="Local checkpoint folder to upload.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face Hub repo id, e.g. user/model.")
    parser.add_argument("--path-in-repo", required=True, help="Destination folder inside the repo.")
    parser.add_argument("--env-file", default=".env", help="Env file containing HF_TOKEN.")
    parser.add_argument("--private", action="store_true", help="Create the repo as private if it does not exist.")
    args = parser.parse_args()

    load_env_file(args.env_file)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set.")

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(folder)

    api = HfApi(token=token)
    try:
        api.repo_info(repo_id=args.repo_id, repo_type="model", token=token)
    except Exception:
        create_repo(args.repo_id, token=token, private=args.private, exist_ok=True)

    readme = folder / "README.md"
    ensure_model_card_details(readme)
    if readme.exists():
        upload_file(
            repo_id=args.repo_id,
            token=token,
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            commit_message=f"Update model card for {args.path_in_repo}",
        )

    upload_folder(
        repo_id=args.repo_id,
        token=token,
        folder_path=str(folder),
        path_in_repo=args.path_in_repo,
        commit_message=f"Upload {args.path_in_repo}",
    )

    api.create_tag(
        repo_id=args.repo_id,
        tag=args.path_in_repo.replace("/", "-"),
        tag_message=f"Milestone {args.path_in_repo}",
        exist_ok=True,
    )
    print(f"Uploaded {folder} to {args.repo_id}/{args.path_in_repo}")


if __name__ == "__main__":
    main()
