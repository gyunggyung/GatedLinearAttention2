# gdn2_kla_1.3B 10B Fair Run + HF Upload Runbook

This runbook documents the exact training job requested for the single
candidate:

```text
gdn2_kla_1.3B
```

The goal is a fair 10B-token ablation against the published GDN-2 recipe:

- same model scale: 1.3B
- same data source: FineWeb-Edu `sample/100BT`
- same sequence length: 4K
- same global batch tokens: 524,288
- deterministic streaming order with dataloader workers disabled
- lower token budget: 10B instead of 100B
- one architecture change: Kaczmarz-normalized GDN-2 erase/write gates

## Candidate

`gdn2_kla_1.3B` modifies GDN-2 by folding a Kaczmarz-style update-size
normalizer into the erase and write gates:

```math
\lambda_t = \frac{\eta_t}{\|k_t\|_2^2 + \epsilon}
```

```math
S_t =
\left(I - k_t(\lambda_t b_t \odot k_t)^\top\right)D_tS_{t-1}
+
k_t(\lambda_t w_t \odot v_t)^\top.
```

This is a recurrent-only linear attention model. It does not use full
attention, SWA, MLA, or RoPE-based context extension.

## Dataset Fairness

The old path opened the 100BT parquet files in sorted order and stopped when
the token budget was reached. That is acceptable for smoke tests, but it is not
ideal for a fair 10B ablation because the first 10B streamed tokens may depend
on shard/file order.

The current run uses deterministic streaming shuffle:

```text
DATA_SHUFFLE_SEED=3407
DATA_SHUFFLE_BUFFER=100000
```

Pipeline:

```text
FineWeb-Edu sample/100BT parquet files
-> HuggingFace streaming dataset
-> deterministic streaming shuffle(seed=3407, buffer=100000)
-> split_dataset_by_node(rank, world_size)
-> tokenizer
-> 4K token chunks
-> local chunk-buffer shuffle
-> stop at 10B trained tokens
```

This is not a fully materialized uniform 10B-document subset, but it is much
less order-biased than sorted streaming prefix. It is also reproducible: the
same seed and code path can be reused for a plain `gdn2_1.3B` 10B baseline.

For a publication-grade comparison, the next step would be a materialized
manifest:

```text
document ids / shard offsets -> deterministic random seed -> fixed 10B manifest
```

That manifest can then drive both baseline and candidate training.

The candidate launcher sets:

```text
TRAIN_NUM_WORKERS=0
```

This is deliberate for the first fair run. PyTorch-style iterable datasets can
be replicated across worker processes unless the dataset implements explicit
worker sharding. Rank-level sharding is already handled by
`split_dataset_by_node`, so disabling dataloader workers avoids accidental
sample duplication. If throughput is too low, add explicit worker sharding
before raising this value.

## Batch And GPU Settings

The 4K fair-comparison setting uses:

```text
TRAIN_CONFIG=tsz128x4k_10B
MICRO_BATCH_SIZE=16
ACTIVATION_CHECKPOINTING=on
DEVICES=8
TRAIN_NUM_WORKERS=0
GLOBAL_BATCH_TOKENS=524288
```

Why `MICRO_BATCH_SIZE=16`:

```text
16 * 4096 * 8 = 524,288 tokens
```

So the run uses all 8 H200 GPUs with the largest micro batch that preserves the
published GDN-2 global batch size. Larger micro batches would change the global
batch and make the comparison less clean.

In the actual H200 environment, `MICRO_BATCH_SIZE=16` without activation
checkpointing OOMs at the first forward pass. The candidate launcher therefore
enables:

```text
ACTIVATION_CHECKPOINTING=on
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

If this is still too tight, the fallback is `MICRO_BATCH_SIZE=8`, which keeps the
same effective global batch through `gradient_accumulation_steps=2`:

```text
8 * 4096 * 8 * 2 = 524,288 tokens
```

Expected schedule:

```text
sequence length: 4096
micro batch/GPU: 16
grad accumulation: 1
optimizer steps: 19,074
stop tokens: 10,000,269,312
```

## Hugging Face Upload

The run reads `HF_TOKEN` from `.env`. The token is never printed.

Upload defaults:

```text
HF_UPLOAD=1
HF_REPO_ID=Gated_Linear_Attention2
HF_UPLOAD_INTERVAL_TOKENS=1000000000
HF_PRIVATE=true
HF_UPLOAD_BLOCKING=false
```

That means:

- create/use a private HF repo;
- upload to `<HF token owner>/Gated_Linear_Attention2` by default;
- if `HF_REPO_ID` includes a namespace, use that exact repo id;
- save a model-only checkpoint at every 1B tokens;
- upload milestones asynchronously so training can continue;
- upload 10 milestones total for a 10B-token run.

Milestone layout in the HF repo:

```text
README.md
checkpoints/checkpoint-01B/
checkpoints/checkpoint-02B/
...
checkpoints/checkpoint-10B/
```

Each checkpoint folder contains:

```text
model-ckpt.pth
training_metadata.json
README.md
```

The HF checkpoint is model-only. The local `latest-model-ckpt.pth` remains the
resume checkpoint with optimizer state.

## Model Card

`pretrain.py` writes a model card snapshot at every milestone. The model card
states:

- this is not a `transformers.AutoModelForCausalLM` checkpoint;
- it is a LitGPT/Fabric checkpoint;
- how the architecture differs from GDN-2;
- training data source and token budget;
- tokenizer and sequence length;
- evaluation plan.

The upload helper also uploads that card to the repo root as `README.md`.

## Launch Command

Default command:

```bash
./scripts/pretrain_gdn2_kla_10bt.sh
```

Equivalent explicit environment:

```bash
DEVICES=8 \
MICRO_BATCH_SIZE=16 \
TRAIN_CONFIG=tsz128x4k_10B \
DATA_SHUFFLE_SEED=3407 \
DATA_SHUFFLE_BUFFER=100000 \
TRAIN_NUM_WORKERS=0 \
ACTIVATION_CHECKPOINTING=on \
HF_UPLOAD=1 \
HF_REPO_ID=Gated_Linear_Attention2 \
HF_UPLOAD_INTERVAL_TOKENS=1000000000 \
HF_PRIVATE=true \
./scripts/pretrain_gdn2_kla_10bt.sh
```

If a specific HF repo is desired:

```bash
HF_REPO_ID=your-username/Gated_Linear_Attention2 \
./scripts/pretrain_gdn2_kla_10bt.sh
```

## Logs

Recommended detached launch:

```bash
mkdir -p runs/logs
nohup ./scripts/pretrain_gdn2_kla_10bt.sh \
  > runs/logs/gdn2_kla_10bt_4k_hf.log 2>&1 &
echo $! > runs/logs/gdn2_kla_10bt_4k_hf.pid
```

Monitor:

```bash
tail -f runs/logs/gdn2_kla_10bt_4k_hf.log
tail -f runs/outputs/tsz128x4k_10B_gdn2_kla_1.3B_fineweb_edu_10bt/hf_upload.log
```

## Expected Time

For 10B tokens:

| Total tokens/s | Expected time |
|---:|---:|
| 100K | 27.8 hours |
| 200K | 13.9 hours |
| 250K | 11.1 hours |
| 300K | 9.3 hours |
| 500K | 5.6 hours |

The actual throughput must be read from logs after Triton compilation and the
first 100-200 iterations.

## Evaluation After Training

After the 10B run, compare with the GDN-2 paper tasks:

1. Language modeling
   - WikiText perplexity
   - LAMBADA perplexity
2. LAMBADA and commonsense accuracy
   - PIQA
   - HellaSwag
   - WinoGrande
   - ARC-e
   - ARC-c
   - OpenBookQA
   - SIQA
   - BoolQ
3. Synthetic RULER retrieval
   - S-NIAH-1/2/3 at the paper's lengths
   - MK-NIAH-1 at the paper's lengths
4. Real-world retrieval
   - SWDE
   - SQuAD
   - FDA
   - TriviaQA
   - NQ
   - DROP

The first claim should be narrow:

```text
At 10B FineWeb-Edu tokens and 4K training length, does Kaczmarz-GDN2 improve
plain recurrent GDN-2 under the same data/order/batch setup?
```

Only after this fair 4K run should the project move to:

```text
32K continuation -> 128K stress run -> 1M streaming eval wrapper
```
