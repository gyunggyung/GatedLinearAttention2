# gdn2_kla_1.3B: Single Candidate Design

This document defines the only experimental candidate in this workspace:

```text
gdn2_kla_1.3B
```

It is a recurrent-only linear attention model. It does not use hybrid attention,
SWA, MLA, or full softmax attention layers.

## Why This Candidate Is Needed

Standard softmax attention stores an explicit KV cache. This is strong for exact
retrieval, but long-context decoding memory grows with sequence length. Linear
attention instead compresses history into a fixed-size recurrent state:

```math
S_t = S_{t-1} + k_t v_t^\top,
\qquad
o_t = S_t^\top q_t.
```

The benefit is linear-time prefill and constant-size recurrent state at decode
time. The cost is interference: many key-value associations share the same
finite matrix `S_t`.

GDN-2 improves this memory editing problem by separating the active edit into
key-side erase and value-side write gates. Kaczmarz Linear Attention improves a
different part of the same update: the magnitude of the delta-rule step. This
candidate combines those two ideas.

The reason to avoid a hybrid model here is simple: Qwen3.5/Qwen3-Next and Kimi
Linear already compete in the hybrid design space. A recurrent-only candidate
tests whether the memory update itself can be improved.

## GDN-2 Recurrence

Let:

- `S_t in R^{d_k x d_v}` be the recurrent fast-weight memory state.
- `q_t, k_t in R^{d_k}` be query and key.
- `v_t in R^{d_v}` be value.
- `D_t = Diag(alpha_t)` be channel-wise decay.
- `b_t in [0, 1]^{d_k}` be the erase gate.
- `w_t in [0, 1]^{d_v}` be the write gate.

GDN-2 first decays the state:

```math
\bar S_t = D_t S_{t-1}.
```

It reads old content through the erase-gated key direction:

```math
r_t = \bar S_t^\top (b_t \odot k_t).
```

It writes only the selected value channels:

```math
z_t = w_t \odot v_t.
```

The update is:

```math
S_t = \bar S_t + k_t (z_t - r_t)^\top.
```

Equivalently:

```math
S_t
=
\left(I - k_t (b_t \odot k_t)^\top \right) D_t S_{t-1}
+
k_t (w_t \odot v_t)^\top.
```

This is better than scalar-gated GDN/KDA because erase and write live on
different axes. Erase is key-side. Write is value-side. A single scalar gate
forces both decisions to share one strength, which is a modeling bottleneck.

## Kaczmarz Step

The delta update can be viewed as an online projection problem. For a local
key-value association, the model wants:

```math
S_t^\top k_t \approx z_t.
```

A Kaczmarz projection step normalizes the update by the squared key norm:

```math
\lambda_t = \frac{\eta_t}{\|k_t\|_2^2 + \epsilon},
\qquad
\eta_t \in [0, 1].
```

This makes the update less sensitive to key scale. Without this normalization,
large-norm keys can erase/write too aggressively and small-norm keys can barely
update the state. In a long context, this matters because such scale errors
accumulate over many recurrent updates.

## Candidate Recurrence

`gdn2_kla_1.3B` folds the Kaczmarz step into GDN-2's erase/write gates:

```math
\tilde b_t = \lambda_t b_t,
\qquad
\tilde w_t = \lambda_t w_t.
```

Then:

```math
S_t
=
\left(I - k_t (\tilde b_t \odot k_t)^\top \right) D_t S_{t-1}
+
k_t (\tilde w_t \odot v_t)^\top.
```

Expanded:

```math
S_t
=
\left(I - k_t (\lambda_t b_t \odot k_t)^\top \right) D_t S_{t-1}
+
k_t (\lambda_t w_t \odot v_t)^\top.
```

This keeps the GDN-2 kernel shape unchanged:

- state shape is still `d_k x d_v`;
- erase gate is still key-channel-wise;
- write gate is still value-channel-wise;
- the chunkwise algorithm can consume the same `b` and `w` tensors;
- no attention/SWA block is introduced.

## Why It Can Help Long Context

Long context is hard for recurrent linear attention because the state is fixed
size. The model must repeatedly decide:

1. what old associations to decay;
2. what stale association to erase;
3. what new value channels to write;
4. how strong the edit should be.

GDN-2 addresses points 1-3. The Kaczmarz step addresses point 4.

This is useful for long-context state tracking in the same broad sense as GDN,
Mamba, and RNN-style models: information flows through a recurrent state rather
than only through a bounded feedforward depth. The memory is available at every
later token through the state transition.

It is not the same as full attention. Full attention can re-read exact past
tokens. This candidate compresses history. Therefore:

- it should be strong for state tracking and repeated memory updates;
- it should be efficient for very long decoding contexts;
- it may still lose to attention on exact rare-token retrieval if the state
  capacity is insufficient.

The hypothesis is narrower and testable: key-norm-normalized GDN-2 updates
should reduce long-context memory interference versus plain recurrent GDN-2.

## How To Run

Readiness check:

```bash
scripts/check_pretrain_readiness.py \
  --train-config tsz128x4k_10B \
  --model-name gdn2_kla_1.3B \
  --devices 8 \
  --throughput 250000
```

Train:

```bash
./scripts/pretrain_gdn2_kla_10bt.sh
```

Optional W&B:

```bash
WANDB_MODE=online ./scripts/pretrain_gdn2_kla_10bt.sh
```

The script sets:

```text
MODEL=gdn2_kla_1.3B
TRAIN_CONFIG=tsz128x4k_10B
EXP_NAME=gdn2_kla_1.3B_fineweb_edu_10bt
EXP_GROUP=fineweb_edu_10bt
```

## Expected Runtime

For 10B tokens at 4K sequence length on 8 H200 GPUs:

| Total tokens/s | Expected time |
|---:|---:|
| 100K | 27.8 hours |
| 200K | 13.9 hours |
| 250K | 11.1 hours |
| 300K | 9.3 hours |
| 500K | 5.6 hours |

One day requires at least about 116K total tokens/s.

This first run intentionally uses 4K sequence length to match the published
GDN-2 pretraining recipe. Longer contexts should be tested later as continuation
or evaluation runs, not mixed into the first fair comparison.

## Evaluation

Do not claim success from training loss alone. Check:

- validation loss against plain `gdn2_1.3B`;
- RULER S-NIAH and MK-NIAH;
- number-range tracking;
- entity-state update probes;
- long-context stability and throughput.

If it improves retrieval/state-tracking without hurting loss or throughput too
much, it is a real candidate. If not, the Kaczmarz step should be treated as an
ablation, not a new architecture.
