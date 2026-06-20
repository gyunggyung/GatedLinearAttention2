# Gated_Linear_Attention2

Fork of [NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2).

## Kaczmarz-GDN2 단일 후보 학습 가이드

현재 실험 후보는 하나입니다.

```text
gdn2_kla_1.3B
```

기존 GDN-2 100B baseline 실행 가이드는 [README.gdn2_100bt_baseline.md](README.gdn2_100bt_baseline.md)에 보존했습니다. 공식 원본 README는 [README.original.md](README.original.md)에 있습니다.

실제 8-GPU 10B 학습 runbook은 [docs/run_gdn2_kla_10b_fair_hf_upload.md](docs/run_gdn2_kla_10b_fair_hf_upload.md)에 정리했습니다.

Apache-2.0 독립 추론 런타임은 [GatedLinearAttention2](GatedLinearAttention2)에 분리했습니다. 이 폴더는 `lit_gpt`, `fla`, NVIDIA GDN-2 Triton kernel을 import하지 않고, Hugging Face의 `gyung/Gated_Linear_Attention2` weights와 repo 내부 `tokenizer/`를 바로 로드합니다.

## 한 줄 요약

`gdn2_kla_1.3B`는 GDN-2의 channel-wise erase/write gate에 Kaczmarz식 key-norm-normalized update step을 접어 넣은 recurrent-only linear attention 모델입니다. 기본 학습 길이는 GDN-2 논문 recipe와 맞춘 4K tokens입니다.

하이브리드 attention은 쓰지 않습니다.

## 왜 이 방식인가

Qwen3.5/Qwen3-Next와 Kimi Linear류는 GDN/KDA와 attention을 섞는 하이브리드 방향입니다. 이번 목표는 그쪽과 다르게, attention 없이 **recurrent linear attention의 memory update 자체**를 개선하는 것입니다.

GDN-2는 fixed-size recurrent memory에서 무엇을 지울지와 무엇을 쓸지를 분리합니다. 하지만 update의 세기는 여전히 key의 스케일에 영향을 받을 수 있습니다.

Kaczmarz step은 update 크기를 다음처럼 정규화합니다.

```math
\lambda_t = \frac{\eta_t}{\|k_t\|_2^2 + \epsilon}
```

그래서 후보 recurrence는 다음 형태가 됩니다.

```math
S_t
=
\left(I - k_t (\lambda_t b_t \odot k_t)^\top \right)D_tS_{t-1}
+
k_t(\lambda_t w_t \odot v_t)^\top
```

긴 문맥에서는 수많은 recurrent update가 누적됩니다. update가 너무 크면 기존 memory를 과하게 지우고, 너무 작으면 중요한 새 정보를 제대로 쓰지 못합니다. 이 후보는 GDN-2의 세밀한 erase/write 선택에 update-size 정규화를 더해서 long-context memory interference를 줄이는 것을 목표로 합니다.

자세한 수식과 설계 근거는 [docs/gdn2_kla_single_candidate.md](docs/gdn2_kla_single_candidate.md)에 정리했습니다. "이게 정확히 attention인지", "KV cache가 늘어나는지", "10T token stream이 어떤 의미에서 가능한지"는 [docs/what_exactly_is_gdn2_kla.md](docs/what_exactly_is_gdn2_kla.md)에 더 자세히 정리했습니다.

GDN-2 논문은 pretraining 기본 길이를 4K로 두고, RULER Table 3에서는 일부 synthetic retrieval task를 8K까지 평가합니다. 따라서 첫 실험은 10B tokens만 쓰되 sequence length는 4K로 맞춰야 GDN-2 100B recipe와 가장 공정하게 비교됩니다. 32K/128K/1M 확장은 후속 long-context 단계로 분리합니다.

## 실행 전 점검

```bash
scripts/check_pretrain_readiness.py \
  --train-config tsz128x4k_10B \
  --model-name gdn2_kla_1.3B \
  --devices 8 \
  --throughput 250000
```

현재 확인된 환경 기준:

- GPU: NVIDIA H200 8장
- 데이터: FineWeb-Edu sample/100BT parquet 140개
- `flash_attn`: 필요 없음
- `torchdata`: dataloader state resume에 사용

장기 학습 전에는 `torchdata` 설치를 권장합니다. 이번 공정 비교 실행은 worker 복제에 의한 샘플 중복 가능성을 피하려고 `TRAIN_NUM_WORKERS=0`을 기본값으로 둡니다.

```bash
pip install --pre --no-cache-dir torchdata --index-url https://download.pytorch.org/whl/nightly
```

## 학습 실행

기본 10B token / 4K context 실험:

```bash
./scripts/pretrain_gdn2_kla_10bt.sh
```

H200 8장 기준 기본값은 `MICRO_BATCH_SIZE=16`, `ACTIVATION_CHECKPOINTING=on`, `TRAIN_NUM_WORKERS=0`입니다. checkpointing 없이 micro batch 16은 첫 forward에서 OOM이 났으므로 켜는 것이 기본입니다. 그래도 메모리가 부족하면 `MICRO_BATCH_SIZE=8`로 낮추면 `gradient_accumulation_steps=2`가 되어 global batch token은 그대로 유지됩니다.

W&B를 켜려면:

```bash
WANDB_MODE=online ./scripts/pretrain_gdn2_kla_10bt.sh
```

출력 디렉터리:

```text
runs/outputs/tsz128x4k_10B_gdn2_kla_1.3B_fineweb_edu_10bt
```

같은 출력 디렉터리가 있으면 resume을 시도합니다.

허깅페이스 업로드 기본 repo 이름은 `Gated_Linear_Attention2`입니다. `.env`의 `HF_TOKEN` owner를 붙여 `<owner>/Gated_Linear_Attention2`로 1B token마다 model-only checkpoint를 업로드합니다.

## 100BT에서 10B를 고르는 방식

현재 기본 코드는 10B token subset을 미리 저장하지 않습니다. FineWeb-Edu `sample/100BT` parquet 전체를 streaming으로 열고, 결정적 shuffle을 적용한 뒤 학습 token budget이 10B에 도달하면 멈춥니다.

현재 흐름:

```text
FineWeb-Edu sample/100BT parquet files
-> HuggingFace streaming dataset
-> deterministic streaming shuffle(seed=3407, buffer=100000)
-> split_dataset_by_node(rank, world_size)
-> tokenizer
-> fixed-length token chunks
-> local chunk-buffer shuffle
-> stop at 10B trained tokens
```

따라서 이것은 **100BT 전체에서 uniform random 10B를 미리 뽑는 방식은 아닙니다.** 하지만 sorted streaming prefix보다는 파일/shard 순서 편향이 작고, 같은 seed와 코드 경로로 baseline/candidate를 재현할 수 있습니다.

권장 기준은 다음과 같습니다.

- smoke/debug: streaming-prefix 방식으로 충분합니다.
- 현재 공정 실행: deterministic streaming shuffle을 쓰고, dataloader worker는 0으로 둬서 worker 복제에 의한 샘플 중복 가능성을 제거합니다.
- publication-grade ablation: deterministic random seed로 shard/document 순서를 섞은 manifest를 만들고 그 순서로 10B를 학습하는 것이 더 낫습니다.
- 논문식 learning-curve 비교: 공식 100B recipe의 정확한 데이터 순서를 알고 있다면 같은 순서로 10B에서 멈추는 것도 의미가 있습니다.

현재 우리는 공식 GDN-2의 정확한 dataloader seed/order까지 재현하는 것이 아니라, 같은 FineWeb-Edu 100BT source와 같은 4K recipe에서 token budget만 10B로 줄이는 실험입니다. 결과를 강하게 주장하려면 random manifest 방식으로 고정하는 편이 더 안전합니다.

## 예상 완료 시간

10B token, 4K sequence length 기준입니다. GDN-2 논문 기본 pretraining length와 맞춘 비교 설정입니다.

| Total tokens/s | 예상 시간 |
|---:|---:|
| 100K | 27.8 hours |
| 200K | 13.9 hours |
| 250K | 11.1 hours |
| 300K | 9.3 hours |
| 500K | 5.6 hours |

하루 안에 끝내려면 전체 throughput이 약 116K tokens/s 이상이면 됩니다. H200 8장 기준 250K tokens/s가 나오면 약 11.1시간입니다.

## 성공 기준

10B token만으로 100B token GDN-2의 모든 benchmark를 이긴다고 주장하면 안 됩니다. 먼저 다음을 봅니다.

- plain `gdn2_1.3B` 대비 validation loss
- RULER S-NIAH / MK-NIAH
- number-range tracking
- entity-state update probe
- 학습 안정성, NaN 여부, throughput 저하

이 후보가 유효하려면 long-context retrieval/state-tracking에서 개선이 있어야 하고, loss와 throughput 손상이 작아야 합니다.

## 후속 Long-Context 확장

1차 실험은 4K/10B로 끝냅니다. 그 뒤에 long-context 능력을 보려면 같은 checkpoint에서 32K, 128K, streaming 1M 순서로 확장합니다.

4K와 32K는 둘 다 global batch token을 524,288로 맞출 수 있습니다.

| 설정 | Sequence length | Sequence batch | Micro batch/GPU | Grad accum | Global batch tokens |
|---|---:|---:|---:|---:|---:|
| 4K | 4,096 | 128 | 4 | 4 | 524,288 |
| 32K | 32,768 | 16 | 1 | 2 | 524,288 |

`pretrain.py`는 `TRAIN_CONFIG`에서 sequence length를 파싱해 model `block_size`에 넣고, streaming tokenizer도 `block_size + 1` 단위로 샘플을 만듭니다. 32K 이상에서는 activation checkpointing이 자동으로 켜집니다.

32K continuation 예시:

```bash
TRAIN_CONFIG=tsz16x32k_10B \
EXP_NAME=gdn2_kla_1.3B_fineweb_edu_10bt_32k_cont \
MICRO_BATCH_SIZE=1 \
./scripts/pretrain_gdn2_kla_10bt.sh
```

## RoPE와 128K/1M

이 후보는 pure GDN-2 recurrent model이라 RoPE로 context를 늘리는 방식이 아닙니다. `gdn2_kla_1.3B`에서는 `nope=True`로 RoPE cache도 만들지 않습니다.

128K나 1M으로 늘리는 핵심은 RoPE scaling이 아니라 다음 제약입니다.

- GDN recurrent state는 context length와 무관하게 고정 크기입니다.
- prefill/training compute는 여전히 token 수에 비례합니다.
- 현재 학습 경로는 full-sequence logits를 만들기 때문에 매우 긴 길이에서는 logits memory가 병목입니다.
- 128K는 H200에서 smoke run 후 판단할 수 있지만, 1M 학습은 현재 코드 그대로는 현실적이지 않습니다.
- 1M급 학습/서빙을 하려면 streaming recurrent cache wiring, chunked logits/loss, 긴 시퀀스 dataloader, 별도 evaluation harness가 필요합니다.
