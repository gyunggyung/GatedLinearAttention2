# Gated DeltaNet-2 순수 GDN Pre-training 실행 가이드

기존 공식 README는 [README.original.md](README.original.md)에 보존했습니다.

이 워크스페이스는 논문 [Gated DeltaNet-2](https://arxiv.org/html/2605.22791v1)의 recurrent-only `Gated DeltaNet-2` pre-training 방법론에 맞춰, `HuggingFaceFW/fineweb-edu`의 [`sample/100BT`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/100BT)를 바로 학습할 수 있게 정리한 버전입니다.

## 논문 방법론과 일치하는 부분

- 모델: `gdn2_1.3B`, 어텐션/SWA 없는 순수 recurrent GDN-2
- 데이터: FineWeb-Edu 100BT sample, `data/fineweb-edu/data/sample/100BT/*.parquet`
- 학습 토큰: 100B tokens
- 기본 sequence length: 4K tokens
- global batch: 524,288 tokens, 논문 표현으로 0.5M tokens
- warmup: 1B tokens
- optimizer: AdamW, LR `4e-4`, weight decay `0.1`, grad clip `1.0`, cosine decay

중요: 이 코드는 논문 pre-training recipe에 맞춰 학습을 시작하게 만든 것입니다. 논문 표의 최종 점수를 그대로 보장하려면 checkpoint 후 동일한 evaluation harness, 정확한 tokenizer/데이터 순서, 커널/라이브러리 버전, seed와 분산 실행 조건까지 맞아야 합니다.

## 현재 준비 상태

`scripts/check_pretrain_readiness.py` 기준 현재 환경은 다음 상태입니다.

- GPU: NVIDIA H200 8장, 각 139.8 GiB
- 데이터: Parquet 140개, 286.39 GB decimal
- 필수 패키지: `torch`, `lightning_fabric`, `pytorch_lightning`, `datasets`, `transformers`, `fla`, `wandb` 확인됨
- 빠진 패키지: `torchdata`
- `flash_attn`: 없음. 순수 `gdn2_1.3B`에는 필요 없고, hybrid/SWA 모델에만 필요합니다.

현재 상태는 “학습 시작 가능, 장기 학습 전 `torchdata` 설치 권장”입니다. `torchdata`가 없으면 코드가 `num_workers=0`으로 강제되어 샘플 중복은 피하지만, dataloader state resume이 빠집니다. 100B 장기 학습에서는 설치하는 쪽이 맞습니다.

```bash
pip install --pre --no-cache-dir torchdata --index-url https://download.pytorch.org/whl/nightly
```

준비 상태와 ETA는 언제든 다시 확인할 수 있습니다.

```bash
scripts/check_pretrain_readiness.py
```

## H200 8장 예상 완료 시간

기본 4K 설정의 schedule은 다음과 같습니다.

- `train_config`: `tsz128x4k_100B`
- micro batch per GPU: `4`
- gradient accumulation: `4`
- effective global batch: `524,288` tokens
- optimizer steps: `190,735`
- micro iterations: `762,940`
- stop tokens: `100,000,071,680`

실제 throughput은 첫 Triton compile 이후 100-200 iteration 정도를 보고 확정해야 합니다. 현재 문서의 ETA는 H200 8장 전체 throughput 가정별 계산입니다.

| Total tokens/s | 100B 예상 시간 |
|---:|---:|
| 100K | 11.57 days |
| 200K | 5.79 days |
| 250K | 4.63 days |
| 300K | 3.86 days |
| 500K | 2.31 days |

스크립트 기본 ETA 가정은 `250K tokens/s`입니다. 실제 로그의 `total training throughput` 값으로 `EXPECTED_TOKENS_PER_SEC`를 조정하면 됩니다.

## 기본 4K / 100B 학습

```bash
./scripts/pretrain_fineweb_edu_100bt_gdn2.sh
```

출력은 기본적으로 `runs/outputs/tsz128x4k_100B_gdn2_1.3B_fineweb_edu_100bt` 아래에 저장됩니다. 같은 출력 디렉터리가 있으면 resume을 시도합니다.

W&B를 켜려면:

```bash
WANDB_MODE=online ./scripts/pretrain_fineweb_edu_100bt_gdn2.sh
```

검증 Parquet가 있으면:

```bash
VALIDATION_DATA=/path/to/val_parquet_dir \
  ./scripts/pretrain_fineweb_edu_100bt_gdn2.sh
```

## 32K 이상 장거리 학습

논문 pre-training 기본값은 4K입니다. 아래 설정은 논문과 같은 recurrent-only GDN-2, 같은 FineWeb-Edu 100B, 같은 0.5M global batch를 유지하면서 sequence length만 늘리는 장거리 확장 실험입니다.

| Config | GPUs | Micro batch/GPU | Grad accum | Effective batch | 상태 |
|---|---:|---:|---:|---:|---|
| `tsz128x4k_100B` | 8 | 4 | 4 | 524,288 | 논문 pretrain 기본 |
| `tsz16x32k_100B` | 8 | 1 | 2 | 524,288 | 장거리 학습 권장 시작점 |
| `tsz8x64k_100B` | 8 | 1 | 1 | 524,288 | H200 8장 실험 가능 범위 |
| `tsz4x128k_100B` | 4 | 1 | 1 | 524,288 | 4 GPU로 제한해야 batch 동일 |
| `tsz4x128k_100B` | 8 | 1 | 1 | 1,048,576 | batch가 논문과 달라짐 |

32K:

```bash
TRAIN_CONFIG=tsz16x32k_100B MICRO_BATCH_SIZE=1 \
  ./scripts/pretrain_fineweb_edu_100bt_gdn2.sh
```

64K:

```bash
TRAIN_CONFIG=tsz8x64k_100B MICRO_BATCH_SIZE=1 \
  ./scripts/pretrain_fineweb_edu_100bt_gdn2.sh
```

128K에서 0.5M batch를 유지하려면 4 GPU만 사용합니다.

```bash
DEVICES=4 TRAIN_CONFIG=tsz4x128k_100B MICRO_BATCH_SIZE=1 \
  ./scripts/pretrain_fineweb_edu_100bt_gdn2.sh
```

실제 최대 길이는 HBM, Triton kernel workspace, full-vocab logits 메모리에 의해 결정됩니다. 현재 코드 기준으로 H200 8장에서는 64K를 실험 가능한 상한으로 보고 시작하는 것이 현실적이고, 128K는 4 GPU 또는 batch 변경 조건에서 smoke run으로 먼저 확인해야 합니다. 256K 이상은 현재 full logits path 때문에 권장하지 않습니다.

## 이번에 수정한 내용

- `pretrain.py`
  - `lightning` 메타 패키지가 없어도 `lightning_fabric`으로 동작하게 fallback을 추가했습니다.
  - 기본 모델을 `gdn2_1.3B`로 변경했습니다.
  - FineWeb-Edu raw Parquet streaming/tokenization을 기본 경로로 설정했습니다.
  - `tsz128x4k_100B`, `tsz16x32k_100B`, `tsz8x64k_100B`처럼 train config에서 sequence length를 읽어 model `block_size`에 반영합니다.
  - 100B 종료가 optimizer step 경계에서 일어나도록 token schedule을 정리했습니다.
  - 32K 이상에서는 activation checkpointing이 자동으로 켜집니다.
  - H200 8장 기준 ETA를 로그로 출력할 수 있게 했습니다.

- `data.py`
  - `sample/100BT/*.parquet` 구조를 recursive로 읽게 했습니다.
  - `torchdata`가 없으면 `num_workers=0`으로 강제해 IterableDataset 중복 샘플링을 막습니다.

- `lit_gpt/model.py`
  - 순수 GDN 모델은 FlashAttention 없이 import됩니다.
  - activation checkpointing 옵션을 추가했습니다.

- `lit_gpt/speed_monitor.py`, `lit_gpt/utils.py`
  - `lightning_fabric` fallback을 추가했습니다.

- `scripts/pretrain_fineweb_edu_100bt_gdn2.sh`
  - 이 환경에서 바로 실행하는 순수 GDN-2 100BT 학습 스크립트입니다.
  - `TRAIN_CONFIG`, `MICRO_BATCH_SIZE`, `DEVICES`, `EXPECTED_TOKENS_PER_SEC`를 환경변수로 바꿀 수 있습니다.

- `scripts/check_pretrain_readiness.py`
  - GPU, 데이터, dependency, token schedule, ETA를 학습 시작 전 점검합니다.

- `scripts/tsz1024x4k_100B_swa_gdn2.sh`
  - 기존 Slurm/SWA 스크립트를 새 순수 GDN 실행 스크립트로 넘기는 wrapper로 바꿨습니다.

- `.gitignore`
  - `.hf_cache/`, `data/`, `runs/`를 git 추적에서 제외했습니다.
