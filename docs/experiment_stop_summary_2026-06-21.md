# 2026-06-21 실험 중단 요약

이 문서는 `gdn2_kla_1.3B` / `GatedLinearAttention2` 10B token 실험을 중단하면서 남기는 최종 요약이다.

결론부터 쓰면, 이번 실험은 **성능 목표 기준으로 실패**다.

의미 있는 전체 승리는 나오지 않았다. 일부 단일 task에서 아주 작은 승리나 동률에 가까운 값은 있었지만, GDN2, Mamba-2, Gated DeltaNet, KDA, Mamba-3 계열을 모델 전체로 이겼다고 볼 수 없다.

## 실행 중단 상태

2026-06-21 12:57 KST 기준으로 실행 중이던 평가 프로세스를 모두 종료했다.

최종 확인:

```text
GPU 0-7: 평가 프로세스 없음
memory.used: 1 MiB 수준
```

중단한 작업:

- 10B RULER 잔여 `niah_single_1`
- 01B-07B RULER learning-curve 평가
- 추가 real-world / final checkpoint 평가

## 실험 목표

이번 후보의 목표는 다음이었다.

```text
GDN2의 gate 방식과 recurrent linear attention의 효율성을 결합하고,
Kaczmarz식 key-norm-normalized update를 추가해서
GDN2 100B baseline보다 적은 10B token 학습으로도 강한 결과가 나오는지 확인한다.
```

구조적으로는 다음을 노렸다.

- linear attention의 고정 크기 recurrent state
- GDN2의 key-side erase gate `b_t`
- GDN2의 value-side write gate `w_t`
- context length에 비례해 커지는 KV cache 회피
- Kaczmarz식 update 크기 정규화
- long-context state tracking / retrieval 개선 가능성

하지만 결과적으로 이 목표는 달성하지 못했다.

## 우리가 만든 구조가 정확히 무엇인지

이 모델은 Transformer softmax attention이 아니다.

또한 Transformer + recurrent linear attention hybrid도 아니다.

이번 모델은 다음에 가깝다.

```text
pure recurrent gated linear attention
+ GDN2 erase/write gates
+ Kaczmarz-style update normalization
```

즉, 사용자가 가져온 linear attention 설명처럼 softmax attention의 `O(T^2)` pairwise attention matrix를 만들지 않고, recurrent state를 업데이트하는 방식이다.

단순 linear attention과 다른 점은 gate다.

| 구성 | 단순 linear attention | GDN2 / 우리 후보 |
|---|---|---|
| 과거 정보 저장 | state에 계속 누적 | state에 누적 |
| 지우기 | 약하거나 없음 | erase gate로 제어 |
| 쓰기 | 새 value를 그대로 더함 | write gate로 제어 |
| update 크기 보정 | 보통 없음 | Kaczmarz step 추가 |
| KV cache | token별 KV 저장 불필요 | token별 KV 저장 불필요 |
| softmax attention | 없음 | 없음 |

따라서 “linear attention에 GDN2 스타일 gate를 붙였느냐”는 질문에는 **맞다**고 답할 수 있다.

다만 “Transformer의 장점까지 가져왔느냐”는 질문에는 **아니다**라고 답해야 한다. 사용자가 하이브리드는 하지 말자고 했기 때문에 softmax attention block을 넣지 않았다.

## 표준 벤치 결과

완료된 Table 2 계열 표준 평가 결과는 다음과 같다. Social IQA는 local `datasets` 호환성 문제로 제외했다.

| Checkpoint | Avg 8 acc | Wiki PPL | LAMBADA acc | HellaSwag | ARC-C | BoolQ |
|---|---:|---:|---:|---:|---:|---:|
| 01B | 39.88 | 45.14 | 20.67 | 31.96 | 21.84 | 62.11 |
| 02B | 42.00 | 35.86 | 26.02 | 35.80 | 23.29 | 61.71 |
| 03B | 42.60 | 33.12 | 27.69 | 37.11 | 24.06 | 60.00 |
| 04B | 43.22 | 31.51 | 30.10 | 37.81 | 24.23 | 60.21 |
| 05B | 43.68 | 29.96 | 28.76 | 39.67 | 25.00 | 60.95 |
| 06B | 44.77 | 28.31 | 32.93 | 40.04 | 26.45 | 60.83 |
| 07B | 44.30 | 26.81 | 31.17 | 40.77 | 25.60 | 58.69 |
| 08B | 45.00 | 25.86 | 33.98 | 41.19 | 25.60 | 59.20 |
| 09B | 45.09 | 25.16 | 34.81 | 41.59 | 24.83 | 59.27 |
| 10B | 45.20 | 24.52 | 34.76 | 41.46 | 26.11 | 57.80 |
| GDN2 recurrent | 53.11 | 15.90 | 48.09 | 56.84 | 38.23 | 59.54 |

좋아진 점:

- 01B에서 10B까지 평균 정확도는 `39.88 -> 45.20`으로 상승했다.
- WikiText perplexity는 `45.14 -> 24.52`로 개선됐다.
- LAMBADA accuracy는 `20.67 -> 34.76`으로 개선됐다.
- HellaSwag는 `31.96 -> 41.46`으로 개선됐다.

문제점:

- 10B까지 학습해도 GDN2 recurrent와 평균 정확도 차이가 `53.11 - 45.20 = 7.91pt` 남았다.
- LAMBADA, HellaSwag, ARC-Challenge 격차가 크다.
- BoolQ는 중간 checkpoint에서 높았지만 최종 10B에서 `57.80`으로 떨어졌다.
- 표준 벤치 전체에서는 어떤 강한 승리도 없다.

## 이긴 항목

최종 10B 기준으로 이긴 항목은 매우 제한적이다.

### Table 2 표준 평가

| 비교 대상 | 이긴 항목 |
|---|---|
| GDN2 recurrent | 없음 |
| GDN2 hybrid | 없음 |
| Mamba-2 recurrent | 없음 |
| Mamba-2 hybrid | 없음 |
| Gated DeltaNet recurrent | 없음 |
| Gated DeltaNet hybrid | 없음 |
| KDA recurrent | 없음 |
| KDA hybrid | 없음 |
| Mamba-3 recurrent SISO | BoolQ: 57.80 vs 55.90 |
| Mamba-3 recurrent MIMO | BoolQ: 57.80 vs 57.74 |
| Mamba-3 hybrid | 없음 |

중간 checkpoint까지 포함하면 09B의 BoolQ가 일부 모델을 이겼다.

| Checkpoint | 비교 대상 | 이긴 항목 |
|---|---|---|
| 09B | Gated DeltaNet recurrent | BoolQ: 59.27 vs 58.78 |
| 09B | Mamba-3 recurrent SISO | BoolQ: 59.27 vs 55.90 |
| 09B | Mamba-3 recurrent MIMO | BoolQ: 59.27 vs 57.74 |
| 09B | Mamba-3 hybrid SISO | BoolQ: 59.27 vs 57.86 |
| 09B | Mamba-3 hybrid MIMO | BoolQ: 59.27 vs 57.98 |

하지만 이건 전체 모델 승리가 아니다. BoolQ 하나의 부분 승리다.

## RULER 결과

10B RULER는 4개 task 중 3개가 완료된 상태에서 중단했다. `niah_single_1`은 아직 완료되지 않았다.

완료된 결과:

| Task | 1K | 2K | 4K | 8K |
|---|---:|---:|---:|---:|
| `niah_single_2` | 92.8 | 75.8 | 25.8 | 4.8 |
| `niah_single_3` | 24.4 | 12.0 | 2.2 | 0.8 |
| `niah_multikey_1` | 21.2 | 18.2 | 20.2 | 9.6 |

RULER에서 GDN2 recurrent 대비:

| 항목 | 우리 10B | GDN2 recurrent | 결과 |
|---|---:|---:|---|
| `niah_single_2 1K` | 92.8 | 100.0 | 패배 |
| `niah_single_2 2K` | 75.8 | 100.0 | 패배 |
| `niah_single_2 4K` | 25.8 | 93.0 | 크게 패배 |
| `niah_single_2 8K` | 4.8 | 39.2 | 패배 |
| `niah_single_3 1K` | 24.4 | 92.0 | 크게 패배 |
| `niah_single_3 2K` | 12.0 | 89.8 | 크게 패배 |
| `niah_single_3 4K` | 2.2 | 25.8 | 패배 |
| `niah_multikey_1 1K` | 21.2 | 72.6 | 크게 패배 |
| `niah_multikey_1 2K` | 18.2 | 46.4 | 패배 |
| `niah_multikey_1 4K` | 20.2 | 28.4 | 패배 |

부분적으로 이긴 RULER 항목:

| 비교 대상 | 이긴 항목 |
|---|---|
| Mamba-3 recurrent MIMO | `niah_multikey_1 4K`: 20.2 vs 18.0 |
| Transformer hybrid | `niah_single_2 8K`: 4.8 vs 0.0 |
| Mamba-3 recurrent SISO | `niah_multikey_1 4K`: 20.2 vs 20.2, 사실상 동률 |

이것도 의미 있는 전체 승리로 보기 어렵다. RULER의 핵심 task들에서 대부분 크게 졌다.

## Real-World Retrieval 결과

완료된 10B real-world 결과는 다음과 같다.

| Task | 우리 10B |
|---|---:|
| SWDE contains | 19.98 |
| SQuAD completion contains | 31.50 |
| FDA contains | 5.35 |
| TriviaQA exact match | 1.71 |
| NQ exact match | 1.02 |
| DROP EM | 0.13 |
| DROP F1 | 2.89 |

GDN2 recurrent 논문값과 비교하면 전반적으로 낮다. 특히 TriviaQA, NQ, DROP이 매우 낮다.

## 잘한 점

이번 실험에서 얻은 긍정적인 점은 다음이다.

1. 학습 파이프라인은 10B token까지 완주했다.
2. 1B 단위 checkpoint 업로드/저장 구조를 만들었다.
3. 표준 벤치 학습 곡선을 01B부터 10B까지 확보했다.
4. RULER 직접 평가 스크립트를 만들었다.
5. RULER 중간 저장 기능을 추가했다.
6. 독립 런타임 `GatedLinearAttention2/`를 분리했다.
7. 모델 구조 설명, 학습 runbook, 평가 계획 문서를 만들었다.
8. 01B에서 10B로 갈수록 loss/표준 성능이 대체로 개선되는 것을 확인했다.

즉, 실험 인프라와 재현 가능한 평가 체계는 남았다.

## 못한 점

성능 관점에서는 다음이 실패다.

1. GDN2를 어떤 핵심 benchmark에서도 의미 있게 이기지 못했다.
2. Mamba-2, Gated DeltaNet, KDA도 대부분 이기지 못했다.
3. Mamba-3 대비 승리는 BoolQ 또는 RULER 일부 단일 항목에 그쳤다.
4. RULER long-context 결과가 기대보다 훨씬 낮았다.
5. Real-world retrieval 결과가 매우 낮았다.
6. 10B token으로는 100B token baseline 격차를 메우지 못했다.
7. Kaczmarz update normalization이 실제 성능 개선으로 이어졌다는 증거가 없다.

## 실패 원인 추정

### 1. 10B token이 너무 적다

가장 단순한 이유다. 비교 대상은 GDN2 논문의 100B token baseline이다. 10B는 그 10분의 1이다.

학습 곡선을 보면 좋아지고는 있다.

```text
Avg 8 acc: 39.88 -> 45.20
Wiki PPL: 45.14 -> 24.52
LAMBADA acc: 20.67 -> 34.76
```

하지만 이 속도로는 20B를 더 학습해도 GDN2를 바로 이길 가능성은 낮다. 개선은 이어질 수 있지만, 격차가 너무 크다.

### 2. Hybrid가 아니라 pure recurrent linear attention이다

이번 후보는 Transformer softmax attention을 쓰지 않았다. 따라서 Transformer의 정밀한 token-to-token 검색 능력이 없다.

이 선택은 의도적이었다. 차별점을 위해 하이브리드를 하지 않고 단일 recurrent linear attention 후보만 테스트했다.

하지만 결과적으로 표준 language modeling, reasoning, retrieval task에서는 이 손실이 크게 나타난 것으로 보인다.

### 3. GDN2 gate만 가져온다고 GDN2 성능이 자동으로 나오지 않는다

GDN2의 장점은 gate 하나만이 아니다.

성능에는 다음이 모두 영향을 준다.

- 정확한 recurrence 설계
- gate parameterization
- normalization
- decay dynamics
- initialization
- optimizer recipe
- data ordering
- scale
- kernel 구현의 numerical behavior

우리는 GDN2의 erase/write gate를 사용했지만, Kaczmarz step을 추가하면서 update dynamics가 달라졌다. 이 변화가 오히려 일부 task에서는 memory를 약하게 쓰거나 잘못 지우게 만들었을 가능성이 있다.

### 4. Kaczmarz step이 과도하게 보수적인 update를 만들었을 수 있다

Kaczmarz식 step은 key norm으로 update 크기를 조절한다.

의도는 안정성이다.

```text
key가 크면 update를 줄이고,
key가 작으면 update를 상대적으로 키운다.
```

하지만 long-context retrieval에서는 중요한 정보를 강하게 써야 하는 순간이 있다. update normalization이 너무 보수적으로 작동하면 needle 정보를 state에 충분히 쓰지 못할 수 있다.

RULER `niah_single_3`가 매우 낮은 것은 이 가능성을 의심하게 한다.

### 5. 4K 학습에서 8K RULER 일반화가 약했다

학습은 4K context로 했다. RULER는 8K까지 평가했다.

GDN2는 recurrent 구조라 이론상 길이 확장이 가능하지만, 실제 성능은 학습 분포와 state capacity에 의존한다.

이번 결과에서는 8K가 특히 약했다.

```text
niah_single_2 8K: 4.8
niah_single_3 8K: 0.8
niah_multikey_1 8K: 9.6
```

즉, “이론상 recurrent라서 긴 길이에 강하다”는 말은 이번 checkpoint에서는 증명되지 않았다.

### 6. 모델이 아직 일반 언어 능력 자체가 부족하다

RULER도 결국 prompt를 읽고 답을 생성해야 한다. 기본 언어 모델링 능력이 낮으면 retrieval 구조가 있어도 답 형식, 지시 이해, key extraction에서 실패한다.

표준 벤치 격차가 큰 상태라 RULER도 약해졌을 가능성이 크다.

## 20B를 더 학습하면 달라질 가능성

20B까지 늘리면 더 좋아질 가능성은 있다.

하지만 지금 추세로 보면 **비슷한 결론이 나올 가능성이 높다.**

단순 log-fit 예측은 대략 다음이었다.

| 지표 | 20B 예측 |
|---|---:|
| Avg 8 acc | 약 47.0 |
| Wiki PPL | 약 18.0 |
| LAMBADA acc | 약 38.9 |
| HellaSwag | 약 45.0 |
| ARC-Challenge | 약 27.3 |
| BoolQ | 약 57.8 |

이는 GDN2 recurrent avg `53.11`, LAMBADA acc `48.09`, HellaSwag `56.84`, ARC-Challenge `38.23`와 여전히 큰 차이가 있다.

따라서 20B 추가 학습은 연구적으로는 확인할 수 있지만, GPU 비용 대비 성공 가능성이 낮다.

## 최종 판단

이번 실험은 다음 결론으로 중단한다.

```text
GatedLinearAttention2 / gdn2_kla_1.3B 10B-token 실험은
학습과 평가 인프라는 구축했지만,
성능 목표는 달성하지 못했다.
```

실패 판단의 핵심 근거:

- GDN2를 표준 벤치에서 이긴 항목 없음
- GDN2를 RULER에서 이긴 항목 없음
- Mamba-2, Gated DeltaNet, KDA 대비 의미 있는 승리 없음
- Mamba-3 대비 일부 단일 항목 승리만 존재
- long-context 장점이 RULER에서 확인되지 않음
- real-world retrieval도 약함

후속으로 같은 방향을 계속하려면 단순 20B 추가 학습보다 구조 ablation이 먼저다.

우선순위는 다음이다.

1. Kaczmarz step 제거 ablation
2. Kaczmarz strength / clipping 범위 ablation
3. GDN2 원본 recurrence와 동일 recipe 재현
4. 4K가 아니라 8K/16K continuation으로 RULER 확인
5. hybrid를 허용할지 재검토

현재 상태에서는 GPU를 계속 쓰는 것이 타당하지 않다.
