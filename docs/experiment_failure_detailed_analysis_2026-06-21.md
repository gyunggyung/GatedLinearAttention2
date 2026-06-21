# GatedLinearAttention2 실험 실패 상세 분석

이 문서는 `GatedLinearAttention2` / `gdn2_kla_1.3B` 10B token 실험의 상세 사후 분석이다.

목표는 다음 네 가지를 명확히 남기는 것이다.

1. 무엇을 만들었는가
2. 무엇을 평가했는가
3. 왜 실패했다고 판단하는가
4. 다음에 한다면 어떤 순서로 해야 하는가

## 1. 실험 결론

이번 실험은 성능 목표 기준으로 실패다.

핵심 이유:

- GDN2 recurrent를 표준 benchmark에서 이기지 못했다.
- GDN2 recurrent를 RULER에서도 이기지 못했다.
- Mamba-2, Gated DeltaNet, KDA 대비 의미 있는 승리가 없다.
- Mamba-3 대비로는 일부 단일 항목 승리만 있다.
- long-context에서 유리하다는 가설을 RULER로 증명하지 못했다.

이번 후보가 완전히 무의미하다는 뜻은 아니다. 학습은 진행됐고, 01B에서 10B로 갈수록 점수는 개선됐다. 그러나 “GDN2보다 유망한 단일 recurrent linear attention 후보”라는 강한 주장은 현재 숫자로 지지되지 않는다.

## 2. 모델 구조

모델 이름:

```text
gdn2_kla_1.3B
```

기본 구조:

```text
pure recurrent gated linear attention
```

사용한 핵심 요소:

| 요소 | 사용 여부 | 설명 |
|---|---|---|
| Q/K/V projection | 사용 | query, key, value 기반 recurrent linear attention |
| Softmax attention | 사용 안 함 | Transformer식 full attention 없음 |
| KV cache | token별 KV cache 없음 | fixed recurrent state 사용 |
| GDN2 erase gate | 사용 | key 방향에서 지울 정보 제어 |
| GDN2 write gate | 사용 | value 방향에서 쓸 정보 제어 |
| Kaczmarz step | 추가 | key norm 기반 update 크기 보정 |
| Hybrid attention | 사용 안 함 | 실험 차별점을 위해 배제 |

중요한 점:

```text
이 모델은 Transformer의 장점과 GDN2의 장점을 모두 합친 hybrid가 아니다.
```

사용자가 하이브리드는 하지 말자고 했기 때문에, softmax attention block은 없다. 따라서 Transformer의 강한 token-to-token retrieval 능력은 들어 있지 않다.

## 3. 수식적 의도

단순 linear attention은 과거 정보를 state에 누적한다.

```math
S_t = S_{t-1} + k_t v_t^\top
```

이 방식은 효율적이지만 지우기 어렵다. GDN2는 erase/write gate를 써서 memory update를 더 세밀하게 만든다.

우리 후보는 여기에 Kaczmarz식 step을 넣었다.

```math
\lambda_t = \frac{\eta_t}{\|k_t\|_2^2 + \epsilon}
```

의도:

- key norm이 클 때 update가 너무 커지는 것을 방지한다.
- key norm이 작을 때 update가 너무 약해지는 것을 줄인다.
- 긴 sequence에서 update 폭주나 memory interference를 줄인다.

그러나 실제 결과를 보면 이 보정이 성능 개선으로 이어졌다는 증거가 없다.

가능한 부작용:

- 중요한 정보를 써야 할 때 update가 너무 약해졌을 수 있다.
- erase gate와 write gate가 이미 조절하는데, Kaczmarz step이 한 번 더 눌러서 memory write가 보수적으로 변했을 수 있다.
- RULER needle retrieval에서 key/value를 충분히 state에 남기지 못했을 수 있다.

## 4. 학습 설정

| 항목 | 값 |
|---|---|
| 학습 token | 10B |
| context length | 4K |
| 데이터 | FineWeb-Edu sample/100BT |
| GPU | H200 8장 |
| checkpoint | 1B 단위 저장 |
| tokenizer | TinyLlama/TinyLlama_v1.1 |
| 비교 대상 | GDN2 논문 Table 2, Table 3, Table 4 |

비교상 불리한 점:

- 논문 baseline은 100B token 학습이다.
- 우리는 10B token만 학습했다.

비교상 의도한 점:

- GDN2 논문 recipe와 맞추기 위해 4K context로 먼저 학습했다.
- 10배 적은 token으로 얼마나 성능이 나오는지 보는 목적이었다.

결과적으로 10B는 너무 적었다.

## 5. 표준 벤치 결과 분석

표준 평가 learning curve:

| Checkpoint | Avg 8 acc | Wiki PPL | LAMBADA acc | PIQA | HellaSwag | ARC-C | BoolQ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 01B | 39.88 | 45.14 | 20.67 | 62.24 | 31.96 | 21.84 | 62.11 |
| 02B | 42.00 | 35.86 | 26.02 | 64.25 | 35.80 | 23.29 | 61.71 |
| 03B | 42.60 | 33.12 | 27.69 | 64.64 | 37.11 | 24.06 | 60.00 |
| 04B | 43.22 | 31.51 | 30.10 | 63.93 | 37.81 | 24.23 | 60.21 |
| 05B | 43.68 | 29.96 | 28.76 | 65.56 | 39.67 | 25.00 | 60.95 |
| 06B | 44.77 | 28.31 | 32.93 | 65.29 | 40.04 | 26.45 | 60.83 |
| 07B | 44.30 | 26.81 | 31.17 | 65.18 | 40.77 | 25.60 | 58.69 |
| 08B | 45.00 | 25.86 | 33.98 | 66.27 | 41.19 | 25.60 | 59.20 |
| 09B | 45.09 | 25.16 | 34.81 | 66.65 | 41.59 | 24.83 | 59.27 |
| 10B | 45.20 | 24.52 | 34.76 | 67.03 | 41.46 | 26.11 | 57.80 |
| GDN2 recurrent | 53.11 | 15.90 | 48.09 | 72.80 | 56.84 | 38.23 | 59.54 |

관찰:

- 모델은 분명히 학습된다.
- Perplexity는 안정적으로 내려간다.
- LAMBADA와 HellaSwag도 상승한다.
- 하지만 GDN2와의 격차가 너무 크다.
- BoolQ는 초반부터 높지만, 최종 10B에서 떨어진다.

해석:

```text
학습이 망가진 것은 아니다.
하지만 아키텍처/레시피가 baseline을 이길 만큼 강하지 않았다.
```

## 6. RULER 분석

완료된 10B RULER 일부 결과:

| Task | 1K | 2K | 4K | 8K |
|---|---:|---:|---:|---:|
| `niah_single_2` | 92.8 | 75.8 | 25.8 | 4.8 |
| `niah_single_3` | 24.4 | 12.0 | 2.2 | 0.8 |
| `niah_multikey_1` | 21.2 | 18.2 | 20.2 | 9.6 |

아직 완료되지 않은 항목:

```text
niah_single_1
```

중단 판단 때문에 남은 RULER는 더 진행하지 않았다.

### RULER에서 보이는 문제

`niah_single_2`는 1K에서 `92.8`로 어느 정도 된다. 그러나 4K와 8K에서 급격히 무너진다.

```text
1K: 92.8
2K: 75.8
4K: 25.8
8K: 4.8
```

`niah_single_3`는 더 심각하다.

```text
1K: 24.4
2K: 12.0
4K: 2.2
8K: 0.8
```

이것은 long-context retrieval/state tracking 장점이 제대로 나오지 않았다는 뜻이다.

### RULER에서 이긴 항목

부분 승리는 있었다.

| 비교 대상 | 이긴 항목 |
|---|---|
| Mamba-3 recurrent MIMO | `niah_multikey_1 4K`: 20.2 vs 18.0 |
| Transformer hybrid | `niah_single_2 8K`: 4.8 vs 0.0 |
| Mamba-3 recurrent SISO | `niah_multikey_1 4K`: 20.2 vs 20.2, 사실상 동률 |

하지만 이것은 전체 승리가 아니다.

GDN2, Mamba-2, Gated DeltaNet, KDA 대비로는 의미 있는 승리가 없다.

## 7. Real-World Retrieval 분석

완료된 10B 결과:

| Task | 점수 |
|---|---:|
| SWDE contains | 19.98 |
| SQuAD completion contains | 31.50 |
| FDA contains | 5.35 |
| TriviaQA exact match | 1.71 |
| NQ exact match | 1.02 |
| DROP EM | 0.13 |
| DROP F1 | 2.89 |

해석:

- SWDE/SQuAD completion은 낮지만 완전히 0은 아니다.
- TriviaQA/NQ/DROP은 매우 낮다.
- 모델이 answer format, QA prompt, exact match에 약하다.
- 단순 pretraining checkpoint라 instruction following / QA robustness가 부족할 수 있다.

## 8. 실패 원인 상세

### 원인 1: 10B token으로는 baseline 격차가 너무 컸다

이번 실험은 10B token만 학습했다. GDN2 baseline은 100B token이다.

10B에서 개선은 있었지만, 격차는 컸다.

```text
Avg 8 acc: ours 45.20 vs GDN2 53.11
LAMBADA acc: ours 34.76 vs GDN2 48.09
HellaSwag: ours 41.46 vs GDN2 56.84
ARC-Challenge: ours 26.11 vs GDN2 38.23
```

이 정도 차이는 단순히 10B를 20B로 늘린다고 바로 뒤집히기 어렵다.

### 원인 2: softmax attention 부재

RULER와 retrieval task는 정보 검색 능력이 중요하다.

Transformer softmax attention은 token-to-token interaction을 직접 만든다.

이번 모델은 fixed recurrent state를 쓴다. 효율성은 좋지만, state에 압축된 정보만 사용할 수 있다.

정보가 state에 잘못 압축되거나 지워지면 나중에 복구할 방법이 없다.

### 원인 3: gate와 Kaczmarz step의 상호작용이 검증되지 않았다

GDN2의 gate는 이미 update를 조절한다.

우리는 여기에 Kaczmarz normalization을 추가했다.

가능한 문제:

- erase가 약해져서 오래된 정보가 남는다.
- write가 약해져서 needle 정보가 충분히 저장되지 않는다.
- key norm에 따라 task별로 update가 불안정하게 달라진다.
- gate가 학습해야 할 역할을 normalization이 방해한다.

이것은 ablation 없이는 확정할 수 없다. 그러나 RULER 결과가 낮으므로 강하게 의심할 수 있다.

### 원인 4: 4K 학습과 8K 평가의 분포 차이

학습은 4K다. 평가는 8K까지 했다.

recurrent 구조는 이론적으로 더 긴 stream을 처리할 수 있지만, 성능은 학습 길이에 영향을 받는다.

8K 점수가 특히 낮은 것은 이 분포 차이와 관련 있을 수 있다.

### 원인 5: 일반 언어 모델링 능력 부족

RULER도 결국 prompt를 읽고, 답을 생성하고, format을 맞춰야 한다.

기본 language modeling 성능이 낮으면 long-context 구조가 있어도 benchmark가 낮게 나온다.

### 원인 6: 평가 format과 모델 상태의 불일치

이 모델은 FineWeb-Edu pretraining만 했다.

Instruction tuning, QA tuning, retrieval tuning은 하지 않았다.

따라서 real-world retrieval exact match 계열에서 약한 것은 어느 정도 예상 가능하다. 하지만 GDN2 논문 baseline도 같은 pretraining 계열 비교라면 이것만으로 설명되지는 않는다.

## 9. 잘한 것

성능은 실패했지만, 작업 산출물은 남았다.

### 인프라

- 10B token 학습을 완료했다.
- 1B 단위 checkpoint를 만들었다.
- Hugging Face 업로드 구조를 만들었다.
- 표준 평가, RULER 평가, real-world retrieval 평가를 준비했다.
- RULER 중간 저장 기능을 추가했다.
- GPU 병렬 평가 스크립트와 retry 스크립트를 만들었다.

### 문서

- 구조 설명 문서
- 초심자용 설명 문서
- 학습 runbook
- 평가 계획 문서
- 실패 요약 문서
- 결과 비교 문서

### 데이터

- 01B-10B 표준 benchmark learning curve
- 10B RULER 일부 결과
- 10B real-world retrieval 결과

이 자료는 다음 후보를 만들 때 baseline으로 쓸 수 있다.

## 10. 못한 것

- 성능 목표 미달
- GDN2 대비 실패
- RULER long-context 가설 검증 실패
- Kaczmarz step이 유효하다는 증거 없음
- 20B 이상으로 밀어붙일 만한 신호 부족
- hybrid를 배제했을 때 표준 benchmark 손실이 컸음

## 11. 앞으로 어떻게 해야 하나

무작정 20B, 30B로 늘리는 것은 우선순위가 낮다.

먼저 구조 ablation을 해야 한다.

### 1순위: Kaczmarz 제거 ablation

비교:

```text
GDN2 gate only
vs
GDN2 gate + Kaczmarz
```

목표:

- Kaczmarz step이 실제로 도움이 되는지 확인
- RULER에서 update가 약해지는지 확인

### 2순위: Kaczmarz strength 조절

현재 step이 너무 강하거나 약할 수 있다.

실험 후보:

- `lambda_t` clipping 제거
- `lambda_t` 상한 조절
- write gate에만 적용
- erase gate에만 적용
- learned scale 추가

### 3순위: GDN2 원본 재현

같은 코드, 같은 데이터, 같은 10B budget에서 원본 GDN2 recurrence를 먼저 돌려야 한다.

그 결과와 비교해야 Kaczmarz 후보의 의미를 정확히 알 수 있다.

현재는 논문 100B baseline과 비교하고 있어서 data/token budget 차이가 크다.

### 4순위: RULER 우선 smoke test

다음 후보는 10B까지 바로 태우면 안 된다.

권장:

```text
0.5B 또는 1B 학습
-> RULER 1K/2K/4K quick eval
-> 신호가 있으면 5B
-> 그 다음 10B
```

### 5순위: long-context continuation

4K에서 바로 8K RULER가 약했다.

다음은 4K pretraining 후 다음을 시도해야 한다.

- 8K short continuation
- 16K short continuation
- RULER length curriculum

### 6순위: hybrid 재검토

이번 실험은 차별점을 위해 hybrid를 배제했다.

하지만 결과적으로 pure recurrent linear attention만으로는 표준 benchmark와 retrieval 모두 약했다.

성능이 목표라면 hybrid를 다시 고려해야 한다.

가능한 방향:

- 대부분 GDN2/linear block
- 일부 layer만 softmax attention
- sparse attention + recurrent memory
- retrieval-heavy layer만 attention

## 12. 다음 실험을 한다면 최소 조건

다음 실험은 아래 조건을 만족해야 한다.

1. 원본 GDN2 10B local baseline을 먼저 만든다.
2. 후보 모델은 1B smoke에서 RULER 개선 신호가 있어야 한다.
3. 표준 benchmark가 baseline보다 크게 나빠지면 즉시 중단한다.
4. RULER는 중간 저장으로 task/length별 바로 확인한다.
5. 10B 학습 전에 ablation 2개 이상을 비교한다.

## 13. 최종 판단

이번 실험은 실패다.

하지만 실패 방식은 유용하다.

알게 된 것:

- GDN2 gate를 가져오는 것만으로는 충분하지 않다.
- Kaczmarz update normalization은 현재 설정에서 유효성이 없다.
- 10B token만으로 GDN2 100B를 이기는 것은 현실적이지 않았다.
- pure recurrent linear attention 단독은 benchmark 경쟁력이 부족했다.
- long-context 이론상 장점은 실제 RULER 점수로 검증되지 않았다.

따라서 현 상태에서 GPU를 더 태우는 것은 타당하지 않다.

다음 단계는 더 큰 학습이 아니라 더 작은 ablation이다.
