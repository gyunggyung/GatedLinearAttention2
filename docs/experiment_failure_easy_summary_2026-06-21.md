# GatedLinearAttention2 실험 실패 쉬운 요약

이 문서는 이번 `GatedLinearAttention2` / `gdn2_kla_1.3B` 실험을 사전 지식이 거의 없는 사람도 이해할 수 있게 정리한 쉬운 요약이다.

결론부터 말하면, **이번 실험은 성능 목표 기준으로 실패**다.

학습은 끝났고, 평가 코드도 만들었고, 여러 benchmark도 돌렸다. 하지만 우리가 기대한 핵심 목표인 “GDN2보다 더 좋은 리니어 어텐션 구조”는 증명하지 못했다.

## 우리가 하려던 것

Transformer는 긴 문맥에서 계산량과 KV cache가 커진다. 그래서 더 긴 문맥을 효율적으로 다루기 위해 linear attention 계열 모델을 보려고 했다.

단순 linear attention은 빠르지만 약점이 있다.

- 과거 정보를 정밀하게 다시 찾는 능력이 softmax attention보다 약할 수 있다.
- 오래된 정보를 잘 지우지 못할 수 있다.
- state tracking에서 실수가 날 수 있다.

그래서 우리는 GDN2의 장점을 가져오려고 했다.

GDN2는 memory를 업데이트할 때 다음 두 가지 gate를 쓴다.

| Gate | 역할 |
|---|---|
| erase gate | 기존 memory에서 무엇을 지울지 정함 |
| write gate | 새 정보 중 무엇을 쓸지 정함 |

우리 후보는 여기에 Kaczmarz식 update 크기 보정까지 추가했다.

쉽게 말하면:

```text
그냥 계속 더하는 linear attention이 아니라,
무엇을 지우고 무엇을 쓸지 조절하는 linear attention을 만들었다.
```

## 정확히 어떤 모델인가

이번 모델은 Transformer가 아니다.

그리고 Transformer와 linear attention을 섞은 hybrid도 아니다.

이번 모델은 다음에 가깝다.

```text
pure recurrent gated linear attention
```

즉, 과거 token마다 KV cache를 계속 저장하는 방식이 아니라, 고정 크기 memory state를 계속 업데이트하는 방식이다.

장점이 기대된 부분:

- KV cache가 context length에 비례해서 커지지 않는다.
- 이론상 긴 stream을 계속 읽을 수 있다.
- GDN2식 gate가 있으니 단순 linear attention보다 state update가 나을 수 있다.

하지만 실제 실험 결과는 기대보다 낮았다.

## 결과 한 줄 요약

**GDN2를 못 이겼다.**

**Mamba-2도 못 이겼다.**

**Gated DeltaNet / KDA도 못 이겼다.**

**Mamba-3도 전체로는 못 이겼다.**

Mamba-3 일부 항목에서 작은 승리가 있었지만, 전체 성능 승리라고 볼 수 없다.

## 표준 벤치 결과

01B에서 10B까지 학습하면서 점수는 좋아졌다.

| Checkpoint | Avg 8 acc | Wiki PPL | LAMBADA acc | HellaSwag | ARC-C |
|---|---:|---:|---:|---:|---:|
| 01B | 39.88 | 45.14 | 20.67 | 31.96 | 21.84 |
| 05B | 43.68 | 29.96 | 28.76 | 39.67 | 25.00 |
| 10B | 45.20 | 24.52 | 34.76 | 41.46 | 26.11 |
| GDN2 recurrent | 53.11 | 15.90 | 48.09 | 56.84 | 38.23 |

좋아지긴 했다. 하지만 GDN2와 차이가 너무 크다.

## RULER 결과

RULER는 long-context 능력을 보는 중요한 평가다.

나온 10B RULER 결과는 다음과 같다.

| Task | 1K | 2K | 4K | 8K |
|---|---:|---:|---:|---:|
| `niah_single_2` | 92.8 | 75.8 | 25.8 | 4.8 |
| `niah_single_3` | 24.4 | 12.0 | 2.2 | 0.8 |
| `niah_multikey_1` | 21.2 | 18.2 | 20.2 | 9.6 |

문제는 4K, 8K에서 점수가 크게 무너졌다는 점이다.

GDN2와 비교하면 거의 대부분 크게 진다.

즉, “우리 모델이 long-context에서 강하다”는 주장은 현재 결과로는 할 수 없다.

## 왜 망했나

쉬운 말로 정리하면 이유는 이렇다.

### 1. 학습량이 너무 적었다

우리는 10B token을 학습했다.

비교 대상 GDN2 논문 baseline은 100B token이다.

10분의 1만 학습하고 이기려 한 것이다. 점수가 오르긴 했지만, 격차를 줄이기에는 부족했다.

### 2. Transformer attention을 안 썼다

우리는 차별점을 위해 hybrid를 하지 않았다.

그래서 Transformer softmax attention의 강한 검색 능력은 없다.

긴 문맥에서 효율은 기대할 수 있지만, 정확히 필요한 token을 다시 찾는 능력은 약할 수 있다.

### 3. GDN2 gate만 가져온다고 GDN2가 되는 것이 아니다

GDN2 성능은 gate 하나만으로 나오는 것이 아니다.

다음이 전부 맞아야 한다.

- update 수식
- gate 설계
- decay 설계
- normalization
- initialization
- optimizer
- 학습 데이터 순서
- 학습 token 수

우리는 gate를 가져왔지만, Kaczmarz step도 추가했다. 이 추가가 오히려 memory update를 약하게 만들었을 수 있다.

### 4. Kaczmarz step이 너무 보수적이었을 수 있다

Kaczmarz step은 update 크기를 key norm으로 보정한다.

목표는 안정성이다.

하지만 중요한 정보를 강하게 써야 할 때 update가 너무 약해졌을 수 있다.

RULER에서 needle 정보를 잘 못 찾은 것은 이 가능성을 보여준다.

### 5. 4K 학습만으로 8K 일반화가 잘 되지 않았다

학습 context는 4K였다.

RULER는 8K까지 봤다.

recurrent 구조라 이론상 더 긴 길이를 처리할 수는 있지만, 실제 성능은 학습 분포와 state capacity에 영향을 받는다.

이번에는 8K에서 크게 무너졌다.

## 잘한 점

성능은 실패했지만 남은 것은 있다.

- 10B token 학습은 완료했다.
- 1B 단위 checkpoint를 만들었다.
- 표준 벤치 learning curve를 확보했다.
- RULER 평가 코드를 만들었다.
- RULER 중간 저장 기능을 만들었다.
- 독립 추론 런타임 폴더를 만들었다.
- 문서와 실행 방법을 정리했다.

즉, 연구 인프라는 남았다.

## 못한 점

- GDN2를 못 이겼다.
- Mamba-2를 못 이겼다.
- Gated DeltaNet / KDA를 못 이겼다.
- Mamba-3도 전체적으로 못 이겼다.
- RULER long-context 장점을 보여주지 못했다.
- Real-world retrieval도 낮았다.
- 10B token으로 100B baseline을 이기겠다는 목표는 실패했다.

## 20B를 더 하면 나아질까

조금은 나아질 가능성이 있다.

하지만 지금 추세로는 20B를 더 해도 GDN2를 이길 가능성은 낮다.

현재 추세상 20B 예측은 대략 다음이다.

| 지표 | 20B 예상 |
|---|---:|
| Avg 8 acc | 약 47.0 |
| LAMBADA acc | 약 38.9 |
| HellaSwag | 약 45.0 |
| ARC-Challenge | 약 27.3 |

GDN2 recurrent는 다음이다.

| 지표 | GDN2 recurrent |
|---|---:|
| Avg acc | 53.11 |
| LAMBADA acc | 48.09 |
| HellaSwag | 56.84 |
| ARC-Challenge | 38.23 |

아직 차이가 크다.

따라서 단순히 20B를 더 학습하는 것은 비용 대비 좋지 않다.

## 앞으로 하려면 어떻게 해야 하나

같은 방향을 계속하려면 무작정 학습량을 늘리기보다 원인을 나눠서 확인해야 한다.

우선순위는 다음이다.

1. Kaczmarz step을 제거한 모델과 비교한다.
2. Kaczmarz strength를 약하게 조절한다.
3. GDN2 원본 recurrence를 같은 코드와 같은 데이터로 재현한다.
4. 4K가 아니라 8K 또는 16K continuation을 짧게 해본다.
5. RULER를 먼저 작게 돌려서 long-context 가능성을 본다.
6. 그래도 안 되면 hybrid attention을 다시 검토한다.

## 최종 결론

이번 실험은 망한 것이 맞다.

정확히는 다음과 같다.

```text
학습과 평가 인프라는 성공했다.
하지만 모델 아이디어의 성능 검증은 실패했다.
```

GPU를 계속 태우면서 20B, 30B로 밀어붙일 상황은 아니다.

먼저 구조 ablation을 해야 한다.
