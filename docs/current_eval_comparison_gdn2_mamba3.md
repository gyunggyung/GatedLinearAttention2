# 현재 평가 비교: GatedLinearAttention2 vs GDN2 / Mamba-3

마지막 갱신: 2026-06-21 KST, 첫 10B token 학습 후 평가 진행 중.

이 문서는 현재까지 나온 점수, 아직 실행 중인 평가, 그리고 지금 숫자로 말할 수 있는 것과 말하면 안 되는 것을 정리한다.

## 한 줄 결론

최종 10B-token checkpoint는 현재 완료된 GDN2 논문 Table 2 표준 벤치마크 기준으로 **Gated DeltaNet-2를 이긴 항목이 없다.**

Mamba-3와 비교하면 기준에 따라 다르다.

| 비교 대상 | 현재 10B 결과 |
|---|---|
| Gated DeltaNet-2 recurrent | 완료된 Table 2 기준 0승 |
| Mamba-3 recurrent SISO | BoolQ 1개 승리 |
| Mamba-3 recurrent MIMO | BoolQ 1개 아주 근소한 승리 |
| Mamba-3 hybrid SISO | 0승 |
| Mamba-3 hybrid MIMO | 0승 |

BoolQ에서 recurrent Mamba-3를 이긴 것은 숫자상 사실이다. 그러나 전체 모델이 더 좋다고 말할 수는 없다. Perplexity, LAMBADA, PIQA, HellaSwag, WinoGrande, ARC, OpenBookQA에서는 명확히 낮다.

RULER는 아직 핵심 미완료 항목이다. 10B RULER 작업은 GPU 0-3에서 실행 중이며, 오래 걸리고 있다. 기존 10B RULER는 중간 저장 기능이 들어가기 전에 시작되어 끝날 때까지 split JSON이 보이지 않는다.

## 우리가 정확히 무엇을 만든 것인가

현재 평가 중인 모델 이름은 다음과 같다.

```text
gdn2_kla_1.3B
```

핵심 checkpoint는 다음이다.

```text
runs/outputs/tsz128x4k_10B_gdn2_kla_1.3B_fineweb_edu_10bt/hf_checkpoints/checkpoint-10B
```

학습 설정은 다음과 같다.

| 항목 | 값 |
|---|---|
| 학습 토큰 | 10B |
| 학습 context length | 4K |
| 데이터 | FineWeb-Edu sample/100BT |
| 구조 계열 | recurrent-only gated linear attention |
| Transformer softmax attention | 사용 안 함 |
| Hybrid attention | 사용 안 함 |
| Tokenizer | TinyLlama/TinyLlama_v1.1 |

중요한 정리:

```text
우리는 Transformer + linear attention hybrid를 만든 것이 아니다.
```

사용자가 하이브리드는 하지 말고 단일 후보만 보자고 했기 때문에, full softmax attention block은 넣지 않았다. 따라서 Transformer의 장점인 정밀한 token-to-token softmax 검색 능력은 구조적으로 들어 있지 않다.

현재 모델은 다음에 가깝다.

```text
GDN2 스타일 recurrent linear attention
+ GDN2의 erase/write gate
+ Kaczmarz식 key-norm-normalized update
```

즉, 사용자가 가져온 linear attention 설명에서 말하는 “softmax attention의 정확한 pairwise interaction을 포기하고 효율을 얻는” 계열이 맞다. 여기에 GDN2식 게이트를 붙여서 단순 linear attention의 약점인 memory overwrite, 누적 오염, 지우기 어려움 문제를 줄이려는 구조다.

## 리니어 어텐션 설명과 우리 모델의 관계

사용자가 가져온 설명의 요지는 다음이다.

```text
Linear attention은 softmax attention의 O(T^2) 비용을 줄이기 위해
kernel feature와 행렬 곱 순서 변경을 사용한다.
그래서 sequence length에 대해 더 선형적인 비용을 갖지만,
softmax attention의 정밀한 token-to-token 표현력은 일부 잃을 수 있다.
```

우리 모델도 이 계열이다. 다만 가장 단순한 linear attention처럼 state에 계속 더하기만 하는 방식은 아니다.

단순 linear attention은 보통 다음 느낌이다.

```text
S_t = S_{t-1} + k_t v_t^T
o_t = q_t^T S_t
```

이 방식은 빠르지만 문제가 있다.

- 오래된 정보가 잘 지워지지 않는다.
- 비슷한 key가 반복되면 memory가 오염될 수 있다.
- state tracking에서 “이전 상태를 새 상태로 갱신”하는 능력이 약할 수 있다.

GDN2는 여기에 gate를 넣는다.

```text
b_t = erase gate
w_t = write gate
```

의미는 다음과 같다.

| 기호 | 의미 |
|---|---|
| `q_t` | 지금 무엇을 읽을지 정하는 query |
| `k_t` | 새 정보를 어느 주소에 넣을지 정하는 key |
| `v_t` | 실제로 쓸 값 |
| `S_t` | 고정 크기 recurrent memory state |
| `b_t` | key 방향에서 무엇을 지울지 정하는 erase gate |
| `w_t` | value 방향에서 무엇을 쓸지 정하는 write gate |

GDN2식 recurrent update는 다음 직관을 갖는다.

```text
1. 이전 state에서 decay로 일부를 잊는다.
2. erase gate b_t로 기존 memory 중 관련 부분을 지운다.
3. write gate w_t로 새 value 중 쓸 부분만 쓴다.
4. query q_t로 업데이트된 state를 읽는다.
```

우리 모델은 이 GDN2식 erase/write gate를 사용한다. 여기에 추가로 Kaczmarz식 update 크기 정규화를 넣었다.

```math
\lambda_t = \frac{\eta_t}{\|k_t\|_2^2 + \epsilon}
```

직관은 간단하다.

- key가 너무 크면 update가 과하게 세질 수 있다.
- key가 너무 작으면 update가 너무 약해질 수 있다.
- 그래서 key norm으로 update step을 보정한다.

요약하면 다음이다.

```text
우리는 linear attention에 GDN2 스타일의 gate를 붙인 것이 맞다.
정확히는 GDN2의 recurrent linear attention update에
Kaczmarz식 update normalization을 추가한 모델이다.
```

## 하지만 왜 점수가 아직 낮은가

구조적으로 좋은 아이디어를 넣었다고 해서 바로 GDN2나 Mamba-3를 이기는 것은 아니다.

현재 가장 큰 이유는 네 가지다.

1. 학습량이 10B token뿐이다.
2. 비교 대상 GDN2/Mamba-3 논문값은 100B token 학습 기준이다.
3. 우리는 hybrid softmax attention을 쓰지 않았기 때문에 Transformer식 정밀 검색 능력은 없다.
4. GDN2 gate + Kaczmarz normalization이 실제로 좋은지 아직 충분히 검증되지 않았다.

특히 Table 2의 LAMBADA, HellaSwag, ARC 같은 표준 언어/상식 벤치마크는 단순 long-context memory만으로 풀리지 않는다. 일반 언어 모델링 품질, 데이터량, 학습 안정성, 표현력, tokenizer, hyperparameter가 모두 중요하다.

현재 01B에서 10B까지는 분명히 좋아지고 있다.

| Checkpoint | Avg 8 acc | Wiki PPL | LAMBADA acc | HellaSwag norm | ARC-Challenge | BoolQ |
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

개선 추세는 있다. 그러나 GDN2 100B와의 차이는 아직 크다.

## 완료된 Table 2 표준 결과

Social IQA는 현재 설치된 `datasets` 버전에서 legacy dataset script가 막혀 local 평균에서 제외했다.

| Checkpoint | Avg 8 acc | Wiki PPL | LAMBADA PPL | LAMBADA acc | PIQA | HellaSwag norm | WinoGrande | ARC-Easy | ARC-Challenge | OpenBookQA | BoolQ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01B | 39.88 | 45.14 | 117.08 | 20.67 | 62.24 | 31.96 | 51.70 | 50.29 | 21.84 | 18.20 | 62.11 |
| 02B | 42.00 | 35.86 | 66.48 | 26.02 | 64.25 | 35.80 | 50.12 | 53.37 | 23.29 | 21.40 | 61.71 |
| 03B | 42.60 | 33.12 | 55.95 | 27.69 | 64.64 | 37.11 | 51.14 | 55.13 | 24.06 | 21.00 | 60.00 |
| 04B | 43.22 | 31.51 | 47.36 | 30.10 | 63.93 | 37.81 | 53.51 | 55.13 | 24.23 | 20.80 | 60.21 |
| 05B | 43.68 | 29.96 | 46.01 | 28.76 | 65.56 | 39.67 | 52.49 | 56.44 | 25.00 | 20.60 | 60.95 |
| 06B | 44.77 | 28.31 | 34.80 | 32.93 | 65.29 | 40.04 | 52.25 | 59.18 | 26.45 | 21.20 | 60.83 |
| 07B | 44.30 | 26.81 | 38.41 | 31.17 | 65.18 | 40.77 | 51.46 | 59.72 | 25.60 | 21.80 | 58.69 |
| 08B | 45.00 | 25.86 | 32.62 | 33.98 | 66.27 | 41.19 | 52.41 | 59.55 | 25.60 | 21.80 | 59.20 |
| 09B | 45.09 | 25.16 | 29.92 | 34.81 | 66.65 | 41.59 | 51.38 | 59.81 | 24.83 | 22.40 | 59.27 |
| 10B | 45.20 | 24.52 | 29.12 | 34.76 | 67.03 | 41.46 | 51.70 | 60.14 | 26.11 | 22.60 | 57.80 |
| GDN2 recurrent | 53.11 | 15.90 | 11.41 | 48.09 | 72.80 | 56.84 | 57.85 | 72.43 | 38.23 | 31.60 | 59.54 |

## 최종 10B vs GDN2 Recurrent

Perplexity는 낮을수록 좋고, 나머지 정확도 계열 지표는 높을수록 좋다.

| 지표 | 우리 10B | GDN2 recurrent | 결과 |
|---|---:|---:|---|
| Wiki PPL | 24.52 | 15.90 | 패배 |
| LAMBADA PPL | 29.12 | 11.41 | 패배 |
| LAMBADA acc | 34.76 | 48.09 | 패배 |
| PIQA | 67.03 | 72.80 | 패배 |
| HellaSwag norm | 41.46 | 56.84 | 패배 |
| WinoGrande | 51.70 | 57.85 | 패배 |
| ARC-Easy | 60.14 | 72.43 | 패배 |
| ARC-Challenge | 26.11 | 38.23 | 패배 |
| OpenBookQA | 22.60 | 31.60 | 패배 |
| BoolQ | 57.80 | 59.54 | 패배 |

현재 결론:

```text
최종 10B checkpoint는 완료된 Table 2 기준 GDN2 recurrent를 이긴 항목이 없다.
```

## 최종 10B vs Mamba-3

### Mamba-3 Recurrent SISO 대비

| 지표 | 우리 10B | Mamba-3 SISO recurrent | 결과 |
|---|---:|---:|---|
| Wiki PPL | 24.52 | 16.30 | 패배 |
| LAMBADA PPL | 29.12 | 12.99 | 패배 |
| LAMBADA acc | 34.76 | 45.06 | 패배 |
| PIQA | 67.03 | 72.31 | 패배 |
| HellaSwag norm | 41.46 | 55.58 | 패배 |
| WinoGrande | 51.70 | 56.20 | 패배 |
| ARC-Easy | 60.14 | 70.45 | 패배 |
| ARC-Challenge | 26.11 | 34.56 | 패배 |
| OpenBookQA | 22.60 | 31.00 | 패배 |
| BoolQ | 57.80 | 55.90 | 승리, +1.90 |

결과:

```text
1승 9패
```

### Mamba-3 Recurrent MIMO 대비

| 지표 | 우리 10B | Mamba-3 MIMO recurrent | 결과 |
|---|---:|---:|---|
| Wiki PPL | 24.52 | 16.45 | 패배 |
| LAMBADA PPL | 29.12 | 11.66 | 패배 |
| LAMBADA acc | 34.76 | 47.82 | 패배 |
| PIQA | 67.03 | 72.36 | 패배 |
| HellaSwag norm | 41.46 | 56.49 | 패배 |
| WinoGrande | 51.70 | 55.78 | 패배 |
| ARC-Easy | 60.14 | 72.38 | 패배 |
| ARC-Challenge | 26.11 | 38.07 | 패배 |
| OpenBookQA | 22.60 | 30.00 | 패배 |
| BoolQ | 57.80 | 57.74 | 승리, +0.06 |

결과:

```text
1승 9패
```

### Mamba-3 Hybrid 대비

| 비교 대상 | 결과 |
|---|---|
| Mamba-3 SISO hybrid | 0승 10패 |
| Mamba-3 MIMO hybrid | 0승 10패 |

가장 가까운 지표는 BoolQ다.

| 지표 | 우리 10B | Mamba-3 SISO hybrid | Mamba-3 MIMO hybrid |
|---|---:|---:|---:|
| BoolQ | 57.80 | 57.86 | 57.98 |

즉, hybrid Mamba-3와 비교하면 BoolQ도 이기지 못한다.

## 09B 중간 checkpoint에서 이긴 항목

최종 10B가 아니라 09B까지 포함하면 BoolQ에서 일부 모델을 이겼다.

| Checkpoint | 비교 대상 | 이긴 항목 |
|---|---|---|
| 09B | Gated DeltaNet recurrent | BoolQ: 59.27 vs 58.78 |
| 09B | Mamba-3 recurrent SISO | BoolQ: 59.27 vs 55.90 |
| 09B | Mamba-3 recurrent MIMO | BoolQ: 59.27 vs 57.74 |
| 09B | Mamba-3 hybrid SISO | BoolQ: 59.27 vs 57.86 |
| 09B | Mamba-3 hybrid MIMO | BoolQ: 59.27 vs 57.98 |

그러나 이것은 모델 전체 승리가 아니다. BoolQ 하나의 중간 checkpoint 승리다.

## Partial Real-World Retrieval 결과

완료된 10B real-world retrieval split은 다음과 같다.

| Task | 우리 10B | GDN2 recurrent | 결과 |
|---|---:|---:|---|
| FDA contains | 5.35 | 19.98 | 패배 |
| TriviaQA exact match | 1.71 | 61.37 | 패배 |
| NQ exact match | 1.02 | 19.64 | 패배 |
| DROP EM | 0.13 | 17.87 | 패배 |
| DROP F1 | 2.89 | 17.87 | 직접 비교해도 패배 |

이 결과는 매우 낮다. 모델이 instruction/QA 형식에 약하거나, 평가 형식이 현재 checkpoint에 불리하거나, 둘 다일 가능성이 있다. 이 결과만으로 long-context 장점을 말하면 안 된다.

## RULER 상태

RULER는 long-context 질문에 가장 중요한 평가다.

현재 실행 중인 10B RULER:

| GPU | RULER task | Lengths |
|---:|---|---|
| 0 | niah_single_1 | 1K, 2K, 4K, 8K |
| 1 | niah_single_2 | 1K, 2K, 4K, 8K |
| 2 | niah_single_3 | 1K, 2K, 4K, 8K |
| 3 | niah_multikey_1 | 1K, 2K, 4K, 8K |

예상 output:

```text
runs/eval/gdn2_paper/10B/splits/ruler_niah_single_1.json
runs/eval/gdn2_paper/10B/splits/ruler_niah_single_2.json
runs/eval/gdn2_paper/10B/splits/ruler_niah_single_3.json
runs/eval/gdn2_paper/10B/splits/ruler_niah_multikey_1.json
```

기존 10B RULER는 중간 저장 기능을 넣기 전에 시작했기 때문에 완료 전에는 점수가 보이지 않는다.

## RULER 중간 저장 수정

RULER가 너무 오래 걸려서 다음 실행부터는 중간 결과를 볼 수 있도록 수정했다.

수정 파일:

```text
scripts/ruler_eval_gla2.py
```

새 동작:

| 시점 | JSON 동작 |
|---|---|
| process 시작 | `status: running` 파일 생성 |
| task 시작 | `current_task` 기록 |
| 각 length 완료 | 해당 length 점수 즉시 기록 |
| 전체 완료 | `status: complete` 기록 |

JSON은 임시 파일에 쓴 뒤 replace하는 방식으로 갱신한다. 따라서 읽는 쪽에서 반쯤 쓰인 JSON을 볼 가능성을 줄였다.

예시 확인 코드:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("runs/eval/gdn2_paper/01B/ruler_table3.json")
data = json.loads(path.read_text())
print(data["status"], data["updated_at_kst"])
print(data["results"])
PY
```

이미 실행 중이던 10B RULER는 재시작하지 않았다. 재시작하면 1시간 이상 계산한 GPU work를 버리게 되기 때문이다.

## 지금 말할 수 있는 것

안전하게 말할 수 있는 것:

- 10B checkpoint는 01B-09B보다 많은 표준 지표에서 개선됐다.
- 최종 10B checkpoint는 표준 Table 2 기준 GDN2 100B보다 아직 많이 낮다.
- 최종 10B checkpoint는 recurrent Mamba-3 대비 BoolQ만 이겼다.
- 최종 10B checkpoint는 hybrid Mamba-3 대비 이긴 항목이 없다.
- Real-world retrieval partial 결과는 현재 매우 약하다.
- RULER가 long-context 성능 판단의 핵심이다.

말하면 안 되는 것:

- 10B checkpoint가 GDN2를 전체적으로 이겼다고 말하면 안 된다.
- Mamba-3를 전체적으로 이겼다고 말하면 안 된다.
- RULER가 끝나기 전까지 long-context 우월성을 주장하면 안 된다.
- 중간 checkpoint의 BoolQ 상승만으로 아키텍처 전체가 좋다고 말하면 안 된다.

## 다음에 해야 할 일

1. 10B RULER split JSON을 기다린다.
2. RULER split을 `runs/eval/gdn2_paper/10B/ruler_table3.json`로 병합한다.
3. `runs/eval/gdn2_paper/GDN2_PAPER_EVAL_RESULTS.md`를 다시 생성한다.
4. 01B-04B RULER learning curve를 확인한다.
5. 어떤 metric에서 GDN2, Mamba-3 recurrent, Mamba-3 hybrid를 이기는지 다시 계산한다.

RULER가 끝나기 전까지 가장 정직한 요약은 다음이다.

```text
모델은 학습되며 좋아지고 있지만, 최종 10B checkpoint는 아직 넓은 벤치마크 승리를 보여주지 못했다.
```
