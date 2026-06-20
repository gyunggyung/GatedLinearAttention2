# gdn2_kla_1.3B는 정확히 무엇을 하는가

이 문서는 지금 실험하려는 `gdn2_kla_1.3B`가 정확히 무엇인지, 이것이 attention인지, 왜 long-context에 유리할 수 있는지, 그리고 "10T 토큰도 가능한가"라는 질문에 대한 정확한 답을 정리한다.

## 짧은 결론

`gdn2_kla_1.3B`는 **softmax attention이 아니라 recurrent linear attention 계열의 token mixer**다.

더 정확히는:

```text
Linear Attention
-> DeltaNet / Gated DeltaNet
-> GDN-2: erase gate와 write gate를 channel-wise로 분리
-> gdn2_kla_1.3B: GDN-2 gate에 Kaczmarz식 update-size normalization 추가
```

즉 지금 하는 일은 **Transformer의 full softmax attention을 개선하는 것**이 아니라, **linear attention의 recurrent memory update rule을 개선하는 것**이다.

하이브리드 attention은 쓰지 않는다. SWA, full attention, MLA도 쓰지 않는다.

## 이게 attention인가

답은 "넓은 의미에서는 attention 계열이고, 좁은 의미의 Transformer softmax attention은 아니다"이다.

Transformer attention은 보통 다음과 같다.

```math
o_t = \sum_{i \le t}
\operatorname{softmax}(q_t^\top k_i)_i v_i.
```

이 방식은 과거 token들의 `K, V`를 모두 저장하고, query가 과거 전체를 다시 본다. 그래서 exact retrieval에는 강하지만 KV cache가 context length에 비례해서 커진다.

Linear attention은 softmax attention의 kernelized/linearized 형태로 볼 수 있다.

```math
o_t =
\frac{
\phi(q_t)^\top \sum_{i \le t}\phi(k_i)v_i^\top
}{
\phi(q_t)^\top \sum_{i \le t}\phi(k_i)
}.
```

핵심은 과거 전체를 token별 KV cache로 저장하지 않고, 누적 state로 압축한다는 점이다.

단순화하면:

```math
S_t = S_{t-1} + k_t v_t^\top,
\qquad
o_t = S_t^\top q_t.
```

이때 `S_t`가 과거 문맥을 압축한 recurrent memory다. 그래서 GDN, GDN-2, KDA, Kimi Delta Attention 같은 모델은 보통 **linear attention / recurrent attention / fast-weight memory** 계열로 분류된다.

## GDN-2는 무엇을 개선했나

단순 linear attention은 state에 계속 쓰기만 하면 서로 다른 key-value association이 섞인다. 긴 문맥에서는 이 interference가 커진다.

GDN-2는 이 문제를 delta rule로 다룬다.

기호:

- `S_t in R^{d_k x d_v}`: recurrent memory state
- `q_t, k_t in R^{d_k}`: query, key
- `v_t in R^{d_v}`: value
- `D_t = Diag(alpha_t)`: channel-wise decay
- `b_t in [0, 1]^{d_k}`: key-side erase gate
- `w_t in [0, 1]^{d_v}`: value-side write gate

GDN-2 recurrence:

```math
S_t
=
\left(I - k_t(b_t \odot k_t)^\top \right)D_tS_{t-1}
+
k_t(w_t \odot v_t)^\top.
```

이 식은 두 가지 일을 한다.

1. `b_t`로 기존 state에서 어떤 key 방향을 지울지 고른다.
2. `w_t`로 새 value의 어떤 channel을 쓸지 고른다.

기존 GDN/KDA 계열은 erase와 write가 같은 scalar gate에 묶이는 경우가 많다. GDN-2는 erase와 write를 분리해서 memory edit를 더 세밀하게 만든다.

## 우리가 추가한 것은 무엇인가

`gdn2_kla_1.3B`는 GDN-2의 gate에 Kaczmarz Linear Attention 방식의 update-size normalization을 추가한다.

Kaczmarz step:

```math
\lambda_t
=
\frac{\eta_t}{\|k_t\|_2^2 + \epsilon},
\qquad
\eta_t \in [0, 1].
```

후보 recurrence:

```math
S_t
=
\left(I - k_t(\lambda_t b_t \odot k_t)^\top \right)D_tS_{t-1}
+
k_t(\lambda_t w_t \odot v_t)^\top.
```

즉 GDN-2의 `b_t`, `w_t`가 결정하는 "어디를 지우고 쓸지"는 유지하면서, `lambda_t`가 "얼마나 세게 업데이트할지"를 key norm 기준으로 조절한다.

## 왜 이게 필요하나

recurrent linear attention에서 long-context 성능의 병목은 과거를 못 보는 것이 아니라, **고정 크기 state에 너무 많은 정보를 압축하면서 생기는 memory interference**다.

긴 문맥에서는 다음 update가 수천, 수만, 수백만 번 반복된다.

```math
S_0 \to S_1 \to S_2 \to \cdots \to S_T.
```

이때 update가 너무 강하면:

```text
새 token 하나가 기존 memory를 과하게 지운다.
```

update가 너무 약하면:

```text
중요한 새 정보가 state에 충분히 쓰이지 않는다.
```

GDN-2는 무엇을 지울지/쓸지를 개선한다. 하지만 update의 크기 자체는 key vector의 scale에 영향을 받을 수 있다.

`gdn2_kla_1.3B`는 여기에 key-norm normalization을 넣어서, key scale 때문에 update가 과하거나 약해지는 문제를 줄이려는 실험이다.

## long-context에서 유리한 방식인가

이론적으로는 long-context state tracking에 유리한 방향이 맞다.

이유:

1. state가 recurrent하게 다음 token으로 전달된다.
2. 과거 token 수가 늘어도 layer별 recurrent state 크기는 고정이다.
3. GDN-2는 오래된 memory를 선택적으로 지우고 새 memory를 선택적으로 쓴다.
4. Kaczmarz step은 그 edit의 크기를 안정화한다.

이 점에서 GDN/Mamba/RNN 계열과 같은 장점이 있다. 즉 memory가 feedforward depth 안에서만 존재하는 것이 아니라, 시간축을 따라 계속 흐른다.

하지만 중요한 한계도 있다.

full attention은 과거 token을 그대로 다시 읽을 수 있다. recurrent linear attention은 과거를 `S_t`에 압축한다. 따라서:

- 상태 추적, 반복 업데이트, 긴 흐름 유지에는 유리할 수 있다.
- exact needle retrieval, rare-token verbatim lookup은 full attention보다 불리할 수 있다.
- state size가 고정이므로 무한한 정보를 손실 없이 저장하는 것은 불가능하다.

따라서 정확한 주장은 이것이다.

```text
gdn2_kla_1.3B는 long-context에서 KV cache를 늘리지 않고 state tracking을 잘 하도록 설계한 recurrent linear attention 개선 실험이다.
```

## GDN-2 논문은 몇 K로 실험했나

GDN-2 논문에서 pretraining recipe는 4K다. 논문 본문 실험 설정에는 training length가 4K tokens이고, hybrid 모델의 sliding-window attention size가 2K라고 되어 있다.

하지만 평가가 전부 4K에서 끝나는 것은 아니다. Table 3의 RULER synthetic retrieval은 일부 task를 8K까지 평가한다.

- S-NIAH-1: 1K, 2K, 4K, 8K
- S-NIAH-2: 1K, 2K, 4K, 8K
- S-NIAH-3: 1K, 2K, 4K
- MK-NIAH-1: 1K, 2K, 4K

즉 논문의 정확한 해석은 다음이다.

```text
학습 기본 길이: 4K
RULER 평가 길이: task에 따라 1K/2K/4K/8K
공식 32K pretraining recipe: 논문/공식 README의 기본값으로는 제시되지 않음
```

또 논문 throughput figure는 sequence length 2K/4K/8K/16K에서 hybrid 1.3B 모델의 training throughput scaling을 비교한다. 이것은 긴 sequence에서 kernel이 얼마나 효율적인지 보여주는 자료이지, 32K나 128K pretraining 결과를 보고한 것은 아니다.

다음처럼 말하면 안 된다.

```text
모든 long-context retrieval에서 full attention보다 항상 좋다.
10B 학습만으로 100B GDN-2 benchmark를 반드시 이긴다.
10T token 정보를 손실 없이 저장한다.
```

## KV cache는 늘어나나

순수 `gdn2_kla_1.3B` 아키텍처 기준으로는 softmax attention의 KV cache가 없다.

Transformer attention layer의 decode cache는 보통 다음처럼 context length `T`에 비례한다.

```math
\text{KV cache size}
\propto
L \cdot T \cdot H \cdot d.
```

여기서:

- `L`: layer 수
- `T`: context length
- `H`: head 수
- `d`: head dimension

GDN-2 계열은 layer마다 recurrent state를 저장한다.

```math
\text{GDN state size}
\propto
L \cdot H \cdot d_k \cdot d_v.
```

여기에는 `T`가 없다. context가 길어져도 state 크기는 고정이다.

이 저장소의 후보 설정은 GDN state가 대략 다음 크기다.

```text
per layer state ~= num_heads * d_k * d_v
                = 16 * 128 * 128
                = 262,144 scalars
```

18 layer이면 sequence 하나당:

```text
18 * 262,144 = 4,718,592 scalars
```

bf16이면 recurrent state만 대략 9 MB 수준이다. 여기에 short convolution cache 같은 작은 state가 추가된다.

반면 Transformer KV cache는 token이 늘 때마다 계속 커진다.

## 10T 토큰도 한 번에 처리 가능한가

여기서 "한 번에"라는 말은 구분해야 한다.

### 1. 10T training tokens

가능한 말이다. 하지만 이것은 하나의 sequence 길이가 10T라는 뜻이 아니다.

예를 들어 현재 기본 10B token 학습은 다음처럼 많은 batch/chunk를 반복해서 처리한다.

```text
sequence length = 4K
token budget = 10B
micro iterations = 76,296
```

즉 10B token은 training budget이지 context length가 아니다.

10T token 학습도 같은 의미라면 가능하다. 다만 시간과 비용이 매우 크다.

### 2. context length가 10T인 하나의 sequence

아키텍처의 recurrent state 관점에서는 stream 처리 자체는 가능하다.

```text
token 1 처리 -> state update
token 2 처리 -> state update
...
token 10T 처리 -> state update
```

이때 KV cache는 10T에 비례해서 늘지 않는다. state 크기는 고정이다.

하지만 현실적으로 "10T 토큰을 한 번에 GPU tensor로 넣어서 forward"하는 것은 불가능하다.

이유:

- 입력 tensor 자체가 너무 크다.
- 출력 logits를 모든 위치에 대해 만들면 메모리가 터진다.
- prefill compute time은 여전히 `O(T)`라서 10T token은 시간이 막대하다.
- 고정 크기 state가 10T token의 모든 정보를 손실 없이 보존할 수는 없다.
- 현재 pretraining wrapper는 4K/32K/64K/128K 같은 block 단위 학습을 기준으로 작성돼 있다.

정확한 표현은 다음과 같다.

```text
GDN-2/KLA 구조는 10T token stream을 state 크기 증가 없이 순차적으로 소비할 수는 있다.
그러나 10T token을 하나의 giant forward batch로 "한 번에" 처리하는 것은 아니다.
```

### 3. 현재 코드가 바로 10T streaming inference를 지원하나

현재 코드는 pretraining이 중심이다.

`GatedDeltaNet2` layer 자체에는 recurrent state와 short-conv state를 cache로 들고 가는 인터페이스가 있다. 하지만 이 저장소의 `GPT` wrapper는 현재 pretraining/full-sequence forward를 중심으로 되어 있고, GDN recurrent cache를 완전한 streaming generation API로 연결하는 serving glue는 별도로 더 정리해야 한다.

즉:

```text
아키텍처: KV cache 없이 streaming 가능
현재 학습 코드: 4K block pretraining 중심, 32K 이상은 후속 long-context 확장 실험
실제 10T streaming serving: 별도 cache wrapper와 generation loop 필요
```

## 지금 실행하는 실험은 무엇인가

지금 실행하려는 것은 10B token pretraining ablation이다.

```bash
./scripts/pretrain_gdn2_kla_10bt.sh
```

## FineWeb-Edu 100BT에서 10B는 어떻게 뽑나

현재 구현은 별도 10B subset을 만들지 않는다. `sample/100BT` parquet 전체를 streaming으로 열고, global trained tokens가 10B에 도달하면 학습을 종료한다.

```text
sorted parquet files
-> streaming parquet dataset
-> split_dataset_by_node(rank, world_size)
-> tokenizer
-> fixed-length chunks
-> local chunk buffer shuffle
-> stop at 10B tokens
```

따라서 현재 방식은 완전한 random 10B sampling이 아니다. 파일 순서와 shard 분할의 영향을 받는다. local buffer 안에서 token chunk 순서를 일부 섞지만, 100BT 전체에서 균일하게 문서를 뽑는 것은 아니다.

실험 기준:

- 빠른 구현 검증이면 현재 방식으로 충분하다.
- 공정한 ablation이면 seed가 고정된 random manifest가 더 낫다.
- 공식 GDN-2 learning curve와 비교하려면 공식 데이터 순서가 필요하고, 그 순서로 10B에서 멈추는 것이 가장 직접적이다.

이 저장소의 현재 기본은 "same source, same 4K recipe, lower token budget"이다. "uniform random 10B subset"이라고 부르면 안 된다.

이 실험은 다음 질문에 답하기 위한 것이다.

```text
GDN-2의 channel-wise erase/write gate에
Kaczmarz식 update-size normalization을 붙이면
plain recurrent GDN-2보다 long-context state tracking이 좋아지는가?
```

성공 기준은 다음이다.

- plain `gdn2_1.3B` 대비 validation loss가 나빠지지 않는다.
- RULER S-NIAH / MK-NIAH가 좋아진다.
- number-range tracking, entity-state update가 좋아진다.
- throughput과 안정성이 크게 망가지지 않는다.

## 요약

`gdn2_kla_1.3B`는 attention을 완전히 버리는 것이 아니다. 정확히는 **softmax attention을 recurrent linear attention으로 대체하고, 그 memory update rule을 개선하는 실험**이다.

이 방식은 long-context에서 KV cache가 커지지 않는다는 큰 장점이 있다. 대신 모든 과거 token을 정확히 다시 읽는 full attention과 달리, 고정 크기 state에 압축한다는 한계가 있다.

따라서 이 후보의 가치는 "무한 context를 완벽히 기억한다"가 아니라, 다음에 있다.

```text
긴 token stream을 고정 state로 처리하면서,
GDN-2보다 안정적으로 memory를 지우고 쓰는가?
```
