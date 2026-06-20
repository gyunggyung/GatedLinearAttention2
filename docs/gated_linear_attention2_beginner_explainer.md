# GatedLinearAttention2 초심자 설명서

이 문서는 수학을 거의 모르는 사람도 `GatedLinearAttention2`가 무엇을
하려는 모델인지 이해할 수 있게 쓰는 설명서다. 하지만 용어는 흐리지
않고 정확하게 쓴다. 어려운 수식은 먼저 쉬운 말로 풀고, 그 다음 같은
내용을 수식으로 다시 적는다.

## 한 줄 요약

`GatedLinearAttention2`는 Transformer의 일반적인 softmax attention처럼
과거 token의 `K, V`를 전부 저장해서 다시 보는 모델이 아니다.

대신 과거 내용을 하나의 작은 "기억장부"인 recurrent state에 계속
압축해서 넣고, 새 token이 들어올 때마다 그 장부를 조금씩 고치는
linear attention 계열 모델이다.

이 모델의 핵심 아이디어는 다음이다.

```text
GDN-2의 좋은 점:
  무엇을 지울지와 무엇을 쓸지를 따로 정한다.

우리가 추가한 점:
  얼마나 세게 지우고 쓸지를 key의 크기에 맞춰 자동 조절한다.
```

## 먼저 용어부터 정리

### Token

모델이 읽는 글자 조각이다.

예를 들어 문장:

```text
Artificial intelligence can help education.
```

이 문장은 tokenizer를 거치면 여러 token id로 바뀐다. 모델은 글자 자체를
바로 읽는 것이 아니라 token id를 읽는다.

### Embedding

token id를 숫자 벡터로 바꾼 것이다.

사람에게는 `bank`라는 단어가 글자지만, 모델에게는 길이가 수천인 숫자
목록이다. 이 숫자 목록이 embedding이다.

### Query, Key, Value

attention 계열 모델에서는 보통 token마다 세 가지 벡터를 만든다.

- `query`, 줄여서 `q`: 지금 token이 무엇을 찾고 싶은지 나타내는 벡터
- `key`, 줄여서 `k`: 이 token이 어떤 주소나 이름표를 갖는지 나타내는 벡터
- `value`, 줄여서 `v`: 실제로 저장하거나 꺼낼 내용 벡터

쉬운 비유:

```text
key   = 서랍 이름표
value = 서랍 안에 넣을 내용
query = 어떤 이름표의 서랍을 찾을지 묻는 질문
```

다만 실제 모델에서는 이것들이 사람이 읽는 단어나 문장이 아니라 숫자
벡터다.

### State

여기서 `state`는 모델이 지금까지 읽은 과거 문맥을 압축해서 들고 있는
기억장부다.

Transformer attention은 과거 token마다 `key`와 `value`를 계속 저장한다.
이 저장소를 보통 KV cache라고 부른다.

반면 GDN-2 계열은 과거 token을 전부 저장하지 않고, 다음 행렬 하나에
압축한다.

```text
S_t
```

여기서 `t`는 몇 번째 token까지 읽었는지를 뜻한다. `S_t`는 "t번째
token까지 읽은 뒤의 기억 상태"다.

## Transformer attention과 무엇이 다른가

Transformer의 일반적인 softmax attention은 다음처럼 생각할 수 있다.

```text
1. 과거 token들의 key와 value를 모두 저장한다.
2. 새 token이 들어오면 query를 만든다.
3. query가 과거 key 전체와 비교된다.
4. 관련 있어 보이는 value들을 섞어서 결과를 만든다.
```

수식으로 쓰면 대략 다음과 같다.

```math
o_t = \sum_{i \le t} \operatorname{softmax}(q_t^\top k_i)_i v_i
```

기호 설명:

- `o_t`: t번째 token에서 나온 출력
- `q_t`: t번째 token의 query
- `k_i`: i번째 과거 token의 key
- `v_i`: i번째 과거 token의 value
- `q_t^T k_i`: 지금 query와 과거 key가 얼마나 비슷한지 보는 점수
- `softmax`: 점수들을 더해서 1이 되는 비율로 바꾸는 함수

이 방식은 과거를 정확히 다시 볼 수 있어서 강하다. 하지만 단점도 있다.

```text
context가 길어질수록 KV cache가 계속 커진다.
```

예를 들어 1천 token보다 100만 token을 기억하려면 KV cache도 대략 그만큼
커진다.

## Linear attention은 무엇인가

linear attention은 과거 token을 하나하나 저장하지 않고, 누적 state에
넣는다.

가장 단순한 형태는 이렇게 볼 수 있다.

```math
S_t = S_{t-1} + k_t v_t^\top
```

뜻:

```text
이전 기억장부 S_{t-1}에
새 key-value 쌍 k_t, v_t를 써서
새 기억장부 S_t를 만든다.
```

출력은 이렇게 만든다.

```math
o_t = S_t^\top q_t
```

뜻:

```text
지금 query q_t로 기억장부 S_t를 조회해서 출력 o_t를 만든다.
```

여기서 중요한 점:

```text
S_t의 크기는 context 길이 T에 따라 커지지 않는다.
```

이것이 long-context에서 linear attention이 매력적인 이유다.

하지만 문제가 있다.

```text
고정 크기 장부에 너무 많은 내용을 계속 쓰면 서로 섞인다.
```

이 문제를 memory interference라고 부른다.

## DeltaNet은 무엇을 하려 했나

단순 linear attention은 새 내용을 계속 더하기만 한다.

```math
S_t = S_{t-1} + k_t v_t^\top
```

이러면 오래된 잘못된 기억이나 이미 바뀐 상태를 제대로 지우기 어렵다.

DeltaNet 계열은 이렇게 생각한다.

```text
새 내용을 그냥 더하지 말고,
이미 같은 key에 저장된 오래된 내용을 읽어낸 뒤,
그 차이만큼 고쳐 쓰자.
```

쉬운 예:

```text
장부에 "Alice의 위치 = 학교"라고 적혀 있다.
새 문장에서 "Alice는 집으로 갔다"가 나온다.

그러면 "학교" 위에 무작정 "집"을 더하는 것이 아니라,
Alice 위치 칸의 오래된 내용을 지우고 "집"으로 고쳐야 한다.
```

이것이 state tracking에 중요하다.

## Gate란 무엇인가

`gate`는 0과 1 사이의 조절값이다.

```text
0에 가까우면 거의 막는다.
1에 가까우면 많이 통과시킨다.
```

문을 생각하면 된다.

```text
gate = 0.0  -> 문이 닫힘
gate = 0.5  -> 반쯤 열림
gate = 1.0  -> 문이 열림
```

모델에서는 이 gate도 사람이 정하는 것이 아니라 neural network가 token마다
계산한다.

## GDN-2가 가져온 핵심 아이디어

GDN-2, 즉 Gated DeltaNet-2는 memory update를 두 부분으로 나눈다.

```text
erase gate: 무엇을 지울지 정한다.
write gate: 무엇을 쓸지 정한다.
```

GDN-2 이전의 일부 모델에서는 지우기와 쓰기가 하나의 scalar gate에 묶여
있었다.

scalar gate는 숫자 하나다.

```text
beta_t = 0.7
```

숫자 하나로 지우기도 하고 쓰기도 하면 너무 단순하다.

GDN-2는 이것을 더 세밀하게 바꾼다.

```text
b_t = key 방향별 erase gate
w_t = value 방향별 write gate
```

`b_t`와 `w_t`는 숫자 하나가 아니라 여러 숫자로 된 벡터다.

```text
b_t = [0.1, 0.9, 0.2, ...]
w_t = [0.8, 0.0, 0.5, ...]
```

뜻:

```text
key 쪽 어떤 칸은 많이 지우고, 어떤 칸은 조금만 지운다.
value 쪽 어떤 내용은 많이 쓰고, 어떤 내용은 쓰지 않는다.
```

## GDN-2 수식

GDN-2의 state update는 다음과 같다.

```math
S_t
=
\left(I - k_t(b_t \odot k_t)^\top \right)D_tS_{t-1}
+
k_t(w_t \odot v_t)^\top
```

처음 보면 어렵지만 하나씩 보면 된다.

### 기호 설명

- `S_t`: t번째 token을 읽은 뒤의 기억장부
- `S_{t-1}`: 이전 기억장부
- `k_t`: 지금 token의 key
- `v_t`: 지금 token의 value
- `b_t`: erase gate, key 쪽에서 무엇을 지울지 정한다
- `w_t`: write gate, value 쪽에서 무엇을 쓸지 정한다
- `D_t`: decay, 오래된 기억을 조금 약하게 만드는 장치
- `I`: 아무것도 바꾸지 않는 단위행렬
- `\odot`: 같은 위치끼리 곱한다는 뜻
- `^\top`: 벡터나 행렬을 뒤집는다는 뜻

### 같은 식을 단계별로 풀기

먼저 오래된 기억을 조금 약하게 만든다.

```math
\bar S_t = D_t S_{t-1}
```

쉬운 말:

```text
오래된 기억장부를 그대로 믿지 말고 조금 흐리게 만든다.
```

그 다음, 지금 key와 관련된 오래된 내용을 읽는다.

```math
r_t = \bar S_t^\top (b_t \odot k_t)
```

쉬운 말:

```text
지금 key와 관련된 칸을 찾되,
erase gate b_t가 허락한 부분만 읽는다.
```

새로 쓸 내용도 write gate로 고른다.

```math
z_t = w_t \odot v_t
```

쉬운 말:

```text
value 전체를 다 쓰지 않고,
write gate w_t가 허락한 부분만 쓴다.
```

마지막으로 state를 고친다.

```math
S_t = \bar S_t + k_t(z_t - r_t)^\top
```

쉬운 말:

```text
기억장부에서 지금 key 위치에 있던 오래된 내용 r_t를 빼고,
새 내용 z_t를 넣는다.
```

즉 GDN-2는 단순히 "계속 더하기"가 아니라 "읽고, 지우고, 새로 쓰기"를
한다.

## 우리가 GDN-2에서 그대로 가져온 것

`GatedLinearAttention2`는 완전히 처음부터 만든 구조가 아니다. 좋은 기존
구조 위에 작은 핵심 변경을 얹은 실험이다.

GDN-2에서 그대로 가져온 것:

- recurrent state `S_t`
- query/key/value projection
- channel-wise decay `D_t`
- key-side erase gate `b_t`
- value-side write gate `w_t`
- short convolution on `q`, `k`, `v`
- GDN-2 chunkwise Triton kernel
- recurrent decoding 가능성
- Fused RMSNorm 기반 normalization
- SwiGLU/LLaMA 스타일 MLP
- 1.3B급 모델 크기
- FineWeb-Edu pretraining recipe 방향
- 4K 기본 학습 길이

한마디로:

```text
기억장부를 어떻게 만들고, 지우고, 쓰는 큰 틀은 GDN-2에서 가져왔다.
```

## 우리가 새로 넣은 것

우리가 추가한 것은 `Kaczmarz step`이다.

이름은 어렵지만 생각은 간단하다.

```text
key가 너무 크면 업데이트가 너무 세질 수 있다.
key가 너무 작으면 업데이트가 너무 약해질 수 있다.

그러니 key의 크기를 보고 업데이트 세기를 자동으로 보정하자.
```

여기서 key의 크기는 다음이다.

```math
\|k_t\|_2^2
```

뜻:

```text
key 벡터 안의 숫자들을 제곱해서 모두 더한 값
```

우리는 update strength를 이렇게 둔다.

```math
\lambda_t =
\frac{\eta_t}{\|k_t\|_2^2 + \epsilon}
```

기호 설명:

- `\lambda_t`: 이번 token에서 update를 얼마나 세게 할지 정하는 값
- `\eta_t`: 모델이 token마다 예측하는 0과 1 사이의 기본 step 크기
- `\|k_t\|_2^2`: key의 크기
- `\epsilon`: 0으로 나누는 것을 막는 아주 작은 숫자

쉬운 말:

```text
key가 크면 분모가 커져서 lambda가 작아진다.
key가 작으면 분모가 작아서 lambda가 상대적으로 커진다.
```

이렇게 하면 key 크기 때문에 update가 너무 흔들리는 문제를 줄일 수 있다.

## 우리 모델의 최종 수식

GDN-2에는 erase gate `b_t`와 write gate `w_t`가 있다.

우리는 여기에 `\lambda_t`를 곱한다.

```math
\tilde b_t = \operatorname{clip}(\lambda_t b_t, 0, 1)
```

```math
\tilde w_t = \operatorname{clip}(\lambda_t w_t, 0, 1)
```

`clip(x, 0, 1)`은 값이 0보다 작으면 0으로, 1보다 크면 1로 잘라낸다는 뜻이다.

그 다음 GDN-2 update에 `b_t`, `w_t` 대신 `\tilde b_t`, `\tilde w_t`를 넣는다.

```math
S_t
=
\left(I - k_t(\tilde b_t \odot k_t)^\top \right)D_tS_{t-1}
+
k_t(\tilde w_t \odot v_t)^\top
```

완전히 펼쳐 쓰면 다음이다.

```math
S_t
=
\left(I - k_t(\operatorname{clip}(\lambda_t b_t,0,1) \odot k_t)^\top \right)D_tS_{t-1}
+
k_t(\operatorname{clip}(\lambda_t w_t,0,1) \odot v_t)^\top
```

핵심은 이거다.

```text
GDN-2:
  어디를 지울지 b_t가 정한다.
  어디를 쓸지 w_t가 정한다.

GatedLinearAttention2:
  어디를 지울지 b_t가 정한다.
  어디를 쓸지 w_t가 정한다.
  그리고 얼마나 세게 할지 lambda_t가 key 크기를 보고 보정한다.
```

## 코드에서는 어디에 있나

모델 설정:

```text
lit_gpt/config.py
```

중요 설정:

```python
name="gdn2_kla_1.3B"
gdn2_per_layer=1
gdn2_use_qk_l2norm_in_kernel=False
gdn2_use_kaczmarz_step=True
nope=True
```

뜻:

- `gdn2_per_layer=1`: 모든 layer가 GDN-2 mixer를 쓴다
- `gdn2_use_kaczmarz_step=True`: 우리가 추가한 Kaczmarz step을 켠다
- `gdn2_use_qk_l2norm_in_kernel=False`: 기존 q/k L2 normalization kernel은 끄고, Kaczmarz step으로 update 크기를 조절한다
- `nope=True`: RoPE positional embedding을 쓰지 않는다

실제 update 코드:

```text
lit_gpt/gdn2.py
```

중요 흐름:

```python
k_norm_sq = k_heads.square().sum(dim=-1, keepdim=True).clamp_min(self.kaczmarz_eps)
eta = self.eta_proj(hidden_states).sigmoid().unsqueeze(-1)
step = eta / k_norm_sq
b = (b.float() * step).clamp(min=0.0, max=1.0)
w = (w.float() * step).clamp(min=0.0, max=1.0)
```

이 코드가 위 수식의 `\lambda_t`와 `\tilde b_t`, `\tilde w_t`에 해당한다.

## 기존 모델들과 비교

### Transformer softmax attention

Transformer attention:

```text
과거 token들의 K,V를 전부 저장한다.
query가 과거 전체를 다시 본다.
```

장점:

- 정확한 검색에 강하다.
- 과거 token을 직접 다시 볼 수 있다.

단점:

- KV cache가 context 길이에 비례해서 커진다.
- 긴 문맥 decode에서 메모리 부담이 크다.

우리 모델:

```text
과거 token별 K,V를 전부 저장하지 않는다.
고정 크기 recurrent state에 압축한다.
```

장점:

- 순수 GDN-2 layer 기준 KV cache가 context 길이에 비례해서 늘지 않는다.
- 긴 흐름에서 state를 계속 업데이트할 수 있다.

단점:

- 과거 token을 원문 그대로 다시 보는 것은 아니다.
- state 크기가 고정이라 정보 손실이 생길 수 있다.

### 단순 linear attention

단순 linear attention:

```math
S_t = S_{t-1} + k_t v_t^\top
```

문제:

```text
계속 쓰기만 하면 기억이 섞인다.
```

우리 모델:

```text
GDN-2처럼 지우기와 쓰기를 분리한다.
그리고 Kaczmarz step으로 update 세기를 조절한다.
```

### Gated DeltaNet

Gated DeltaNet은 delta rule에 gate를 넣어 update를 조절한다.

하지만 erase와 write가 같은 gate 강도에 묶이면 이런 문제가 생긴다.

```text
지우기는 약하게 하고 싶은데 쓰기는 강하게 하고 싶다.
또는 지우기는 강하게 하고 싶은데 쓰기는 약하게 하고 싶다.
```

같은 숫자 하나로 둘 다 처리하면 이런 선택이 어렵다.

### KDA

KDA는 channel-wise decay를 쓰고, memory update를 더 안정적으로 만든다.
하지만 GDN-2처럼 erase gate와 write gate를 완전히 분리하는 쪽은 아니다.

### GDN-2

GDN-2는 erase와 write를 분리한다.

```text
GDN-2 = 어디를 지울지와 어디를 쓸지를 잘 나눈 모델
```

우리 모델은 GDN-2를 기반으로 한다.

```text
GatedLinearAttention2 = GDN-2 + key-norm-normalized update strength
```

즉 GDN-2보다 완전히 다른 모델이 아니라, GDN-2의 memory edit rule을 한 단계
더 보정하는 실험이다.

### Mamba

Mamba도 recurrent state를 통해 긴 token stream을 처리한다. 그래서 넓은
의미에서는 long-context state model 계열의 장점이 있다.

하지만 Mamba는 selective state space model이고, GDN-2는 linear attention
fast-weight memory 계열이다. 둘 다 recurrent state를 쓰지만 update 방식은
다르다.

간단히:

```text
Mamba:
  state space model 쪽 전통

GatedLinearAttention2:
  linear attention / delta rule / fast-weight memory 쪽 전통
```

## 우리 모델이 "더 좋을 수 있는" 이유

긴 문맥에서 중요한 것은 세 가지다.

```text
1. 오래된 쓸모없는 정보를 줄인다.
2. 바뀐 정보를 정확히 고쳐 쓴다.
3. update가 너무 세거나 약하지 않게 한다.
```

GDN-2는 1번과 2번을 잘 하도록 설계됐다.

우리 모델은 여기에 3번을 추가한다.

예를 들어 key 벡터가 너무 큰 token이 들어오면, 일반 update는 state를 너무
세게 바꿀 수 있다. 그러면 이전에 저장한 중요한 기억을 망칠 수 있다.

반대로 key 벡터가 너무 작은 token은 state를 거의 못 바꿀 수 있다.

Kaczmarz step은 이 문제를 줄이려는 장치다.

```math
\lambda_t =
\frac{\eta_t}{\|k_t\|_2^2 + \epsilon}
```

쉬운 말:

```text
key가 큰 token은 조심해서 쓰고,
key가 작은 token은 너무 무시되지 않게 한다.
```

## 왜 long-context에 유리할 수 있나

Transformer KV cache:

```text
context가 길어질수록 저장해야 할 K,V가 계속 늘어난다.
```

GatedLinearAttention2 recurrent state:

```text
context가 길어져도 layer별 state 크기는 고정이다.
```

우리 모델의 GDN state 크기는 대략 다음과 같다.

```text
per layer state = num_heads * d_k * d_v
                = 16 * 128 * 128
                = 262,144 scalars
```

18 layer이면:

```text
18 * 262,144 = 4,718,592 scalars
```

bf16 기준 recurrent state만 보면 약 9MB 수준이다. 여기에 short convolution
cache 같은 작은 state가 추가된다.

중요한 점:

```text
이 state 크기에는 context length T가 직접 곱해지지 않는다.
```

그래서 이론적으로는 아주 긴 token stream도 state 크기를 늘리지 않고
순차 처리할 수 있다.

하지만 "모든 것을 완벽히 기억한다"는 뜻은 아니다.

고정 크기 state에 무한히 많은 정보를 넣으면 정보 손실은 생긴다. 이 모델의
목표는 무한한 원문 저장이 아니라, 필요한 상태를 잘 추적하고 업데이트하는
것이다.

## KV cache가 정말 안 늘어나나

순수 `gdn2_kla_1.3B` 구조 기준으로는 Transformer식 KV cache가 없다.

Transformer KV cache 크기:

```math
O(L \cdot T \cdot H \cdot d)
```

GDN-2 recurrent state 크기:

```math
O(L \cdot H \cdot d_k \cdot d_v)
```

기호 설명:

- `L`: layer 수
- `T`: context 길이
- `H`: head 수
- `d`: head dimension
- `d_k`: key dimension
- `d_v`: value dimension

Transformer에는 `T`가 들어간다.

GDN-2 state에는 `T`가 직접 들어가지 않는다.

이것이 long-context memory 측면의 가장 큰 차이다.

## 이 모델이 못하는 것도 분명히 있다

이 모델에 대해 과장하면 안 된다.

틀린 주장:

```text
무한 context를 완벽히 기억한다.
10B token만 학습하면 100B token GDN-2를 무조건 이긴다.
full attention보다 모든 retrieval에서 무조건 좋다.
```

정확한 주장:

```text
GatedLinearAttention2는 GDN-2의 memory edit rule에
key-norm-normalized update strength를 넣은 recurrent linear attention 실험이다.

KV cache를 context 길이에 비례해서 늘리지 않고 긴 stream을 처리할 수 있다.
하지만 과거 전체를 원문 그대로 저장하는 것이 아니라 fixed-size state에 압축한다.
```

## 왜 4K로 먼저 학습하나

GDN-2 논문 recipe의 기본 pretraining length는 4K다.

그래서 지금 10B token 실험도 먼저 4K로 한다.

이유:

```text
아키텍처 변경의 효과를 보려면 context length까지 같이 바꾸지 않는 편이 낫다.
```

먼저 4K에서:

```text
GDN-2 baseline 대비 loss와 benchmark가 어떤지 본다.
```

그 다음에:

```text
32K, 128K, 1M 같은 long-context 확장을 따로 실험한다.
```

## 1B부터 10B까지 평가하는 이유

우리는 1B, 2B, ..., 10B checkpoint를 저장한다.

이렇게 하면 다음을 볼 수 있다.

```text
학습 token이 늘수록 성능이 정말 좋아지는가?
어떤 benchmark는 빨리 좋아지고, 어떤 benchmark는 늦게 좋아지는가?
어떤 benchmark는 10B로도 부족한가?
```

예상 가능한 패턴:

- WikiText/LAMBADA perplexity는 비교적 부드럽게 좋아질 가능성이 크다.
- commonsense multiple-choice는 데이터와 scale 영향을 많이 받아 출렁일 수 있다.
- RULER retrieval은 memory update 능력과 generation 안정성 둘 다 필요하다.
- real-world retrieval은 pretraining token 수, exact answer format, 긴 입력 처리에 민감하다.

## 평가에서 볼 것

논문 기준 평가:

- WikiText perplexity
- LAMBADA perplexity
- LAMBADA accuracy
- PIQA
- HellaSwag
- WinoGrande
- ARC-Easy
- ARC-Challenge
- OpenBookQA
- Social IQA
- BoolQ
- RULER S-NIAH-1/2/3
- RULER MK-NIAH-1
- SWDE
- SQuAD completion
- FDA
- TriviaQA
- NQ Open
- DROP

비교할 기준:

```text
GDN-2 논문 recurrent GDN-2 100B token 결과
```

주의:

```text
우리 모델은 10B token만 학습한다.
그래서 100B token GDN-2보다 낮은 항목이 나와도 이상한 것이 아니다.
```

진짜 중요한 질문은 다음이다.

```text
10B token만으로 어디까지 따라가는가?
어떤 task에서는 GDN-2보다 더 좋은 memory update를 보이는가?
어떤 task에서는 데이터 부족이나 구조 한계 때문에 밀리는가?
```

## 전체 구조를 한 번에 보기

모델 한 layer의 흐름은 다음처럼 볼 수 있다.

```text
입력 hidden state x_t
  -> RMSNorm
  -> q, k, v 생성
  -> short convolution으로 가까운 token 정보 섞기
  -> decay gate g 생성
  -> erase gate b 생성
  -> write gate w 생성
  -> eta 생성
  -> key norm으로 lambda 계산
  -> b, w에 lambda 곱하기
  -> recurrent state S 업데이트
  -> query로 state 읽기
  -> output gate와 RMSNorm
  -> output projection
  -> residual connection
  -> MLP
  -> 다음 layer
```

이 구조가 18 layer 반복된다.

## 가장 짧은 정확한 설명

`GatedLinearAttention2`는 GDN-2를 기반으로 한 recurrent linear attention
모델이다. GDN-2처럼 key-side erase gate와 value-side write gate를 따로
사용해서 memory state를 고친다. 우리가 추가한 점은 Kaczmarz 방식의
key-norm-normalized step이다. 이 step은 key 크기에 따라 erase/write update
강도를 조절한다. 목적은 long-context에서 고정 크기 recurrent state를 더
안정적으로 고치고, memory interference를 줄이는 것이다.

## 한 문장으로 다시

```text
GatedLinearAttention2는 "긴 글을 전부 보관하는 모델"이 아니라,
"고정 크기 기억장부를 계속 고쳐 쓰는 모델"이고,
우리는 그 고쳐 쓰는 힘을 더 안정적으로 만들었다.
```

