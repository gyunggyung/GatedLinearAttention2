# GatedLinearAttention2 초심자 설명서

이 문서는 인공지능 모델 구조를 처음 보는 사람도 `GatedLinearAttention2`가
어떤 모델인지 이해할 수 있도록 쓰는 설명서다. 수학을 많이 몰라도 읽을 수
있게 쉬운 말로 먼저 설명하고, 필요한 곳에서만 수식을 함께 적는다.

용어는 쉽게 풀어 쓰되 정확하게 사용한다. `token`, `query`, `key`,
`value`, `state`, `KV cache`, `linear attention`, `GDN-2` 같은 말은 뒤에서
하나씩 설명한다.

## 모델의 핵심

`GatedLinearAttention2`는 Transformer의 일반적인 softmax attention처럼
과거 token의 `K, V`를 전부 저장해서 다시 보는 모델이 아니다.

대신 과거 내용을 하나의 작은 "기억장부"인 recurrent state에 계속
압축해서 넣고, 새 token이 들어올 때마다 그 장부를 조금씩 고치는
linear attention 계열 모델이다.

이 모델의 핵심 아이디어는 다음이다.

```text
GDN-2의 좋은 점:
  무엇을 지울지와 무엇을 쓸지를 따로 정한다.

이 프로젝트에서 추가한 점:
  얼마나 세게 지우고 쓸지를 key의 크기에 맞춰 자동 조절한다.
```

## 모델의 종류

```text
GatedLinearAttention2는 리니어 어텐션 계열 모델이다.
Q, K, V를 모두 쓴다.
다만 Transformer softmax attention처럼 과거 token별 K,V를 전부 저장하지 않는다.
GDN-2처럼 고정 크기 recurrent state에 과거 정보를 압축하고 계속 고쳐 쓴다.
```

이 모델은 attention을 버린 구조가 아니다. token마다 `query`, `key`,
`value`를 만들고, query로 memory를 읽는다.

다른 점은 과거를 저장하는 방식이다.

```text
Transformer:
  과거 token마다 K,V를 저장한다.
  지금 Q가 과거 K 전체를 다시 본다.
  그래서 KV cache가 context 길이에 따라 커진다.

GatedLinearAttention2:
  token마다 Q,K,V를 만든다.
  K,V를 token별 cache로 계속 보관하지 않는다.
  K,V로 recurrent state S_t를 업데이트한다.
  지금 Q는 그 state를 읽는다.
  그래서 순수 recurrent layer 기준 KV cache가 context 길이에 비례해서 늘지 않는다.
```

수식으로 아주 짧게 쓰면 다음이다.

```math
q_t = W_q x_t
```

```math
k_t = W_k x_t
```

```math
v_t = W_v x_t
```

Transformer softmax attention은 과거의 `k_1, k_2, ..., k_t`와
`v_1, v_2, ..., v_t`를 저장한 뒤 다시 본다.

GatedLinearAttention2는 `k_t, v_t`로 state를 고친다.

```math
S_t = update(S_{t-1}, k_t, v_t)
```

그리고 query로 state를 읽는다.

```math
o_t = read(S_t, q_t)
```

이 구조를 가장 짧게 표현하면 다음과 같다.

```text
QKV를 쓰는 recurrent linear attention 모델
```

## 필요한 배경

긴 문맥에서 어려운 문제는 단순히 "많이 저장하기"가 아니다.

정말 어려운 것은 상태 추적이다.

예를 들어 문맥이 이렇게 이어진다고 하자.

```text
처음 정보: Alice는 Seoul에 있다.
업데이트: Alice는 Busan으로 이동했다.
확인할 내용: Alice는 지금 어디에 있는가?
```

좋은 모델은 `Seoul`을 그대로 붙잡고 있으면 안 된다. `Alice의 위치`라는
상태를 찾아서 `Busan`으로 고쳐야 한다.

Transformer softmax attention은 과거 원문 검색에는 강하다. 하지만 긴
decode에서는 과거 token별 KV cache를 계속 들고 있어야 하고, 상태를 하나의
memory cell처럼 직접 고쳐 쓰는 구조는 아니다.

단순 linear attention은 KV cache 문제는 줄인다. 그러나 그냥 계속 더하면
오래된 상태와 새 상태가 섞인다.

```text
Transformer:
  잘 찾지만 길어질수록 KV cache가 커진다.

단순 linear attention:
  KV cache는 줄지만, 오래된 정보와 새 정보가 섞일 수 있다.

GDN-2:
  linear attention state를 쓰면서 지우기와 쓰기를 분리한다.

GatedLinearAttention2:
  GDN-2의 지우기/쓰기 분리에 key 크기 기반 update strength 보정을 추가한다.
```

따라서 GatedLinearAttention2의 목적은 분명하다.

```text
리니어 어텐션의 긴 문맥 효율을 유지하면서,
기존 리니어 어텐션의 상태 추적 약점을 GDN-2 방식으로 줄이고,
GDN-2 update가 key 크기에 따라 흔들릴 수 있는 부분을 한 번 더 보정한다.
```

## 기본 용어

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
query = 어떤 이름표의 서랍을 찾을지 나타내는 요청
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
o_t = \sum_{i \le t} softmax_i(q_t^\top k_i) v_i
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

## Linear Attention의 기본 아이디어

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

### Linear라는 말의 뜻

여기서 `linear`는 "똑똑한 정도가 선형이다"라는 뜻이 아니다.

계산량과 저장량이 context 길이 `T`에 대해 어떻게 늘어나는지를 말한다.

Transformer softmax attention은 새 token이 들어올 때 과거 token 전체와
비교한다. 전체 문맥을 한 번에 학습할 때는 token 쌍을 많이 비교해야 해서
attention 부분의 계산이 대략 `T^2`에 가깝게 커진다.

반면 단순 linear attention은 token을 하나 읽을 때 state를 한 번 고친다.

```text
token 1개 입력 -> state 1번 업데이트
token 1개 입력 -> state 1번 업데이트
token 1개 입력 -> state 1번 업데이트
```

그래서 token 수가 2배가 되면 attention state update도 대략 2배가 된다.
이런 의미에서 linear attention이라고 부른다.

### Linear Attention도 Attention인 이유

Linear attention도 attention이다. 다만 Transformer softmax attention과
저장 방식이 다르다.

공통점:

- `query`, `key`, `value`를 만든다.
- `query`로 과거 정보에서 필요한 내용을 읽는다.
- `key`와 `value`로 과거 정보를 저장한다.

차이점:

- Transformer는 과거 token별 `K,V`를 그대로 보관한다.
- linear attention은 과거 token별 `K,V`를 하나의 state에 누적한다.

따라서 GatedLinearAttention2는 attention이 아닌 완전히 다른 구조가 아니다.

정확히는 다음에 가깝다.

```text
softmax attention을 쓰는 Transformer가 아니라,
recurrent state를 쓰는 linear attention / fast-weight memory 모델이다.
```

### 단순 Linear Attention의 약점

단순 linear attention의 가장 큰 약점은 지우기 어렵다는 점이다.

예를 들어 문맥이 이렇게 바뀐다고 하자.

```text
Alice is in Seoul.
Alice moved to Busan.
Where is Alice?
```

정답은 `Busan`이어야 한다.

단순히 계속 더하는 state update만 있으면 `Seoul` 정보와 `Busan` 정보가
같은 state 안에 섞일 수 있다. 그러면 모델이 오래된 상태와 새 상태를
헷갈릴 수 있다.

그래서 필요한 것은 단순한 누적이 아니라 memory edit이다.

```text
1. 예전 내용을 찾는다.
2. 필요하면 지운다.
3. 새 내용을 쓴다.
4. 쓰는 세기가 너무 크거나 작지 않게 조절한다.
```

GDN-2와 GatedLinearAttention2는 바로 이 문제를 해결하려는 linear
attention 계열이다.

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

## Gate의 의미

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

## GDN-2에서 가져온 구성

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

## 새로 넣은 구성

이 프로젝트에서 추가한 것은 `Kaczmarz step`이다.

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

update strength는 이렇게 둔다.

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

## GatedLinearAttention2의 최종 수식

GDN-2에는 erase gate `b_t`와 write gate `w_t`가 있다.

GatedLinearAttention2는 여기에 `\lambda_t`를 곱한다.

```math
\tilde b_t = clip(\lambda_t b_t, 0, 1)
```

```math
\tilde w_t = clip(\lambda_t w_t, 0, 1)
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
\left(I - k_t(clip(\lambda_t b_t,0,1) \odot k_t)^\top \right)D_tS_{t-1}
+
k_t(clip(\lambda_t w_t,0,1) \odot v_t)^\top
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
- `gdn2_use_kaczmarz_step=True`: 이 프로젝트에서 추가한 Kaczmarz step을 켠다
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

먼저 한 줄로 구분하면 다음과 같다.

| 방식 | 과거를 저장하는 방법 | 강한 점 | 약한 점 |
| --- | --- | --- | --- |
| Transformer softmax attention | token별 `K,V`를 KV cache에 저장 | 과거 원문 위치를 직접 다시 보는 검색 | context가 길수록 KV cache가 커짐 |
| 단순 linear attention | 고정 크기 state `S_t`에 계속 더함 | KV cache가 `T`에 비례해서 늘지 않음 | 오래된 기억과 새 기억이 섞이기 쉬움 |
| Gated DeltaNet | delta rule에 gate를 넣어 state를 고침 | 단순 누적보다 state tracking이 좋음 | 지우기와 쓰기 제어가 충분히 분리되지 않을 수 있음 |
| KDA | decay와 kernelized delta update를 조합 | memory update 안정성이 좋음 | GDN-2식 erase/write 분리와는 다름 |
| GDN-2 | erase gate와 write gate를 따로 둠 | 지울 내용과 쓸 내용을 분리해서 제어 | key 크기에 따른 update 세기 흔들림은 별도 보정이 약함 |
| GatedLinearAttention2 | GDN-2 state에 key-norm step을 추가 | erase/write 방향과 update 세기를 함께 제어 | 아직 실험 모델이라 benchmark로 확인해야 함 |
| Transformer+GDN 하이브리드 | 일부 layer는 attention, 일부 layer는 recurrent state | 검색과 state tracking을 함께 노림 | 이번 실험의 차별점이 아니며 구조가 더 복잡함 |

### Transformer softmax attention

Transformer attention은 "과거를 다시 펼쳐서 보는 방식"이다.

```text
과거 token들의 K,V를 전부 저장한다.
새 query가 과거 key 전체와 비교된다.
관련 높은 value를 골라 섞는다.
```

장점은 명확하다.

- 특정 문장을 원문 그대로 다시 찾는 retrieval에 강하다.
- 긴 문서 안에서 어느 위치가 중요한지 직접 고를 수 있다.
- 이미 검증된 학습 안정성과 scaling recipe가 많다.

하지만 decode에서는 KV cache가 context 길이에 비례해서 커진다.

```text
context 4K  -> 4K token의 K,V 저장
context 32K -> 32K token의 K,V 저장
context 1M  -> 1M token의 K,V 저장
```

그래서 긴 stream을 계속 읽는 상황에서는 메모리와 bandwidth 부담이 커진다.

### 단순 linear attention

단순 linear attention은 "과거를 다시 펼치지 않고 장부 하나에 누적하는
방식"이다.

```math
S_t = S_{t-1} + k_t v_t^\top
```

장점:

- token별 KV cache를 저장하지 않아도 된다.
- recurrent state 크기가 context 길이 `T`에 직접 비례하지 않는다.
- 아주 긴 stream을 순차 처리하는 구조로 만들기 좋다.

문제:

```text
계속 쓰기만 하면 기억이 섞인다.
```

이 문제 때문에 단순 linear attention만으로는 state tracking이 약할 수
있다. 문맥에서 상태가 바뀌면 예전 상태를 지우고 새 상태로 고쳐야 하는데,
그냥 `+ k_t v_t^T`만 하면 "고쳐 쓰기"가 아니라 "덧쓰기"에 가깝다.

GatedLinearAttention2는 여기서 출발한다.

```text
단순 linear attention의 고정 크기 state 장점은 유지한다.
하지만 update는 GDN-2처럼 지우기와 쓰기로 나눈다.
그리고 update 세기는 key norm으로 다시 보정한다.
```

즉 이 프로젝트는 "리니어 어텐션을 개선하는 실험"이다. 정확히는 softmax
attention을 고친 것이 아니라, recurrent linear attention의 memory update
rule을 개선하는 실험이다.

### Gated DeltaNet

Gated DeltaNet은 delta rule에 gate를 넣어 update를 조절한다.

delta rule의 직관은 다음이다.

```text
이미 저장된 값을 읽는다.
새로 원하는 값과 비교한다.
차이만큼 state를 고친다.
```

이것은 단순 linear attention보다 낫다. 하지만 지우기와 쓰기가 같은 gate
강도에 많이 묶이면 이런 선택이 어렵다.

```text
지우기는 약하게 하고 싶은데 쓰기는 강하게 하고 싶다.
지우기는 강하게 하고 싶은데 쓰기는 약하게 하고 싶다.
key 쪽 선택과 value 쪽 선택을 다르게 하고 싶다.
```

GDN-2는 이 점을 더 직접적으로 분리한다.

### KDA

KDA는 channel-wise decay를 쓰고, kernelized delta update를 더 안정적으로
만드는 계열이다.

쉽게 말하면 "기억을 조금씩 흐리게 만드는 장치"와 "state를 고쳐 쓰는
장치"를 함께 쓴다.

GatedLinearAttention2도 decay를 쓴다. 이 부분은 GDN-2 계열의 장점을
그대로 따른다. 하지만 중심 차별점은 KDA 자체가 아니라 다음 두 가지다.

```text
GDN-2식 erase/write gate 분리
key norm으로 update strength를 보정하는 Kaczmarz step
```

### GDN-2

GDN-2는 erase와 write를 분리한다.

```text
GDN-2 = 어디를 지울지와 어디를 쓸지를 따로 정하는 모델
```

GDN-2의 핵심은 다음이다.

- `b_t`: key 쪽에서 무엇을 지울지 정한다.
- `w_t`: value 쪽에서 무엇을 쓸지 정한다.
- `D_t`: 오래된 기억을 어느 정도 decay할지 정한다.
- `S_t`: 과거 token별 KV cache 대신 들고 가는 recurrent memory다.

GatedLinearAttention2는 GDN-2를 기반으로 한다.

```text
GatedLinearAttention2 = GDN-2 + key-norm-normalized update strength
```

즉 GDN-2와 완전히 다른 모델이 아니다. GDN-2의 memory edit rule에
"이번 key의 크기에 비해 update가 너무 세거나 약하지 않은가"를 보는
추가 보정축을 넣은 것이다.

### GatedLinearAttention2

GatedLinearAttention2가 실제로 하는 일은 다음처럼 요약된다.

```text
1. linear attention처럼 고정 크기 state S_t를 쓴다.
2. GDN-2처럼 erase gate b_t와 write gate w_t를 따로 만든다.
3. key norm ||k_t||^2을 계산한다.
4. eta_t / (||k_t||^2 + epsilon)으로 lambda_t를 만든다.
5. b_t와 w_t에 lambda_t를 곱해 update 세기를 보정한다.
6. 보정된 gate로 recurrent state를 고쳐 쓴다.
```

따라서 GatedLinearAttention2는 다음 가설을 검증하려는 구조다.

```text
linear attention의 긴 문맥 효율은 유지하면서,
GDN-2의 state edit 능력을 가져오고,
key 크기 때문에 update가 흔들리는 문제를 줄일 수 있는가?
```

### Mamba

Mamba도 recurrent state를 통해 긴 token stream을 처리한다. 그래서 넓은
의미에서는 long-context state model 계열의 장점이 있다.

하지만 Mamba는 selective state space model이고, GDN-2와 GatedLinearAttention2는
linear attention / delta rule / fast-weight memory 계열이다.

간단히:

```text
Mamba:
  state space model 쪽 전통
  sequence dynamics를 state transition으로 다룬다.

GatedLinearAttention2:
  linear attention / delta rule / fast-weight memory 쪽 전통
  key-value memory를 고정 크기 state로 고쳐 쓴다.
```

둘 다 KV cache 증가 문제를 줄이는 방향이지만, 내부 수학과 inductive bias가
다르다.

### Transformer+GDN 하이브리드

하이브리드는 Transformer attention layer와 GDN/Mamba 같은 recurrent layer를
섞는 방식이다.

장점:

- Transformer layer가 과거 원문 retrieval을 잘한다.
- recurrent layer가 상태 추적과 긴 stream 처리에 도움을 준다.
- 대규모 모델에서는 둘의 장점을 같이 쓰기 쉽다.

단점:

- 어떤 layer를 어떤 비율로 섞을지 설계 변수가 늘어난다.
- KV cache가 완전히 사라지는 것은 아니다.
- 이번 프로젝트처럼 "리니어 어텐션 update rule 하나의 효과"를 보려는
  실험에서는 원인 분석이 흐려진다.

그래서 이번 실험에서는 하이브리드를 쓰지 않는다.

```text
이번 실험의 목적:
  GDN-2 기반 recurrent linear attention 자체를 바꿨을 때
  10B token 학습에서 어떤 성능이 나오는지 확인한다.
```

## GatedLinearAttention2가 더 좋을 수 있는 이유

긴 문맥에서 중요한 것은 세 가지다.

```text
1. 오래된 쓸모없는 정보를 줄인다.
2. 바뀐 정보를 정확히 고쳐 쓴다.
3. update가 너무 세거나 약하지 않게 한다.
```

GDN-2는 1번과 2번을 잘 하도록 설계됐다.

GatedLinearAttention2는 여기에 3번을 추가한다.

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

### GDN-2보다 더 나을 수 있는 이론적 근거

GDN-2의 update는 매우 강력하지만, update 크기는 `k_t`의 크기에 영향을
받는다.

직관적으로 보면 memory edit에는 이런 위험이 있다.

```text
key norm이 큰 token:
  같은 gate 값이어도 state를 너무 크게 바꿀 수 있다.
  잘못하면 중요한 과거 memory를 과하게 지운다.

key norm이 작은 token:
  같은 gate 값이어도 state를 충분히 못 바꿀 수 있다.
  잘못하면 새 정보를 state에 충분히 반영하지 못한다.
```

GatedLinearAttention2는 이 부분을 다음처럼 보정한다.

```math
effective_step_t =
\lambda_t \|k_t\|_2^2
```

```math
\lambda_t =
\frac{\eta_t}{\|k_t\|_2^2 + \epsilon}
```

그러면 `epsilon`이 아주 작고 `\|k_t\|_2^2`가 충분히 클 때 대략 다음처럼
된다.

```math
effective_step_t \approx \eta_t
```

쉬운 말:

```text
key가 얼마나 크든,
실제 state update 세기는 eta_t 근처로 정규화된다.
```

이것이 GDN-2보다 성능적으로 더 나을 수 있는 핵심 가설이다.

```text
GDN-2:
  무엇을 지울지와 무엇을 쓸지는 잘 나눈다.

GatedLinearAttention2:
  무엇을 지울지와 무엇을 쓸지도 나누고,
  그 update가 key norm 때문에 너무 세거나 약해지는 것도 줄인다.
```

상태 추적 task에서는 이 차이가 중요할 수 있다.

```text
Alice = Seoul
Alice = Busan
Alice = Jeju
```

이런 흐름에서는 state를 여러 번 고쳐야 한다. update가 한 번이라도 너무
강하면 이전에 필요한 다른 정보를 망칠 수 있고, 너무 약하면 최신 상태를
못 따라갈 수 있다. 그래서 update strength를 안정화하는 것은 long-context
state tracking에 직접 연결된다.

다만 이것은 "이론적으로 더 나을 수 있다"는 주장이다. 실제로 GDN-2보다
좋은지는 같은 데이터, 같은 token budget, 같은 평가에서 확인해야 한다.
그래서 이 프로젝트는 1B부터 10B checkpoint까지 저장하고 GDN-2 논문
benchmark로 비교한다.

## 왜 long-context에 유리할 수 있나

Transformer KV cache:

```text
context가 길어질수록 저장해야 할 K,V가 계속 늘어난다.
```

GatedLinearAttention2 recurrent state:

```text
context가 길어져도 layer별 state 크기는 고정이다.
```

GatedLinearAttention2의 GDN state 크기는 대략 다음과 같다.

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

학습 과정에서는 1B, 2B, ..., 10B checkpoint를 저장한다.

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
GatedLinearAttention2는 10B token만 학습한다.
그래서 100B token GDN-2보다 낮은 항목이 나와도 이상한 것이 아니다.
```

평가에서 중요한 관찰점은 다음이다.

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
사용해서 memory state를 고친다. 이 프로젝트에서 추가한 점은 Kaczmarz 방식의
key-norm-normalized step이다. 이 step은 key 크기에 따라 erase/write update
강도를 조절한다. 목적은 long-context에서 고정 크기 recurrent state를 더
안정적으로 고치고, memory interference를 줄이는 것이다.

## 한 문장으로 다시

```text
GatedLinearAttention2는 "긴 글을 전부 보관하는 모델"이 아니라,
"고정 크기 기억장부를 계속 고쳐 쓰는 모델"이고,
이 프로젝트는 그 고쳐 쓰는 힘을 더 안정적으로 만들었다.
```
