# ㅇㅇ

## Attention

### nn.Module 상속
```py
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
```
#### 기본 설명
Attention 클래스가 torch.nn.Module의 모든 기능(파라미터 관리, GPU 이동, 상태 저장 등)을 사용 가능하게 하기 위해, 상속이 이루어짐
#### 코드 추가 설명
`torch.nn.Module` 은 PyTorch의 모든 Neural Network의 base class 이다.

### 선형 변환 레이어(`self.attn`) 선언
```py
self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
```
#### 기본 설명
Energy(Attention Score) 계산을 위한 선형 변환 레이어.
#### 코드 추가 설명
1. `nn.Linear(int in_features, int out_features, bool bias=True)`
- 정의: PyTorch에서 사용되는 선형 변환(linear transformation)을 수행하는 클래스로, Fully Connected Layer라고도 불린다.
- `in_features`: 입력 텐서의 크기
- `out_features`: 출력 텐서의 크기
#### 자세한 설명
`nn.Linear`의 매개변수를 보면 출력 텐서의 크기가 입력 텐서의 크기의 절반인 것을 확인할 수 있다. 즉 이 레이어는 선형 변환을 통해 어떠한 텐서의 크기를 절반으로 줄여주는 역할을 나중에 수행하게 된다.

### 파라미터(`self.v`) 선언
```py
self.v = nn.Parameter(torch.rand(hidden_dim))
```
#### 기본 설명
Context Vector 계산을 위한 학습 가능한 벡터 (파라미터)
#### 코드 추가 설명
1. `torch.rand(int d1, d2, ...)`
- 정의: 형태(shape)가 (d1, d2, ...)인 텐서를 생성한다. 텐서의 각 값은 균등분포 [0, 1) 범위의 무작위 값으로 초기화된다.
2. `nn.Parameter(Tensor data)`
- 정의: 텐서를 받아서 같은 값을 가지는 **학습 가능한 텐서**를 반환한다.
#### 자세한 설명
`torch.rand(hidden_dim)`을 이용하여 각 hidden state vector와 같은 길이(= `hidden_dim`)를 가지는 무작위 값 벡터(차원이 하나밖에 없는 텐서이므로 벡터라고 해도 됨)를 생성한다. 이후 `nn.Parameter`를 이용하여 기존 벡터와 같은 무작위 값을 가지는 학습 가능한 벡터를 생성한다. Energy 계산 후 이 벡터를 잘 굴려서 context vector를 뽑아낼 수 있다.

### forward
#### 기본 설명
에너지 값과의 행렬 곱을 통해 어텐션 스코서
- 형태(shape): 보통 (num_layers, batch_size, hidden_dim)
- 역할: 인코더의 각 출력과 비교하여, 디코더가 현재 어느 인코더 위치에 집중할지(어텐션 분포)를 결정하는 기준이 됨
2. `encoder_outputs`
- 정의: Encoder에서 처리한 input sequence의 각 토큰 별 hidden state들을 담고 있는 텐서
- 형태(shape): (batch_size, seq_len, hidden_dim)

### 계산에 필요한 값(`batch_size`, `seq_len`) 추출
```py
batch_size = encoder_outputs.shape[0]
seq_len = encoder_outputs.shape[1]
```
#### 기본 설명
얘내는 attention 메커니즘 내에서 계산할 때 필요한 값들이다.
#### 코드 추가 설명
tensor의 `.shape` 프로퍼티는 텐서의 형태(각 차원 별 크기)을 tuple 형태로 반환한다.
#### 자세한 설명
1. `batch_size`
- 정의: 배치 크기(= `encoder_outputs`의 0번 차원)
- 의미: 한 번에 처리하는 데이터 샘플 수 (예: 32개의 문장을 동시에 처리)
2. `seq_len`
- 정의: 시퀀스 길이(= `encoder_outputs`의 1번 차원)
- 의미: 패딩(padding)이 포함된 원본 입력의 최대 길이

### Decoder의 현재 time step의 hidden state 텐서 전처리
```py
hidden = hidden.permute(1, 0, 2) 
hidden = hidden.expand(batch_size, seq_len, -1)
```
#### 기본 설명
이후의 연산 작업을 위해 텐서 `hidden`을 전처리 하는 과정이다.
#### 코드 추가 설명
1. `.permute(int i1, i2, ...)`
- 정의: 인자 순서에 맞춰서 텐서의 각 차원의 위치를 교환해준다. 행렬 transpose의 다차원 버전이라고 생각하면 편하다.
2. `.expand(int d1, d2, ...)`
- 정의: 텐서의 k번째 차원이 d_k 크기를 가지도록 확장(브로드캐스팅) 해준다 (단, d_k가 -1이면 그대로 유지). 결과적으로 텐서의 형태가 (d1, d2, ...) 이 된다.
#### 자세한 설명
1. `hidden = hidden.permute(1, 0, 2)`
- 역할: 차원 순서를 (0→1, 1→0, 2→2)로 바꾼다. 텐서 `hidden`이 `(batch_size, num_layers, hidden_dim)` 형태의 텐서가 되도록, 즉 batch 차원이 앞으로 오도록 차원 순서를 바꿔준다. 
- 이유: PyTorch의 RNN 계열(LSTM, GRU 등)에서 Decoder의 hidden state(= `hidden`)는 기본적으로 `(num_layers, batch_size, hidden_dim)` 형태로 반환되지만, Encoder의 각 출력을 담은 텐서(= `encoder_outputs`)는 `(batch_size, seq_len, hidden_dim)` 형태를 띄고 있다. 일반적으로 Attention 메커니즘에서는 이 두 텐서를 `batch_size`를 기준으로 연산을 실행하기 때문에, 연산을 정상적으로 실행하기 위해서는 두 텐서의 형태를 호환 가능하게 해주는 작업이 필요하다.
2.  `hidden = hidden.expand(batch_size, seq_len, -1)`
- 역할: 텐서 `hidden`이 `(batch_size, seq_len, hidden_dim)` 형태가 되도록 브로드캐스팅 해준다. (나머지 차원은 그대로 두고 1번째 차원의 크기만 `num_layers`에서 `seq_len`가 되도록 함)
- 이유: 두 텐서 `hidden`과 `encoder_outputs`의 연산을 정상적으로 실행하려면, 형태가 호환 가능하도록 맞춰주는 작업이 필요하다.

### Energy(Attention Score) 계산
```py
energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
```
#### 기본 설명
Energy(Attention Score)는 Attention 메커니즘에서 Decoder의 현재 hidden state와 Encoder의 각 시점의 hidden state 사이의 관련성(유사도, 중요도)을 수치로 나타낸 값이다(각 Encoder 출력마다 하나씩 계산된다). 이 값이 클수록, 해당 인코더 위치(단어)가 디코더의 현재 예측에 더 중요한 정보를 제공한다고 해석할 수 있기에, 후처리를 진행하고 나중에 가중치로 써먹는다. 
#### 계산 방법
1. dot product
- energy = hidden * encoder_outputs.transpose()
2. Additive attention (Bahdanau attention)
- energy = tanh(W * [hidden;encoder_outputs]) 
\
이외에도 여러가지 방법이 존재한다. 이 코드에서는 Additive attention 방식을 채택하였다.
#### 코드 추가 설명
1. `torch.cat(tuple tensors, int dim)`
- 정의: 텐서를 여러 개 받아서 특정 차원 기준으로 합친다.
- `tensors`: 합칠 텐서들을 tuple에 넣어서 전달
- `dim`: 합칠 때 기준이 되는 차원
- 예시: `A.shape = (30, 20, 256), B.shape = (30, 20, 256) -> torch.cat((A, B), dim=2).shape = (30, 20, 512)`
2. `torch.tanh(*args)`
- 정의: non-linear activation function 으로 이용되는 tanh 함수이다. 
- 역할: 여기서는 energy를 Additive attention(Bahdanau attention) 방식으로 계산하기 위해 사용한다.
#### 자세한 설명
Additive attention 방식으로 energy를 계산해보자. `torch.cat`으로 먼저 `hidden`과 `encoder_outputs`를 합쳐준다. 이 때 텐서의 크기가 기존 두 텐서의 2배가 되므로, `self.attn`레이어를 이용하여 선형 변환을 통해 크기를 다시 원래대로(=`hidden_dim`) 돌려놓는다(여기서 `self.attn`레이어가 `W`의 역할도 한다). 이후 레이어 반환값을 `torch.tanh` 함수에 넣어서 계산을 완료한다.

### Context Vector 계산을 위해 파라미터 전처리
```py
v = self.v.repeat(batch_size, 1).unsqueeze(2)
```
#### 기본 설명
파라미터(`self.v`)를 Energy와의 행렬곱 연산에 적용하기 위해 호환 가능한 형태로 전처리해주는 과정이다.
#### 코드 추가 설명
`.repeat(int d1, d2, ...)`
- 정의: 기존 텐서를 i1, i2, ... 방향으로 d1, d2, ... 만큼 반복하여 차원을 생성 \& 확장 한다.
`.unsqueeze(int dim)`
- 정의: 크기가 1인 차원을 기존 텐서에 dim번째 차원에 insert한다.
- 예시: 텐서의 형태(shape)가 (x, z) 일 때 `.unsqueeze(1)` -> (x, 1, z)
#### 자세한 설명
`self.v`의 초기 형태(shape)는 `(hidden_dim,)` 이다. 이 때 `.repeat(batch_size, 1)`을 실행하면 파라미터 형태가 `(batch_size, hidden_dim)` 이 된다. 여기서 `.unsqueeze(2)`를 실행하여 크기가 1인 차원을 2번째에 insert하게 되므로, 최종적으로 파라미터의 형태는 `(batch_size, hidden_dim, 1)`이 된다.

### Context Vector 계산
```py
attention_weights = torch.bmm(energy, v).squeeze(2) 
return torch.softmax(attention_weights, dim=1)
```
#### 기본 설명
context vector를 계산하는 과정이다.
#### 코드 추가 설명
1. `torch.bmm(Tensor t1, Tensor t2)`
- 정의: batch matrix-matrix product를 실행한다. 두 텐서 t1, t2의 shape가 각각 (batch_size, n, m), (batch_size, m, p) 일 때, 뒤의 두 차원끼리 행렬곱을 실행하여 최종적으로 shape가 (batch_size, n, p)인 텐서를 반환한다.
2. `.squeeze(int dim)`
- 정의: dim번째 차원의 크기가 1이라면, 해당 차원을 삭제한다.
3. `torch.softmax(Tensor data, int dim)`
- 정의: 주어진 텐서의 dim번째 차원을 기준으로 각 softmax 값을 계산하여 반환한다.
#### 자세한 설명
`v`는 단순히 최종 텐서 형태 조정을 도와주는 파라미터라는 것을 기억하고 가자. `torch.bmm(energy, v)` 을 실행하면 형태가 `(batch_size, seq_len, 1)` 인 텐서가 반환되고, 2번째 차원이 1이므로 `.squeeze(1)`을 실행하여 최종적으로 `(batch_size, seq_len)` 형태의 텐서인 `attention_weights`를 얻을 수 있다. context vector는 `attention_weight`의 1번째 차원을 기준으로 각 값에 softmax를 취한 벡터이다.

## Decoder
