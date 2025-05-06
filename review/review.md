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

### `self.attn` (선형 변환 레이어)
```py
self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
```
#### 기본 설명
어텐션 스코어 계산을 위한 변환 레이어.
#### 코드 추가 설명
1. `nn.Linear(int in_features, int out_features, bool bias=True)`
- 정의: PyTorch에서 사용되는 선형 변환(linear transformation)을 수행하는 클래스로, Fully Connected Layer라고도 불린다.
- `in_features`: 입력 텐서의 크기
- `out_features`: 출력 텐서의 크기
#### 자세한 설명
`nn.Linear`의 파라미터를 보면 출력 텐서의 크기가 입력 텐서의 크기의 절반인 것을 확인할 수 있다. 즉 이 레이어는 선형 변환을 통해 어떠한 텐서의 크기를 절반으로 줄여주는 역할을 나중에 수행하게 된다.

### v???
```py
        self.v = nn.Parameter(torch.rand(hidden_dim))
```
#### 기본 설명
에너지 값과의 행렬 곱을 통해 어텐션 스코서
- 형태(shape): 보통 (num_layers, batch_size, hidden_dim)
- 역할: 인코더의 각 출력과 비교하여, 디코더가 현재 어느 인코더 위치에 집중할지(어텐션 분포)를 결정하는 기준이 됨
2. `encoder_outputs`
- 정의: Encoder에서 처리한 input sequence의 각 토큰 별 hidden state들을 담고 있는 텐서
- 형태(shape): (batch_size, seq_len, hidden_dim)

### `batch_size`, `seq_len` 추출
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

### `hidden` 전처리
```py
hidden = hidden.permute(1, 0, 2) 
hidden = hidden.expand(batch_size, seq_len, -1)
```
#### 기본 설명
이후의 연산 작업을 위해 텐서 `hidden`을 전처리 하는 과정이다.
#### 코드 추가 설명
1. `.permute(i1, i2, ...)`
- 정의: 인자 순서에 맞춰서 텐서의 각 차원의 위치를 교환해준다. 행렬 transpose의 다차원 버전이라고 생각하면 편하다.
2. `.expand(d1, d2, ...)`
- 정의: 텐서의 k번째 차원이 d_k 크기를 가지도록 확장(브로드캐스팅) 해준다 (단, d_k가 -1이면 그대로 유지). 결과적으로 텐서의 형태가 (d1, d2, ...) 이 된다.
#### 자세한 설명
1. `hidden = hidden.permute(1, 0, 2)`
- 역할: 차원 순서를 (0→1, 1→0, 2→2)로 바꾼다. 텐서 `hidden`이 `(batch_size, num_layers, hidden_dim)` 형태의 텐서가 되도록, 즉 batch 차원이 앞으로 오도록 차원 순서를 바꿔준다. 
- 이유: PyTorch의 RNN 계열(LSTM, GRU 등)에서 Decoder의 hidden state(= `hidden`)는 기본적으로 `(num_layers, batch_size, hidden_dim)` 형태로 반환되지만, Encoder의 각 출력을 담은 텐서(= `encoder_outputs`)는 `(batch_size, seq_len, hidden_dim)` 형태를 띄고 있다. 일반적으로 Attention 메커니즘에서는 이 두 텐서를 `batch_size`를 기준으로 연산을 실행하기 때문에, 연산을 정상적으로 실행하기 위해서는 두 텐서의 형태를 호환 가능하게 해주는 작업이 필요하다.
2.  `hidden = hidden.expand(batch_size, seq_len, -1)`
- 역할: 텐서 `hidden`이 `(batch_size, seq_len, hidden_dim)` 형태가 되도록 브로드캐스팅 해준다. (나머지 차원은 그대로 두고 1번째 차원의 크기만 `num_layers`에서 `seq_len`가 되도록 함)
- 이유: 두 텐서 `hidden`과 `encoder_outputs`의 연산을 정상적으로 실행하려면, 형태가 호환 가능하도록 맞춰주는 작업이 필요하다.

### energy
```py
energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
```
#### 코드 추가 설명
1. `torch.cat(tuple tensors, int dim)`
- 정의: 텐서를 여러 개 받아서 특정 차원 기준으로 합친다.
- `tensors`: 합칠 텐서들을 tuple에 넣어서 전달
- `dim`: 합칠 때 기준이 되는 차원
- 예시: `A.shape = (30, 20, 256), B.shape = (30, 20, 256) -> torch.cat((A, B), dim=2).shape = (30, 20, 512)`
#### 자세한 설명
`torch.cat`으로 먼저 `hidden`과 `encoder_outputs`를 합쳐준다. 이 때 텐서의 크기가 기존 두 텐서의 2배가 되므로, `self.attn`레이어를 이용하여 선형 변환을 통해 크기를 다시 원래대로(=`hidden_dim`) 돌려놓는다. 이후 출력값들을 [-1, 1] 범위로 정규화 하기 위해 activation function으로서 `torch.tanh`을 이용한다.


`source`: encoder에 입력되는 원본 sequence (예: 번역할 원문)
`target`: decoder가 학습?할 target sequence (예: 번역 결과)
`.size(d)`는 d번째(d >= 0) 차원의 시퀀스 길이를 반환하는 함수이다.\
`
