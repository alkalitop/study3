# ㅇㅇ

### Attention

```py
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
```
Attention 클래스가 nn.Module의 모든 기능(파라미터 관리, GPU 이동, 상태 저장 등)을 사용 가능하게 하기 위해, 상속이 이루어짐
```py
self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
```
어텐션 스코어 계산을 위한 변환 레이어. 최종 어텐션 스코어는 이 레이어의 출력값을 기반으로 계산됨.
```py
self.v = nn.Parameter(torch.rand(hidden_dim))
```
`nn.Parameter`: PyTorch에서 학습 가능한 텐서(벡터)를 생성하는 메서드\
`torch.rand(hidden_dim)`:  `hidden_dim`을 [0, 1) 범위의 균등분포로 초기화\
`self.v`는 [0, 1) 범위의 균등분포 초기화된 크기 hidden_dim의 벡터이다.\
코드 자체는 이런 뜻이고, 모델에서의 역할은 에너지 값과의 행렬 곱을 통해 어텐션 스코어를 생성하기 위해 모든 시퀀스 위치에서 공유되는 글로벌 파라미터이다. 
