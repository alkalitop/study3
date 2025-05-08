import torch
import torch.nn as nn
from tqdm import tqdm

# 간단한 번역 데이터
data = [
    # 인사
    ("hello", "안녕하세요"),
    ("good morning", "좋은 아침입니다"),
    ("good afternoon", "좋은 오후입니다"),
    ("good evening", "좋은 저녁입니다"),
    ("good night", "좋은 밤입니다"),
    
    # 감사
    ("thank you", "감사합니다"),
    ("thank you very much", "매우 감사합니다"),
    ("thanks a lot", "정말 감사합니다"),
    
    # 안부
    ("how are you", "어떻게 지내세요"),
    ("how have you been", "어떻게 지내셨나요"),
    ("i am fine", "저는 잘 지내요"),
    ("i am doing well", "저는 잘 지내고 있어요"),
    
    # 작별
    ("goodbye", "안녕히 가세요"),
    ("see you later", "나중에 봐요"),
    ("see you tomorrow", "내일 봐요"),
    ("take care", "잘 지내세요"),
    
    # 질문
    ("what is your name", "이름이 뭐예요"),
    ("my name is john", "제 이름은 존이에요"),
    ("where are you from", "어디서 오셨나요"),
    ("i am from korea", "저는 한국에서 왔어요"),
    
    # 일상
    ("i like coffee", "저는 커피를 좋아해요"),
    ("do you like tea", "차를 좋아하세요"),
    ("i want to eat", "저는 먹고 싶어요"),
    ("i am hungry", "저는 배고파요"),
    
    # 감정
    ("i am happy", "저는 행복해요"),
    ("i am sad", "저는 슬퍼요"),
    ("i am tired", "저는 피곤해요"),
    ("i am excited", "저는 신나요"),
    
    # 시간
    ("what time is it", "몇 시예요"),
    ("it is three o clock", "3시예요"),
    ("i will be late", "저는 늦을 것 같아요"),
    ("i am on time", "저는 정시에 왔어요"),
    
    # 날씨
    ("it is sunny", "날씨가 맑아요"),
    ("it is raining", "비가 와요"),
    ("it is cold", "추워요"),
    ("it is hot", "더워요")
]

import torch
import torch.nn as nn
from tqdm import tqdm

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
    
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
        
        hidden = hidden.permute(1, 0, 2) 
        hidden = hidden.expand(batch_size, seq_len, -1) 
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) 
        
        v = self.v.repeat(batch_size, 1).unsqueeze(2) 
        attention_weights = torch.bmm(energy, v).squeeze(2) 
        
        return torch.softmax(attention_weights, dim=1)

class Encoder(nn.Module):
    # 얜 별로 다를거 없음
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 10, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        return output, hidden, cell 


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 10, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim + embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size + 10)
        self.attention = Attention(hidden_dim) # Attention
        
    def forward(self, x, hidden, cell, encoder_outputs):
        embedded = self.embedding(x) 
        
        attn_weights = self.attention(hidden, encoder_outputs)  
        attn_weights = attn_weights.unsqueeze(1) 
        context = torch.bmm(attn_weights, encoder_outputs)
        
        lstm_input = torch.cat((embedded, context), dim=2) 
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell)) 
        prediction = self.fc(output.squeeze(1)) 
        
        return prediction, hidden, cell



class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim)
        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.size(0)
        target_len = target.size(1)
        vocab_size = self.decoder.fc.out_features
        
        encoder_outputs, hidden, cell = self.encoder(source) 
        decoder_input = target[:, 0].unsqueeze(1)
        outputs = torch.zeros(batch_size, target_len, vocab_size)
        
        for t in range(1, target_len): 
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            decoder_input = target[:, t].unsqueeze(1) if teacher_force else output.argmax(1).unsqueeze(1)
        
        return outputs



def create_vocab(data):
    """단어 사전 생성"""
    input_words = set()
    output_words = set()
    
    for src, tgt in data:
        input_words.update(src.split())
        output_words.update(tgt.split())
    
    # 특수 토큰 추가 - 보통의 단어와 구분해서 처리
    special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
    input_vocab = {
        '<pad>': 0,
        '<unk>': 1,
        '<sos>': 2,
        '<eos>': 3
    }
    output_vocab = {
        '<pad>': 0,
        '<unk>': 1,
        '<sos>': 2,
        '<eos>': 3
    }
    
    # 일반 단어 추가
    for i, word in enumerate(sorted(list(input_words))):
        input_vocab[word] = i + len(special_tokens)
        
    for i, word in enumerate(sorted(list(output_words))):
        output_vocab[word] = i + len(special_tokens)
    
    return input_vocab, output_vocab

def prepare_data(data, input_vocab, output_vocab):
    """데이터를 텐서로 변환"""
    sources = []
    targets = []
    
    # 가장 긴 시퀀스의 길이 찾기
    max_src_len = max(len(src.split()) for src, _ in data) + 2  # +2는 <sos>와 <eos> 토큰
    max_tgt_len = max(len(tgt.split()) for _, tgt in data) + 2
    
    for src, tgt in data:
        # 입력 시퀀스 준비
        src_tokens = ['<sos>'] + src.split() + ['<eos>']
        src_indices = [input_vocab.get(token, input_vocab['<unk>']) for token in src_tokens]
        src_indices += [input_vocab['<pad>']] * (max_src_len - len(src_indices))
        
        # 출력 시퀀스 준비
        tgt_tokens = ['<sos>'] + tgt.split() + ['<eos>']
        tgt_indices = [output_vocab.get(token, output_vocab['<unk>']) for token in tgt_tokens]
        tgt_indices += [output_vocab['<pad>']] * (max_tgt_len - len(tgt_indices))
        
        sources.append(src_indices)
        targets.append(tgt_indices)
    
    # 텐서로 변환 전에 각 인덱스가 범위 내에 있는지 확인
    source_tensor = torch.tensor(sources)
    target_tensor = torch.tensor(targets)
    
    # 디버깅 정보 출력
    print(f"입력 시퀀스 최대 인덱스: {source_tensor.max().item()}")
    print(f"출력 시퀀스 최대 인덱스: {target_tensor.max().item()}")
    
    return source_tensor, target_tensor

def train_model(model, source, target, input_vocab, output_vocab, epochs=1000):
    """모델 학습"""
    criterion = nn.CrossEntropyLoss(ignore_index=output_vocab['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("=== 모델 학습 시작 ===")
    for epoch in tqdm(range(epochs), desc="학습 진행 중"):
        optimizer.zero_grad()
        output = model(source, target)
        
        loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"에포크 {epoch+1}, 손실: {loss.item():.4f}")

def translate(model, sentence, input_vocab, output_vocab):
    """문장 번역"""
    model.eval()
    with torch.no_grad():
        # 역사전 생성 (인덱스 → 단어)
        idx_to_word = {idx: word for word, idx in output_vocab.items()}
        
        # 입력 준비
        tokens = ['<sos>'] + sentence.split() + ['<eos>']
        indices = [input_vocab.get(token, input_vocab['<unk>']) for token in tokens]
        source = torch.tensor([indices])
        
        # 인코더
        encoder_outputs, hidden, cell = model.encoder(source)
        
        # 디코더
        decoder_input = torch.tensor([[output_vocab['<sos>']]])
        translated_words = []
        
        for _ in range(20):  # 최대 20단어까지 생성
            output, hidden, cell = model.decoder(decoder_input, hidden, cell, encoder_outputs)
            predicted_idx = output.argmax(1).item()
            
            if predicted_idx == output_vocab['<eos>']:
                break
                
            # 인덱스가 어휘 사전에 있는지 확인
            if predicted_idx in idx_to_word:
                word = idx_to_word[predicted_idx]
                if word not in ['<pad>', '<unk>', '<sos>', '<eos>']:
                    translated_words.append(word)
            
            # 다음 토큰으로 predicted_idx 사용
            # 만약 범위를 벗어나면 <unk> 토큰 사용
            if predicted_idx >= len(model.decoder.embedding.weight):
                decoder_input = torch.tensor([[output_vocab['<unk>']]])
            else:
                decoder_input = torch.tensor([[predicted_idx]])
        
        return ' '.join(translated_words)

def main():
    # 단어 사전 생성
    input_vocab, output_vocab = create_vocab(data)
    
    # 어휘 사전 크기 출력 (디버깅용)
    print(f"입력 어휘 사전 크기: {len(input_vocab)}")
    print(f"출력 어휘 사전 크기: {len(output_vocab)}")
    
    # 데이터 준비
    source, target = prepare_data(data, input_vocab, output_vocab)
    
    # 모델 초기화
    vocab_size = max(len(input_vocab), len(output_vocab))
    print(f"모델의 어휘 사전 크기: {vocab_size}")
    
    model = Seq2Seq(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=128
    )
    
    # 모델 학습
    epochs = 1000  # 에포크 수 증가
    train_model(model, source, target, input_vocab, output_vocab, epochs=epochs)
    
    # 번역 테스트
    print("\n=== 번역 테스트 ===")
    test_sentences = [
        "hello, how are you, i am fine, glad to meet you",
        "good morning",
        "how are you",
        "thank you",
        "what is your name",
        "i am tired, but it's happy day"
    ]
    
    for sentence in test_sentences:
        translated = translate(model, sentence, input_vocab, output_vocab)
        print(f"\n입력: {sentence}")
        if sentence in [src for src, _ in data]:
            tgt = [tgt for src, tgt in data if src == sentence][0]
            print(f"정답: {tgt}")
        print(f"예측: {translated}")

if __name__ == "__main__":
    main()
