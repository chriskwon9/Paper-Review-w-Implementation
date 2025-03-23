import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
import pandas as pd
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import warnings
import torch.nn.functional as F



import sys
print(sys.executable)


warnings.filterwarnings("ignore")
RUN_EXAMPLES = True




### Transformer의 기본 구조
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.generator = generator

    def encode(self, source, source_mask):
        return self.encoder(self.source_embed(source), source_mask)
    

    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.target_embed(target), memory, source_mask, target_mask)
    

    def forward(self, source, target, source_mask, target_mask):
        memory = self.encode(source, source_mask)
        decoded =  self.decode(memory, source_mask, target, target_mask)
        return self.generator(decoded)
    


### 뒤에 make_model()함수에서 어떻게 쓰이는가

# encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
# decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)

# source_embed = nn.Sequential(Embeddings(d_model, source_vocab), c(position))
# target_embed = nn.Sequential(Embeddings(d_model, target_vocab), c(position))

# generator = Generator(d_model, target_vocab)




# 클래스, 함수정의가 필요한게 
# Encoder, EncoderLayer, Decoder, DecoderLayer, Embeddings, Generator
# 이 모든것이 정의도면 뒤에 make_model()함수로 


# N개의 동일한 레이어를 만들고 모듈리스트에 넣는다
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


### Transformer의 연산 (Self-Attention & Feed-Forward)
class MultiHeadAttention(nn.Module):
    def __init__(self, head, dim_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert dim_model % head == 0, "dim_model은 head의 배수여야 합니다."
        self.dim_k = dim_model // head
        self.head = head
        self.linears = clones(nn.Linear(dim_model, dim_model), 4)
        self.attention = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 입력 [batch_size, seq_len, vocab_size]
        batch_size = query.size(0)

        # [batch_size, seq_len, head, dim_k]
        # [batch_size, head, seq_len, dim_k]
        query = self.linears[0](query).view(batch_size, -1, self.head, self.dim_k).transpose(1, 2)
        key = self.linears[1](key).view(batch_size, -1, self.head, self.dim_k).transpose(1, 2)
        value = self.linears[2](value).view(batch_size, -1, self.head, self.dim_k).transpose(1, 2)

        #  마스크 차원 맞추기 (batch_size, 1, 1, seq_len)
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)

        #  Scaled Dot-Product Attention 적용
        x, self.attention = attention(query, key, value, mask=mask, dropout=self.dropout)

        #  Multi-head 결합 (원래 차원으로 복원)
        # [batch_size, seq_len, dim_model]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.dim_k)

        return self.linears[-1](x)


def attention(query, key, value, mask=None, dropout=None):
    dim_k = query.size(-1)

    #  Query & Key 크기 체크 (디버깅용)
    assert query.size(-1) == key.size(-1), f"Query dim {query.size(-1)} != Key dim {key.size(-1)}"
    assert key.size(-1) == value.size(-1), f"Key dim {key.size(-1)} != Value dim {value.size(-1)}"


    #  Scaled Dot-Product Attention 점수 계산
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_k)

    #  마스크 적용 (mask 차원이 4인지 먼저 확인[batch_size, 1, 1, seq_len])
    if mask is not None:
        assert mask.dim() == 4, f"Mask shape should be (batch_size, 1, 1, seq_len), but got {mask.shape}"
        scores = scores.masked_fill(mask == 0, -1e9)

    #  Softmax & Dropout 적용
    p_attention = scores.softmax(dim=-1)
    if dropout is not None:
        p_attention = dropout(p_attention)

    return torch.matmul(p_attention, value), p_attention




def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_model, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x).relu()
        x = self.dropout(x)
        return self.w_2(x)



class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # pre norm
        # return x + self.dropout(sublayer(self.norm(x)))
        # post norm (논문에서 사용한 방식)
        return self.norm(x + self.dropout(sublayer(x)))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.size = size
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2
    




### Encoder와 Decoder의 레이어
class Embeddings(nn.Module):
    def __init__(self, dim_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, dim_model)
        self.dim_model = dim_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.dim_model)




# Pytorch의 nn.Module을 상속해서 PositionalEncoding을 하나의 모듈로
# (max_len, dim_model) 크기의 0으로 채워진 텐서 생성
# position : (max_len, 1) 형태의 위치 인덱스 행렬 생성
# div_term : 분모 텀

# 짝수 : sin(pos / 10000^(2i/dim_model))
# 홀수 : cos(pos / 10000^(2i/dim_model))
# 여기까지는 (max_len, dim_model)
# .unsqueeze(0) 으로 배치 차원 추가 (1, max_len, dim_model)
# [:, : x.size(1)] : 결국 입력 데이터 (batch_size, seq_len, dim_model)에서 seq_len까지만 필요함


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2) * -(math.log(10000.0) / dim_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)



class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.norm = LayerNorm(size)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x



class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        ## 수정사항
        outputs = []
        for layer in self.layers:
            x = layer(x, mask)
            outputs.append(x)
        outputs[-1] = self.norm(outputs[-1])
        return outputs
    




class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, source_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.source_attn = source_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.size = size

    def forward(self, x, memory, source_mask, target_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda x: self.source_attn(x, m, m, source_mask))
        x = self.sublayer[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, enc_outputs, source_mask, target_mask):
        ## 수정사항
        for i, layer in enumerate(self.layers):
            x = layer(x, enc_outputs[i], source_mask, target_mask)
        return self.norm(x)



class Generator(nn.Module):
    # standard linear + softmax generation step
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)




def make_model(source_vocab, target_vocab, N=3, dim_model=256, dim_ff=512, head=8, dropout=0.1):
    c = copy.deepcopy

    attn = MultiHeadAttention(head, dim_model)
    ff = PositionwiseFeedForward(dim_model, dim_ff, dropout)
    position = PositionalEncoding(dim_model, dropout)

    # 수정된 Encoder: 각 레이어의 출력을 리스트로 반환
    encoder = Encoder(EncoderLayer(dim_model, c(attn), c(ff), dropout), N)
    # 수정된 Decoder: encoder의 출력 리스트에서 각 레이어에 맞는 memory를 사용
    decoder = Decoder(DecoderLayer(dim_model, c(attn), c(attn), c(ff), dropout), N)

    source_embed = nn.Sequential(Embeddings(dim_model, source_vocab), c(position))
    target_embed = nn.Sequential(Embeddings(dim_model, target_vocab), c(position))
    generator = Generator(dim_model, target_vocab)

    # 수정된 EncoderDecoder: encoder가 리스트로 memory를 반환하고 decoder가 이를 사용
    model = EncoderDecoder(
        encoder=encoder,
        decoder=decoder,
        source_embed=source_embed,
        target_embed=target_embed,
        generator=generator
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model





model = make_model(source_vocab=5893, target_vocab=7853)

print(model)