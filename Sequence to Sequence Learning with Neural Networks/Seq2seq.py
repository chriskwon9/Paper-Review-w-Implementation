# python -m spacy download en_core_web_sm
# python -m spacy download de_core_news_sm

## CPU
# pip uninstall -y torch torchtext torchvision torchaudio
# pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# pip install torchtext==0.16.2 --no-cache-dir

## GPU
# pip uninstall torch torchvision torchaudio torchtext -y
# pip cache purge
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/mps
# pip install torchtext==0.16.2 --no-cache-dir


import torch
print(torch.__version__)
#import torch.multiprocessing
#torch.multiprocessing.set_start_method('spawn', force=True)
#torch.set_num_threads(1)


import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import spacy
import datasets
import torchtext
import tqdm
import evaluate

############################################################################################## 필요 패키지 설치 여부
print("All libraries are rightly imported!")


############################################################################################## 시드 고정 및 GPU 사용 여부
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# GPU 사용 가능 여부 확인
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Metal Performance Shaders (Apple GPU)
    print("Using MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")  # CPU만 사용
    print("Using CPU")



############################################################################################## 데이터셋
dataset = datasets.load_dataset("bentrevett/multi30k")

train_data, valid_data, test_data = (
    dataset["train"],
    dataset['validation'],
    dataset['test']
)

print(train_data[0])


############################################################################################## 토크나이징
en_nlp = spacy.load("en_core_web_sm")
de_nlp = spacy.load("de_core_news_sm")


## 토크나이저를 .tokenizer 메서드를 사용해서 불러온다
string = "Hi! Welcome to Korea"
print([token.text for token in en_nlp.tokenizer(string)])


# 데이터에 en_token & de_token을 더해주는 함수 정의     ==      tokenize한 결과물
def tokenize_example(example, en_nlp, de_nlp, max_length, lower, sos_token, eos_token):
    en_tokens = [token.text for token in en_nlp.tokenizer(example['en'])][:max_length]
    de_tokens = [token.text for token in de_nlp.tokenizer(example['de'])][:max_length]

    if lower:
        en_tokens = [token.lower() for token in en_tokens]
        de_tokens = [token.lower() for token in de_tokens]
    
    en_tokens = [sos_token] + en_tokens + [eos_token]
    de_tokens = [sos_token] + de_tokens + [eos_token]

    return {"en_tokens" : en_tokens, "de_tokens" : de_tokens}



# 모든 데이터에 대해 소문자, sos/eos, 토크나이징 해준다
max_length = 1000
lower = True
sos_token = "<sos>"
eos_token = "<eos>"

# tokenize_example에 전달할 인자들을 딕셔너리로 묶어 관리   :   function keyword arguments
fn_kwargs = {
    "en_nlp" : en_nlp,
    "de_nlp" : de_nlp,
    "max_length" : max_length,
    "lower" : lower,
    "sos_token" : sos_token,
    "eos_token" : eos_token,
}

# .map() 함수로 데이터셋의 각 요소를 tokenize_example함수를 적용해주는데, 이때 fn_kwargs를 참고해서
train_data = train_data.map(tokenize_example, fn_kwargs = fn_kwargs)
valid_data = valid_data.map(tokenize_example, fn_kwargs = fn_kwargs)
test_data = test_data.map(tokenize_example, fn_kwargs = fn_kwargs)

# tokenizing이 완료된 데이터를 출력해본다
print(train_data[0])



############################################################################################## 단어 집합 (Vocabulary)
# torchtext를 이용하면 쉽다

min_freq = 2
unk_token = "<unk>"
pad_token = "<pad>"

special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token
]

en_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data['en_tokens'],
    min_freq = min_freq,
    specials = special_tokens,
)
en_vocab.set_default_index(en_vocab[unk_token])

de_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data['de_tokens'],
    min_freq = min_freq,
    specials = special_tokens,
)
de_vocab.set_default_index(de_vocab[unk_token]) 


# .get_itos() 메서드를 사용해서 단어 집합속의 단어들을 뽑아본다
print(en_vocab.get_itos()[:10])
print(en_vocab.get_itos()[9])


print(len(en_vocab), len(de_vocab))



# 토큰들을 정수로 바꿔서 train_data, valid_data, test_data에 넣어주어야 한다    -->     그러기 위한 numericalize_example 함수
def numericalize_example(example, en_vocab, de_vocab):
    en_ids = en_vocab.lookup_indices(example['en_tokens'])
    de_ids = de_vocab.lookup_indices(example["de_tokens"])
    return {"en_ids" : en_ids, "de_ids" : de_ids}


fn_kwargs = {"en_vocab": en_vocab, "de_vocab": de_vocab}

train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs)
test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs)



# 
print(train_data[0])
print(type(train_data[0]['en_ids']))        # list





############################################################################################## Python integer --> Pytorch tensor

data_type = "torch"
format_columns = ['en_ids', 'de_ids']

train_data = train_data.with_format(
    type = data_type, columns = format_columns, output_all_columns = True
)

valid_data = valid_data.with_format(
    type = data_type, columns = format_columns, output_all_columns = True
)

test_data = test_data.with_format(
    type = data_type, columns = format_columns, output_all_columns = True
)


print(train_data[0])
print(type(train_data[0]['en_ids']))        # torch.tensor




############################################################################################## Data Loader
# 신경망의 학습은 배치 단위로 이루어짐

from torch.nn.utils.rnn import pad_sequence

pad_index = de_vocab["<pad>"]


def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_en_ids = [example["en_ids"] for example in batch]
        batch_de_ids = [example["de_ids"] for example in batch]
        batch_en_ids = pad_sequence(batch_en_ids, padding_value = pad_index)
        batch_de_ids = pad_sequence(batch_de_ids, padding_value = pad_index)
        batch = {
            "en_ids" : batch_en_ids,
            "de_ids" : batch_de_ids,
        }

        return batch
    
    return collate_fn


def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        collate_fn = collate_fn,
        shuffle = shuffle,
        num_workers = 0,
    )

    return data_loader


batch_size = 128

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle = True)
valid_data_loader = get_data_loader(train_data, batch_size, pad_index)
test_data_loader = get_data_loader(train_data, batch_size, pad_index)




############################################################################################## Building a Model

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src length, batch size, embedding dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        return hidden, cell



class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
# 출력 차원을 임베딩 차원으로 변환
        self.embedding = nn.Embedding(output_dim, embedding_dim)
# LSTM 구조 정의
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
# LSTM의 출력을 최종 단어 예측 확률 분포로 변환하는 선형층
        self.fc_out = nn.Linear(hidden_dim, output_dim)
# 드롭아웃 적용 (과적합 방지)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
# input = [batch size]
# hidden = [n layers * n directions, batch size, hidden dim]
# cell = [n layers * n directions, batch size, hidden dim]
# n directions in the decoder will both always be 1, therefore :
# hidden = [n layers, batch size, hidden dim]
# context = [n layers, batch size, hidden dim]

        input = input.unsqueeze(0)
# input = [1, batch size] 가 된다
        embedded = self.dropout(self.embedding(input))
# embedded = [1, batch size, embedding dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
# output = [seq length, batch size, hidden dim * n directions]
# hidden = [n layers * n directions, batch size, hidden dim]
# cell = [n layers * n directions, batch size, hidden dim]
# seq length and n directions will always be 1 in this decoder, therefore:
# output = [1, batch size, hidden dim]
# hidden = [n layers, batch size, hidden dim]
# cell = [n layers, batch size, hidden dim]
        prediction = self.fc_out(output.squeeze(0))
# prediction = [batch size, output_dim]
        return prediction, hidden, cell
    



##### for문을 돌리면서 trg_length만큼 출력을 생성
##### teacher_forcing_ratio를 사용하면서 실제 정답을 사용할지, 모델의 예측값 (top1) 을 사용할지 결정한다



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder 
        self.device = device
        assert(
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal"
        assert(
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers"


    def forward(self, src, trg, teacher_forcing_ratio):
# src = [src length, batch size]
# trg = [trg length, batch size]
# teacher forcing ratio is probability to use teacher forcing
# if teacher forcing ratio is 0.75, we use ground truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
# tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
# last hidden state of the encoder is used as initial hidden state of the decoder
        hidden, cell = self.encoder(src)
# hidden = [n layers * n directions, batch size, hidden dim]
# cell = [n layers * n directions, batch size, hidden dim]
# first input to the decoder is the <sos> tokens
        input = trg[0, :]
# input = [batch size]
        for t in range(1, trg_length):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs


############################################################################################## Initializing the Model


input_dim = len(de_vocab)
output_dim = len(en_vocab)
encoder_embedding_dim = 256
decoder_embedding_dim = 256
hidden_dim = 512
n_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Metal Performance Shaders (Apple GPU)
    print("Using MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")  # CPU만 사용
    print("Using CPU")


encoder = Encoder(
    input_dim,
    encoder_embedding_dim,
    hidden_dim,
    n_layers,
    encoder_dropout,
)

decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    hidden_dim,
    n_layers,
    decoder_dropout,
)

model = Seq2Seq(encoder, decoder, device).to(device)



def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


print(model.apply(init_weights))



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"The model has {count_parameters(model)} trainable parameters")




optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss(ignore_index = pad_index)

############################################################################################## Training the Model

from torch.nn.utils.rnn import pad_sequence


def train_fn(model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        src = batch["de_ids"].to(device)
        trg = batch["en_ids"].to(device)
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        # output = [trg length, batch size, trg vocab size]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        # output = [(trg length - 1) * batch size, trg vocab size]
        trg = trg[1:].view(-1)
        # trg = [(trg length - 1) * batch size]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)




def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            src = batch["de_ids"].to(device)
            trg = batch["en_ids"].to(device)
            # src = [src length, batch size]
            # trg = [trg length, batch size]
            output = model(src, trg, 0)  # turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            # output = [(trg length - 1) * batch size, trg vocab size]
            trg = trg[1:].view(-1)
            # trg = [(trg length - 1) * batch size]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)



n_epochs = 10
clip = 1.0
teacher_forcing_ratio = 0.5

best_valid_loss = float("inf")

for epoch in tqdm.tqdm(range(n_epochs)):
    train_loss = train_fn(
        model,
        train_data_loader,
        optimizer,
        criterion,
        clip,
        teacher_forcing_ratio,
        device,
    )
    valid_loss = evaluate_fn(
        model,
        valid_data_loader,
        criterion,
        device,
    )
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "tut1-model.pt")
    print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
    print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")


