import torch
import torch.nn as nn
import torch.utils.data
from datasets import load_dataset
import spacy
import torchtext
from torch.utils.data import DataLoader

##############################################################################################
# (1) 데이터 로드 함수
def load_raw_data():
    dataset = load_dataset("bentrevett/multi30k")
    train_data, valid_data, test_data = dataset["train"], dataset["validation"], dataset["test"]
    return train_data, valid_data, test_data


##############################################################################################
# (2) 토큰화 함수   딕셔너리 형태로 반환
def tokenize_example(example, en_nlp, de_nlp, max_length, lower, sos_token, eos_token):
    en_tokens = [token.text for token in en_nlp.tokenizer(example["en"])][:max_length]
    de_tokens = [token.text for token in de_nlp.tokenizer(example["de"])][:max_length]

    if lower:
        en_tokens = [token.lower() for token in en_tokens]
        de_tokens = [token.lower() for token in de_tokens]
    
    en_tokens = [sos_token] + en_tokens + [eos_token]
    de_tokens = [sos_token] + de_tokens + [eos_token]

    return {"en_tokens": en_tokens, "de_tokens": de_tokens}

##############################################################################################
# (3) 어휘 사전 구축 함수   build_vocab_from_iterator를 활용
def build_vocab(train_data, min_freq=2):
    special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]
    en_vocab = torchtext.vocab.build_vocab_from_iterator(train_data["en_tokens"], min_freq=min_freq, specials=special_tokens)
    de_vocab = torchtext.vocab.build_vocab_from_iterator(train_data["de_tokens"], min_freq=min_freq, specials=special_tokens)

    unk_index = en_vocab["<unk>"]
    en_vocab.set_default_index(unk_index)
    de_vocab.set_default_index(unk_index)

    return en_vocab, de_vocab

##############################################################################################
# (4) 텍스트 → 숫자로 변환 함수
def numericalize_example(example, en_vocab, de_vocab):
    en_ids = en_vocab.lookup_indices(example["en_tokens"])
    de_ids = de_vocab.lookup_indices(example["de_tokens"])
    return {"en_ids": en_ids, "de_ids": de_ids}

##############################################################################################
# (5) DataLoader용 collate_fn 함수
def get_collate_fn(pad_index):
    def collate_fn(batch):
        # 각 예제의 "en_ids"와 "de_ids"를 Tensor로 변환
        batch_en_ids = [torch.tensor(example["en_ids"], dtype=torch.long) for example in batch]
        batch_de_ids = [torch.tensor(example["de_ids"], dtype=torch.long) for example in batch]
        
        batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, batch_first=True, padding_value=pad_index)
        batch_de_ids = nn.utils.rnn.pad_sequence(batch_de_ids, batch_first=True, padding_value=pad_index)
        
        return {"en_ids": batch_en_ids, "de_ids": batch_de_ids}
    return collate_fn


##############################################################################################
# (6) DataLoader 생성 함수
def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = DataLoader(   
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

##############################################################################################
# (7) 전체 데이터 전처리 함수
# tokenize_example
# build_vocab
# numericalize_example
# get_data_loader w. collate_fn
## output : 토크나이징 및 패딩 & 배치처리가 완료된 데이터셋, 영어 & 독어 어휘사전, 패딩 인덱스

def load_and_process_data(batchsize=128):
    train_data, valid_data, test_data = load_raw_data()

    # 토크나이저를 불러온다
    en_nlp = spacy.load("en_core_web_sm")
    de_nlp = spacy.load("de_core_news_sm")

    # 입력으로 쓰일 것들 (딕셔너리 형태로 저장됨)
    fn_kwargs = {
        "en_nlp": en_nlp,
        "de_nlp": de_nlp,
        "max_length": 1000,
        "lower": True,
        "sos_token": "<sos>",
        "eos_token": "<eos>",
    }

    # def tokenize_example()
    train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs)
    valid_data = valid_data.map(tokenize_example, fn_kwargs=fn_kwargs)
    test_data = test_data.map(tokenize_example, fn_kwargs=fn_kwargs)

    # def build_vocab
    en_vocab, de_vocab = build_vocab(train_data)

    fn_kwargs = {"en_vocab": en_vocab, "de_vocab": de_vocab}

    # def numericalize_example
    train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs)
    valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs)
    test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs)

    # paddding index 지정
    pad_index = en_vocab["<pad>"]

    # get_data_loader & collate_fn
    train_data_loader = get_data_loader(train_data, batch_size=batchsize, pad_index=pad_index, shuffle=True)
    valid_data_loader = get_data_loader(valid_data, batch_size=batchsize, pad_index=pad_index)
    test_data_loader = get_data_loader(test_data, batch_size=batchsize, pad_index=pad_index)

    return train_data_loader, valid_data_loader, test_data_loader, en_vocab, de_vocab, pad_index

##############################################################################################
# data.py가 직접 실행될 때만 아래 코드가 실행됨
if __name__ == "__main__":
    print("데이터 전처리 실행 중... \n")
    train_data_loader, valid_data_loader, test_data_loader, en_vocab, de_vocab, pad_index = load_and_process_data()
    print("데이터 전처리 완료! 데이터셋 토크나이징 및 배치처리 + 어휘 사전 구축 + 패딩 인덱스 반환")
    print("\n추가로 어휘 사전 및 특수 토큰 확인")
    print(f"영어 어휘 사전 크기: {len(en_vocab)}")
    print(f"독일어 어휘 사전 크기: {len(de_vocab)}")

    # 특수 토큰 검증
    print("\n영어 특수 토큰 확인:", en_vocab.lookup_tokens([0, 1, 2, 3]))
    print("독일어 특수 토큰 확인:", de_vocab.lookup_tokens([0, 1, 2, 3]))




