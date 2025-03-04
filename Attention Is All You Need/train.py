import torch
import torch.backends
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


from model import make_model
from data1 import load_and_process_data  # 데이터 로드 함수만 가져오기

# 데이터 로드 및 전처리 실행
train_data_loader, valid_data_loader, test_data_loader, en_vocab, de_vocab, pad_index = load_and_process_data()

print("✅ 데이터 로드 완료!")
print(f"훈련 데이터 배치 개수: {len(train_data_loader)}")
print(f"검증 데이터 배치 개수: {len(valid_data_loader)}")
print(f"테스트 데이터 배치 개수: {len(test_data_loader)}")
print(f"영어 어휘 사전 크기 : {len(en_vocab)}")
print(f"독일어 어휘 사전 크기 : {len(de_vocab)}\n")


# train.py에서 데이터 로딩 후, 아래 코드 추가
for batch in train_data_loader:
    print(f"🔹 배치 크기 (영어): {batch['en_ids'].shape}")
    print(f"🔹 배치 크기 (독일어): {batch['de_ids'].shape}")
    print(f"🔹 영어 데이터 샘플: \n{batch['en_ids'][0]}")
    print(f"🔹 독일어 데이터 샘플: \n{batch['de_ids'][0]}")
    break  # 첫 번째 배치만 확인 후 종료





# GPU지원 및 모델 생성
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Metal Performance Shaders (Apple GPU)
    print("Using MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")  # CPU만 사용
    print("Using CPU")
source_vocab_size = len(en_vocab)  
target_vocab_size = len(de_vocab)  

model = make_model(source_vocab_size, target_vocab_size).to(device)



# masking함수 생성 (encoder self attention)
def create_src_mask(src, pad_idx):
    # src: [batch_size, src_len]
    # 패딩이 아닌 위치를 True로 만들어 [batch_size, 1, 1, src_len] 형태의 마스크를 생성
    return (src != pad_idx).unsqueeze(dim=1).unsqueeze(dim=2)


# masking함수 생성 (decoder masked self attention)
def create_trg_mask(trg, pad_idx):
    # trg: [batch_size, trg_len]
    trg_mask = (trg != pad_idx).unsqueeze(dim=1).unsqueeze(dim=2)
    # trg_mask: [batch_size, 1, 1, trg_len]
    seq_len = trg.size(1)
    # 이후 토큰을 막기 위한 subsequent mask: 상삼각 행렬
    subsequent_mask = torch.triu(torch.ones((1, 1, seq_len, seq_len), device=trg.device), diagonal=1).bool()
    # 두 마스크를 결합
    return trg_mask & ~subsequent_mask



# 손실 함수 정의 (pad_index는 빼고 학습)
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)


# 옵티마이저 설정
optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9)


# 학습 루프 설정
def train_epoch(model, data_loader, optimizer, criterion, device, pad_idx):
    model.train()
    epoch_loss = 0
    
    # tqdm progress bar로 배치 반복
    for batch in tqdm(data_loader, desc="Training", leave=False):
        src = batch["en_ids"].to(device)
        trg = batch["de_ids"].to(device)
        
        optimizer.zero_grad()

        src_mask = create_src_mask(src, pad_idx)
        trg_mask = create_trg_mask(trg[:, :-1], pad_idx)
        
        # target 문장에서 마지막 토큰 제외하여 입력값으로 사용
        output = model(src, trg[:, :-1], src_mask, trg_mask)
        
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)

def evaluate(model, data_loader, criterion, device, pad_idx):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            src = batch["en_ids"].to(device)
            trg = batch["de_ids"].to(device)

            src_mask = create_src_mask(src, pad_idx)
            trg_mask = create_trg_mask(trg[:, :-1], pad_idx)
            
            output = model(src, trg[:, :-1], src_mask, trg_mask)
            
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)



def train_model(model, train_loader, valid_loader, optimizer, criterion, device, pad_idx, n_epochs=10):
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, pad_idx)
        valid_loss = evaluate(model, valid_loader, criterion, device, pad_idx)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
    
    # 모델 저장
    torch.save(model.state_dict(), "new_transformer_model.pth")
    print("모델 저장 완료!")




###########################################



if __name__ == "__main__":
    train_model(model, train_data_loader, valid_data_loader, optimizer, criterion, device, pad_index)


