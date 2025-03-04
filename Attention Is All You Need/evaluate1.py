import torch
import torch.nn as nn
import evaluate


from model import make_model
from data1 import load_and_process_data  # Adjust the import as needed

# Mask creation functions
def create_src_mask(src, pad_idx):
    # src: [batch_size, src_len]
    # return[batch_size, 1, 1, src_len]
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)




def create_trg_mask(trg, pad_idx):
    # trg: [batch_size, trg_len]
    trg_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(2)
    # Returns mask of shape [batch_size, 1, 1, trg_len]
    seq_len = trg.size(1)
    # seq_len = trg_len
    # subsequent_mask : autoregressive 성질
    subsequent_mask = torch.triu(
        torch.ones((1, 1, seq_len, seq_len), device=trg.device), diagonal=1
        ).bool()
    # subsequent_mask = [1, 1, trg_len, trg_len]
    # torch.triu & diagonal=1로 상삼각 행렬로 변환
    return trg_mask & ~subsequent_mask

    


# Greedy decoding function
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, pad_idx):
    batch_size = src.size(0)
    # Initialize output with <sos> token
    ys = torch.full((batch_size, 1), start_symbol, dtype=torch.long, device=src.device)

    for _ in range(max_len):
        trg_mask = create_trg_mask(ys, pad_idx)
        # Pass through the model; adjust arguments if needed.
        out = model(src, ys, src_mask, trg_mask)  
        # out: [batch_size, seq_len, vocab_size]
        next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)  
        # 마지막 단어의 확률분포([:, -1, :]) 중 가장 확률이 높은(.argmax(dim=-1)) 단어를 선택
        ys = torch.cat([ys, next_token], dim=1)
        # [batch_size, decoded_len]
        if (next_token.squeeze() == end_symbol).all():
            break
    return ys[:, 1:]  
    # <sos> token 제거한다.

# Helper to strip special tokens from a list of tokens
def strip_special_tokens(tokens):
    return [t for t in tokens if t not in ["<pad>", "<sos>", "<eos>", "<unk>"]]

# Compute BLEU score over a given DataLoader
def compute_bleu(model, data_loader, device, pad_idx, de_vocab, max_len=50):
    bleu_metric = evaluate.load("bleu")
    model.eval()

    all_predictions = []
    # 모델이 생성한 번역을 저장할 리스트
    all_references = []
    # 정답을 저장할 리스트

    with torch.no_grad():
        for batch in data_loader:
            src = batch["en_ids"].to(device)
            # src : [batch_size, src_len]
            trg = batch["de_ids"].to(device)
            # trg : [batch_size, trg_len]
            src_mask = create_src_mask(src, pad_idx)
            # src_mask : [batch_size, 1, 1, src_len]

            # greedy_decoding에서 시작과 끝을 알린다
            start_symbol = de_vocab.lookup_indices(["<sos>"])[0]
            end_symbol = de_vocab.lookup_indices(["<eos>"])[0]
            
            pred_ids = greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, pad_idx)
            # pred_ids : [batch_size, decoded_len]
            
            # 번역된 문장(pred_ids)과 정답(trg)을 실제 단어로 변경
            for pred_seq, ref_seq in zip(pred_ids, trg):
                pred_tokens = de_vocab.lookup_tokens(pred_seq.cpu().numpy())
                ref_tokens = de_vocab.lookup_tokens(ref_seq.cpu().numpy())
                pred_tokens = strip_special_tokens(pred_tokens)
                ref_tokens = strip_special_tokens(ref_tokens)
                
                all_predictions.append(" ".join(pred_tokens))
                all_references.append([" ".join(ref_tokens)])
                # BLEU 계산을 위해 리스트에 저장
                
    results = bleu_metric.compute(predictions=all_predictions, references=all_references)
    return results["bleu"]

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터 전처리 함수 + 어휘 사전 
    train_loader, valid_loader, test_loader, en_vocab, de_vocab, pad_idx = load_and_process_data()
    
    # 모델을 만들고, train.py로 학습된 트랜스포머 모델 불러옴 
    model = make_model(len(en_vocab), len(de_vocab)).to(device)
    model.load_state_dict(torch.load("new_transformer_model.pth", map_location=device))
    
    # 테스트 데이터셋에 대한 BLEU Score 확인
    bleu_score = compute_bleu(model, test_loader, device, pad_idx, de_vocab, max_len=50)
    print(f"Test BLEU score: {bleu_score:.4f}")
