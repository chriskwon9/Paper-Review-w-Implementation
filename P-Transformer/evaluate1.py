import torch
import torch.nn as nn
import evaluate


from model import make_model
from data import load_and_process_data  # Adjust the import as needed

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



def beam_search_decode(model, src, src_mask, max_len, start_symbol, end_symbol, pad_idx, beam_width=3):
    """
    Performs beam search decoding for each sample in the batch.
    Returns a list of decoded token sequences (each as a torch.Tensor without the start token).
    """
    batch_size = src.size(0)
    decoded_sequences = []
    
    # Process one sample at a time for beam search
    for i in range(batch_size):
        # Get the i-th source sample and corresponding mask
        src_i = src[i:i+1]         # shape: [1, src_len]
        src_mask_i = src_mask[i:i+1] # shape: [1, 1, 1, src_len]
        
        # Initialize the beam with the start symbol and zero cumulative log probability.
        beam = [([start_symbol], 0.0)]
        
        for _ in range(max_len):
            new_beam = []
            # For each sequence in the beam, try extending it.
            for seq, cum_log_prob in beam:
                # If this sequence already ended, keep it unchanged.
                if seq[-1] == end_symbol:
                    new_beam.append((seq, cum_log_prob))
                    continue
                # Prepare the current sequence tensor.
                trg_seq = torch.tensor(seq, dtype=torch.long, device=src.device).unsqueeze(0)  # shape: [1, current_len]
                trg_mask = create_trg_mask(trg_seq, pad_idx)
                # Get model output: shape [1, current_len, vocab_size]
                out = model(src_i, trg_seq, src_mask_i, trg_mask)
                # Focus on the last token's output and compute log probabilities.
                log_probs = torch.log_softmax(out[:, -1, :], dim=-1)  # shape: [1, vocab_size]
                log_probs = log_probs.squeeze(0)  # shape: [vocab_size]
                # Get the top beam_width tokens and their log probabilities.
                top_log_probs, top_indices = torch.topk(log_probs, beam_width)
                for token, token_log_prob in zip(top_indices, top_log_probs):
                    new_seq = seq + [token.item()]
                    new_score = cum_log_prob + token_log_prob.item()  # sum log probabilities
                    new_beam.append((new_seq, new_score))
            # Sort the new beam by cumulative score in descending order and keep top beam_width.
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_width]
            # If every sequence in the beam has generated an end_symbol, stop early.
            if all(seq[-1] == end_symbol for seq, _ in beam):
                break
        
        # Select the best sequence (highest cumulative log probability)
        best_seq = beam[0][0]
        # Remove the start token and convert to a tensor.
        decoded_sequences.append(torch.tensor(best_seq[1:], device=src.device))
    
    return decoded_sequences



def strip_special_tokens(tokens):
    return [t for t in tokens if t not in ["<pad>", "<sos>", "<eos>", "<unk>"]]

# BLEU Score 계산하는 함수
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
            
            pred_ids = beam_search_decode(
                model, src, src_mask, max_len, start_symbol, end_symbol, pad_idx, beam_width=3
            )
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
    model.load_state_dict(torch.load("ptransformer_model.pth", map_location=device))
    
    # 테스트 데이터셋에 대한 BLEU Score 확인
    bleu_score = compute_bleu(model, test_loader, device, pad_idx, de_vocab, max_len=50)
    print(f"Test BLEU score: {bleu_score:.4f}")
