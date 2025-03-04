import torch
import spacy
from data1 import load_and_process_data
from model import make_model
from evaluate1 import create_src_mask, create_trg_mask, greedy_decode



def translate_sentence(
        sentence,
        model,
        en_nlp,
        en_vocab,
        de_vocab,
        device,
        pad_idx,
        max_len=50,
):
    model.eval()
    tokens = [tok.text.lower() for tok in en_nlp.tokenizer(sentence)]
    tokens = ["<sos>"] + tokens + ["<eos>"]
    src_ids = [en_vocab[token] for token in tokens]
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_mask = create_src_mask(src_tensor, pad_idx)
    
    start_symbol = de_vocab["<sos>"]
    end_symbol = de_vocab["<eos>"]
    decoded_ids = greedy_decode(model, src_tensor, src_mask, max_len, start_symbol, end_symbol, pad_idx)
    
    pred_tokens = de_vocab.lookup_tokens(decoded_ids.squeeze(0).cpu().numpy())
    # remove special tokens
    pred_tokens = [t for t in pred_tokens if t not in ["<sos>", "<eos>", "<pad>", "<unk>"]]

    return " ".join(pred_tokens)





if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data and vocab
    train_data_loader, valid_data_loader, test_data_loader, en_vocab, de_vocab, pad_idx = load_and_process_data()
    
    # Create and load model
    model = make_model(len(en_vocab), len(de_vocab)).to(device)
    model.load_state_dict(torch.load("new_transformer_model.pth", map_location=device))
    
    # Load spacy tokenizers if not already in data1.py
    en_nlp = spacy.load("en_core_web_sm")
    
    # Try some random sentences
    sentences = [
        "A man is watching Netflix.",
        "Have you prepared for the group meeting?",
        "They are waiting for the bus",
    ]
    
    for s in sentences:
        translation = translate_sentence(s, model, en_nlp, en_vocab, de_vocab, device, pad_idx, max_len=50)
        print(f"English: {s}\nGerman: {translation}\n")