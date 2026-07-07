import argparse
import time
import json
import torch
import torch.nn.functional as F
from config import load_config, GenConfig
from model import ExpenseCategorizer
from tokenizer import BPETokenizer
from pathlib import Path

def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_values[..., -1:]
    return logits.masked_fill(logits < threshold, float('-inf'))

def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cumulative_probs - sorted_probs > p
    sorted_logits[sorted_mask] = float('-inf')
    logits = torch.empty_like(logits).scatter_(-1, sorted_indices, sorted_logits)
    return logits

def generate_category(model: ExpenseCategorizer, tokenizer: BPETokenizer, transaction: str, max_new_tokens: int=10, temperature: float=0.0, top_k: int=50, top_p: float=0.95, device='cpu'):
    prompt = f'TXN: {transaction}\nCAT:'
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated_ids = []
    confidence = 0.0
    with torch.no_grad():
        for i in range(max_new_tokens):
            x = torch.tensor([input_ids + generated_ids], dtype=torch.long, device=device)
            logits, _, _ = model(x)
            next_logits = logits[0, -1, :]
            
            if i == 0:
                probs_all = F.softmax(next_logits, dim=-1)
                
            if temperature > 0:
                next_logits = next_logits / temperature
                next_logits = apply_top_k(next_logits, top_k)
                next_logits = apply_top_p(next_logits, top_p)
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_id = torch.argmax(next_logits, dim=-1).item()
                
            if i == 0:
                confidence = probs_all[next_id].item()
                
            if next_id == tokenizer.eos_id:
                break
            generated_ids.append(next_id)
    return tokenizer.decode(generated_ids).strip(), confidence, len(input_ids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txn', type=str, required=True, help='Bank transaction text')
    args = parser.parse_args()
    
    import re
    extracted_amount = 0.0
    matches = re.findall(r'\d+(?:\.\d{1,2})?', args.txn)
    if matches:
        extracted_amount = float(matches[-1])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_dir = Path('checkpoints')
    model_cfg = load_config(str(ckpt_dir / 'model_config.json'))
    tokenizer = BPETokenizer.load('tokenizer_vocab')
    
    print(f"[INFO]   Architecture: GPT-Style Decoder | Parameters: {model_cfg.param_count_estimate() / 1e6:.2f}M")
    print(f"[INFO]   Attention: Grouped-Query (GQA) | Activation: SwiGLU")
    print("-" * 60)
    print(f"[INPUT]  \"{args.txn}\"")
    
    model = ExpenseCategorizer(model_cfg).to(device)
    best_path = ckpt_dir / 'best_model.pt'
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=True)
    else:
        ckpt = torch.load(ckpt_dir / 'checkpoint.pt', map_location=device, weights_only=False)['model']
    ckpt.pop('freqs_cos', None)
    ckpt.pop('freqs_sin', None)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    
    start_time = time.perf_counter()
    cat, conf, num_toks = generate_category(model, tokenizer, args.txn, device=device)
    end_time = time.perf_counter()
    
    ledger_path = Path('ledger.json')
    if ledger_path.exists():
        with open(ledger_path, 'r') as f:
            ledger = json.load(f)
    else:
        ledger = {'balance': 0.0}
        
    cat_upper = cat.upper()
    operator = ""
    if extracted_amount > 0:
        if cat_upper == 'INCOME':
            ledger['balance'] += extracted_amount
            operator = "+"
        else:
            ledger['balance'] -= extracted_amount
            operator = "-"
        with open(ledger_path, 'w') as f:
            json.dump(ledger, f)
    
    print(f"[TOKENS] [BPE Tokenizer] Encoded to {num_toks} sub-word tokens")
    print("-" * 60)
    print(f"[OUTPUT] Category: {cat_upper}")
    if extracted_amount > 0:
        print(f"[LEDGER] Amount extracted: {operator}${extracted_amount:.2f} | Current Balance: ${ledger['balance']:.2f}")
    
    latency = (end_time - start_time) * 1000
    speed = num_toks / (end_time - start_time)
    print(f"[STATS]  Confidence: {conf*100:.1f}% | Latency: {latency:.1f}ms | Speed: {speed:.2f} tokens/sec")