import os
import random
import struct
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from config import ModelConfig, TrainConfig, TokenizerConfig
from tokenizer import BPETokenizer

def _generate_synthetic_expenses(n: int=100000) -> list[str]:
    # We will generate highly varied, realistic-looking bank transactions.
    categories = {
        "Food": ["MCDONALDS", "STARBUCKS", "CHIPOTLE", "UBER EATS", "DOORDASH", "TST* SWEETGREEN", "WHOLEFDS", "TRADER JOE'S", "KFC", "PIZZA HUT"],
        "Transportation": ["UBER *TRIP", "LYFT", "CHEVRON", "SHELL OIL", "EXXONMOBIL", "MTA*NYCT PAYGO", "BART-DALY CITY", "PARKING METER", "TFL TRAVEL CHARGE"],
        "Utilities": ["COMCAST", "AT&T", "VERIZON WIRELESS", "PG&E", "CON EDISON", "T-MOBILE", "WATER DEPT", "CITY OF LA UTIL"],
        "Entertainment": ["NETFLIX.COM", "SPOTIFY", "HULU", "STEAM GAMES", "PLAYSTATION", "AMC THEATRES", "DISNEY PLUS", "TICKETMASTER"],
        "Shopping": ["AMZN Mktp US", "AMAZON.COM", "TARGET", "WALMART", "APPLE.COM/BILL", "HOME DEPOT", "BEST BUY", "IKEA", "COSTCO WHSE"],
        "Income": ["DIRECT DEP", "PAYROLL", "VENMO CASHOUT", "ZELLE TRANSFER", "PAYPAL TRANSFER", "CASH APP", "DIVIDEND"],
        "Housing": ["RENT PAYMENT", "MORTGAGE", "HOA DUES", "HOME INSR", "APT LEASING"],
        "Health": ["CVS/PHARMACY", "WALGREENS", "KAISER PERMANENTE", "QUEST DIAGNOSTICS", "DENTAL CARE", "VISION CENTER"]
    }
    
    locations = [" NY", " CA", " TX", " WA", " IL", " FL", " UK", " 800-123-4567", " ONLINE", " SF", " LA", ""]
    dates = [" 12/04", " 01/15", " 07/22", " 11/30", " 03/10", ""]
    ids = ["*TO382", "*7721", "*9932", "#84", "- 38291", ""]
    
    stories = []
    for _ in range(n):
        cat = random.choice(list(categories.keys()))
        merchant = random.choice(categories[cat])
        loc = random.choice(locations)
        date = random.choice(dates)
        txn_id = random.choice(ids)
        
        # Construct messy string
        format_type = random.random()
        if format_type < 0.2:
            amount = random.randint(10, 500)
            txn_str = f"i spend {amount} on {merchant.lower()}"
        elif format_type < 0.35:
            amount = random.randint(5, 1000)
            txn_str = f"paid {amount} for {merchant.lower()}"
        elif format_type < 0.5:
            txn_str = f"just bought stuff at {merchant.lower()}"
        else:
            components = [merchant]
            if random.random() > 0.5: components.append(txn_id)
            if random.random() > 0.5: components.append(date)
            if random.random() > 0.5: components.append(loc)
            random.shuffle(components)
            txn_str = "".join(components).strip()
            
            # Chance to make raw bank codes lowercase
            if random.random() < 0.3:
                txn_str = txn_str.lower()
        
        # Format for LLM training
        story = f"TXN: {txn_str}\nCAT: {cat}"
        stories.append(story)
        
    return stories

def prepare_data(data_dir: str, tokenizer: BPETokenizer, max_seq_len: int, train_split: float=0.98):
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    train_file = data_path / 'train.bin'
    val_file = data_path / 'val.bin'
    
    if train_file.exists() and val_file.exists():
        print(f"Tokenized data already exists at {data_path}/")
        return str(train_file), str(val_file)
        
    print("Generating synthetic expense data...")
    stories = _generate_synthetic_expenses(100000)
    
    # Save raw for inspection
    with open(data_path / 'expenses_raw.txt', 'w', encoding='utf-8') as f:
        for s in stories:
            f.write(s + "\n<|end|>\n")
            
    print("Tokenizing data...")
    all_sequences = []
    for story in tqdm(stories, desc="Tokenizing"):
        token_ids = tokenizer.encode(story, add_special_tokens=False)
        token_ids = token_ids + [tokenizer.eos_id] # Add EOS manually
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len-1] + [tokenizer.eos_id]
        if len(token_ids) >= 3:
            all_sequences.append(token_ids)
            
    random.shuffle(all_sequences)
    split_idx = int(len(all_sequences) * train_split)
    train_seqs = all_sequences[:split_idx]
    val_seqs = all_sequences[split_idx:]
    
    _save_sequences(train_seqs, str(train_file), tokenizer.pad_id, max_seq_len)
    _save_sequences(val_seqs, str(val_file), tokenizer.pad_id, max_seq_len)
    
    return str(train_file), str(val_file)

def _save_sequences(sequences: list[list[int]], filepath: str, pad_id: int, max_len: int):
    with open(filepath, 'wb') as f:
        f.write(struct.pack('I', len(sequences)))
        f.write(struct.pack('H', max_len))
        for seq in sequences:
            padded = seq + [pad_id] * (max_len - len(seq))
            padded = padded[:max_len]
            for token_id in padded:
                f.write(struct.pack('H', token_id))

class ExpenseDataset(Dataset):
    def __init__(self, filepath: str, pad_id: int=0):
        self.pad_id = pad_id
        with open(filepath, 'rb') as f:
            self.num_sequences = struct.unpack('I', f.read(4))[0]
            self.seq_len = struct.unpack('H', f.read(2))[0]
            total_tokens = self.num_sequences * self.seq_len
            raw = f.read(total_tokens * 2)
            tokens = struct.unpack(f"{total_tokens}H", raw)
        self.data = torch.tensor(tokens, dtype=torch.long).view(self.num_sequences, self.seq_len)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int):
        tokens = self.data[idx]
        input_ids = tokens[:-1]
        targets = tokens[1:].clone()
        targets[targets == self.pad_id] = -1
        return input_ids, targets

def get_dataloaders(model_cfg: ModelConfig, train_cfg: TrainConfig, tok_cfg: TokenizerConfig):
    data_dir = train_cfg.data_dir
    tok_dir = tok_cfg.save_dir
    tok_path = Path(tok_dir)
    
    if (tok_path / 'config.json').exists():
        tokenizer = BPETokenizer.load(tok_dir)
    else:
        print("Training tokenizer...")
        corpus = "\n".join(_generate_synthetic_expenses(10000))
        tokenizer = BPETokenizer(vocab_size=tok_cfg.vocab_size, special_tokens=tok_cfg.special_tokens)
        tokenizer.train(corpus)
        tokenizer.save(tok_dir)
        
    train_file, val_file = prepare_data(data_dir, tokenizer, model_cfg.max_seq_len, train_cfg.train_split)
    
    train_dataset = ExpenseDataset(train_file, pad_id=tokenizer.pad_id)
    val_dataset = ExpenseDataset(val_file, pad_id=tokenizer.pad_id)
    
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False)
    
    return train_loader, val_loader, tokenizer
