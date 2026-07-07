import argparse
import math
import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
from config import ModelConfig, TrainConfig, GenConfig, TokenizerConfig, get_config, save_config, load_config
from model import ExpenseCategorizer
from dataset import get_dataloaders
from generate import generate_category
from tokenizer import BPETokenizer

def get_lr(step: int, train_cfg: TrainConfig) -> float:
    max_lr = train_cfg.learning_rate
    min_lr = train_cfg.min_lr
    warmup = train_cfg.warmup_steps
    max_steps = train_cfg.max_steps
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)

@torch.no_grad()
def evaluate(model: ExpenseCategorizer, val_loader, device: torch.device, use_amp: bool) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            _, loss, _ = model(batch_x, batch_y)
        total_loss += loss.item()
        num_batches += 1
    model.train()
    return total_loss / max(num_batches, 1)

def save_checkpoint(model: ExpenseCategorizer, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, step: int, val_loss: float, model_cfg: ModelConfig, checkpoint_dir: str, is_best: bool=False):
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'step': step, 'val_loss': val_loss}
    torch.save(checkpoint, ckpt_path / 'checkpoint.pt')
    save_config(model_cfg, str(ckpt_path / 'model_config.json'))
    if is_best:
        torch.save(model.state_dict(), ckpt_path / 'best_model.pt')
        print(f'  ⭐ Saved new best model (val_loss={val_loss:.4f})')
    else:
        print(f'  💾 Checkpoint saved at step {step} (val_loss={val_loss:.4f})')

def load_checkpoint(model: ExpenseCategorizer, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, checkpoint_dir: str, device: torch.device) -> tuple[int, float]:
    ckpt_file = Path(checkpoint_dir) / 'checkpoint.pt'
    if not ckpt_file.exists():
        print('  No checkpoint found. Starting from scratch.')
        return (0, float('inf'))
    print(f'  Loading checkpoint from {ckpt_file}...')
    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    state_dict = checkpoint['model']
    state_dict.pop('freqs_cos', None)
    state_dict.pop('freqs_sin', None)
    model.load_state_dict(state_dict, strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    step = checkpoint['step']
    val_loss = checkpoint.get('val_loss', float('inf'))
    print(f'  Resumed from step {step} (val_loss={val_loss:.4f})')
    return (step, val_loss)

def train(model_cfg: ModelConfig, train_cfg: TrainConfig, gen_cfg: GenConfig, tok_cfg: TokenizerConfig, resume: bool=False):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'\\n🖥️  GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        train_cfg.use_amp = False
        print('\\n⚠️  No GPU detected. Training on CPU.')
    train_loader, val_loader, tokenizer = get_dataloaders(model_cfg, train_cfg, tok_cfg)
    model_cfg.vocab_size = tokenizer.vocab_size
    model = ExpenseCategorizer(model_cfg).to(device)
    print(f'\\nModel Parameters: {model_cfg.param_count_estimate() / 1000000.0:.2f}M')
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=train_cfg.use_amp)
    start_step = 0
    best_val_loss = float('inf')
    if resume:
        start_step, best_val_loss = load_checkpoint(model, optimizer, scaler, train_cfg.checkpoint_dir, device)
    print(f'\\n{'=' * 60}\\nTRAINING\\n{'=' * 60}')
    model.train()
    train_iter = iter(train_loader)
    start_time = time.time()
    pbar = tqdm(range(start_step, train_cfg.max_steps), desc='Training', initial=start_step, total=train_cfg.max_steps)
    for step in pbar:
        lr = get_lr(step, train_cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.zero_grad()
        try:
            batch_x, batch_y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_x, batch_y = next(train_iter)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        with torch.amp.autocast(device_type='cuda', enabled=train_cfg.use_amp):
            _, loss, _ = model(batch_x, batch_y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'lr': f'{lr:.1e}'})
        if (step + 1) % train_cfg.eval_every == 0:
            val_loss = evaluate(model, val_loader, device, train_cfg.use_amp)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            tqdm.write(f'  Step {step + 1}: val_loss={val_loss:.4f} {('⭐' if is_best else '')}')
            sample_txn = 'AMZN Mktp US*TO382'
            cat = generate_category(model, tokenizer, sample_txn, device=device)
            tqdm.write(f"  Test: '{sample_txn}' -> {cat}")
        if (step + 1) % train_cfg.save_every == 0:
            val_loss = evaluate(model, val_loader, device, train_cfg.use_amp)
            is_best = val_loss <= best_val_loss
            if is_best:
                best_val_loss = val_loss
            save_checkpoint(model, optimizer, scaler, step + 1, val_loss, model_cfg, train_cfg.checkpoint_dir, is_best)
    print(f'\\n{'=' * 60}\\nTRAINING COMPLETE\\n{'=' * 60}')
    print(f'  Total time: {(time.time() - start_time) / 60:.1f} min')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', type=str, default='default', choices=['tiny', 'default', 'medium'])
    args = parser.parse_args()
    model_cfg, train_cfg, gen_cfg, tok_cfg = get_config(args.preset)
    train(model_cfg, train_cfg, gen_cfg, tok_cfg)