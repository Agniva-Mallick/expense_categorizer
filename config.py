from dataclasses import dataclass, field
from pathlib import Path
import json

@dataclass
class ModelConfig:
    vocab_size: int = 1024
    max_seq_len: int = 64
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 4
    n_kv_heads: int = 2
    d_ff: int = 688
    dropout: float = 0.1
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_weights: bool = True

    def param_count_estimate(self) -> int:
        embed = self.vocab_size * self.d_model
        per_layer = 3 * self.d_model * (self.d_model // self.n_heads * self.n_kv_heads) + self.d_model * self.d_model + 3 * self.d_model * self.d_ff + 2 * self.d_model
        head = self.d_model
        lm_head = 0 if self.tie_weights else self.vocab_size * self.d_model
        return embed + self.n_layers * per_layer + head + lm_head

@dataclass
class TrainConfig:
    data_dir: str = 'data'
    train_split: float = 0.98
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_steps: int = 5000
    learning_rate: float = 5e-4
    min_lr: float = 5e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 200
    use_amp: bool = True
    checkpoint_dir: str = 'checkpoints'
    save_every: int = 500
    eval_every: int = 100
    log_every: int = 10
    seed: int = 42
    num_workers: int = 0

@dataclass
class GenConfig:
    max_new_tokens: int = 10
    temperature: float = 0.0
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    use_kv_cache: bool = True

@dataclass
class TokenizerConfig:
    vocab_size: int = 1024
    min_frequency: int = 2
    special_tokens: list = field(default_factory=lambda: ['<|pad|>', '<|bos|>', '<|end|>', '<|unk|>'])
    save_dir: str = 'tokenizer_vocab'

def get_config(preset: str='default'):
    presets = {
        'tiny': {
            'model': ModelConfig(vocab_size=256, max_seq_len=64, n_layers=2, d_model=128, n_heads=4, n_kv_heads=2, d_ff=344),
            'train': TrainConfig(batch_size=16, max_steps=500, eval_every=50, save_every=100, warmup_steps=50)
        },
        'default': {
            'model': ModelConfig(),
            'train': TrainConfig()
        },
        'medium': {
            'model': ModelConfig(vocab_size=2048, max_seq_len=64, n_layers=8, d_model=512, n_heads=8, n_kv_heads=4, d_ff=1376),
            'train': TrainConfig(batch_size=32, max_steps=10000, learning_rate=3e-4)
        }
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'.")
    p = presets[preset]
    return p['model'], p['train'], GenConfig(), TokenizerConfig(vocab_size=p['model'].vocab_size)

def save_config(model_cfg: ModelConfig, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(model_cfg.__dict__, f, indent=2)

def load_config(path: str) -> ModelConfig:
    with open(path, 'r') as f:
        return ModelConfig(**json.load(f))
