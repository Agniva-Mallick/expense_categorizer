import json
import re
from pathlib import Path

class BPETokenizer:

    def __init__(self, vocab_size: int=1024, special_tokens: list[str]=None):
        if special_tokens is None:
            special_tokens = ['<|pad|>', '<|bos|>', '<|end|>', '<|unk|>']
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.vocab = {}
        self.merges = {}
        self.inverse_vocab = {}
        for idx, token in enumerate(self.special_tokens):
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token

    def _get_stats(self, vocab: dict[tuple, int]) -> dict[tuple, int]:
        pairs = {}
        for word, freq in vocab.items():
            symbols = list(word)
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs

    def _merge_vocab(self, pair: tuple, v_in: dict[tuple, int]) -> dict[tuple, int]:
        v_out = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word_in, freq in v_in.items():
            w = ' '.join(word_in)
            w = w.replace(bigram, replacement)
            v_out[tuple(w.split())] = freq
        return v_out

    def train(self, text: str):
        print('Pre-tokenizing text...')
        words = re.findall('\\S+|\\s+', text)
        word_freqs = {}
        for w in words:
            word_freqs[w] = word_freqs.get(w, 0) + 1
        vocab_words = {tuple(w): freq for w, freq in word_freqs.items()}
        idx = len(self.special_tokens)
        base_chars = set()
        for word in vocab_words:
            for char in word:
                base_chars.add(char)
        for char in sorted(list(base_chars)):
            self.vocab[char] = idx
            self.inverse_vocab[idx] = char
            idx += 1
        target_merges = self.vocab_size - idx
        if target_merges < 0:
            print(f'Warning: Base characters ({idx}) exceed vocab size ({self.vocab_size}).')
            return
        print(f'Starting with {idx} base characters, learning {target_merges} merges...')
        for i in range(target_merges):
            pairs = self._get_stats(vocab_words)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab_words = self._merge_vocab(best, vocab_words)
            self.merges[best] = idx
            token = ''.join(best)
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
            idx += 1
            if (i + 1) % 50 == 0:
                print(f'  Learned {i + 1}/{target_merges} merges...')
        self.vocab_size = len(self.vocab)
        print(f'Tokenizer training complete. Final vocab size: {self.vocab_size}')

    def encode(self, text: str, add_special_tokens: bool=False) -> list[int]:
        tokens = []
        if add_special_tokens:
            pass
        words = re.findall('\\S+|\\s+', text)
        for w in words:
            symbols = list(w)
            while len(symbols) > 1:
                pairs = self._get_stats({tuple(symbols): 1})
                if not pairs:
                    break
                min_merge = None
                min_idx = float('inf')
                for p in pairs:
                    if p in self.merges and self.merges[p] < min_idx:
                        min_merge = p
                        min_idx = self.merges[p]
                if min_merge is None:
                    break
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == min_merge[0] and (symbols[i + 1] == min_merge[1]):
                        new_symbols.append(min_merge[0] + min_merge[1])
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols
            for sym in symbols:
                tokens.append(self.vocab.get(sym, self.unk_id))
        return tokens

    def decode(self, tokens: list[int]) -> str:
        text = ''
        for t in tokens:
            if t in self.inverse_vocab and t >= len(self.special_tokens):
                text += self.inverse_vocab[t]
        return text

    def save(self, path: str):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        merges_str = {f'{k[0]},{k[1]}': v for k, v in self.merges.items()}
        with open(p / 'config.json', 'w', encoding='utf-8') as f:
            json.dump({'vocab_size': self.vocab_size, 'special_tokens': self.special_tokens, 'vocab': self.vocab, 'merges': merges_str}, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(Path(path) / 'config.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok = cls(data['vocab_size'], data['special_tokens'])
        tok.vocab = data['vocab']
        tok.inverse_vocab = {int(v): k for k, v in tok.vocab.items()}
        merges = {}
        for k, v in data.get('merges', {}).items():
            parts = k.split(',')
            if len(parts) == 2:
                merges[parts[0], parts[1]] = int(v)
        tok.merges = merges
        return tok