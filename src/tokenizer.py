"""Byte-level BPE tokenizer, implemented from scratch (no external libraries).

Vocabulary layout (ids):
    0..3            special tokens  <|endoftext|> <|pad|> <|user|> <|assistant|>
    4..259          the 256 single bytes
    260..V-1        learned merges, in the order they were learned

How it works
  1. Pre-tokenization: text is split into chunks with a GPT-2-style regex (words with their
     leading space, numbers, punctuation runs, whitespace). Merges never cross chunk borders.
  2. Each chunk is converted to UTF-8 bytes, so ANY text can be encoded and there is no <unk>.
  3. Training: repeatedly find the most frequent adjacent pair of tokens across the corpus and
     merge it into a new token, until the vocabulary reaches the target size.
  4. Encoding applies the learned merges in the order they were learned (lowest rank first).
  5. Decoding concatenates the bytes of each token and decodes as UTF-8.
"""
import json
import re
from collections import Counter, defaultdict

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|user|>", "<|assistant|>"]
N_SPECIAL = len(SPECIAL_TOKENS)
BYTE_OFFSET = N_SPECIAL                       # id of byte b is BYTE_OFFSET + b
FIRST_MERGE_ID = BYTE_OFFSET + 256

# GPT-2 style pre-tokenizer (ASCII-friendly version using \w, which suits this English corpus)
PRETOKENIZE = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+|\w+""")


class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = [tuple(m) for m in (merges or [])]       # list of (id_a, id_b), rank = index
        self._build()

    # ------------------------------------------------------------------ vocabulary
    def _build(self):
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = {i: tok.encode("utf-8") for i, tok in enumerate(SPECIAL_TOKENS)}
        for b in range(256):
            self.vocab[BYTE_OFFSET + b] = bytes([b])
        for i, (a, b) in enumerate(self.merges):
            self.vocab[FIRST_MERGE_ID + i] = self.vocab[a] + self.vocab[b]
        self.special_to_id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self._cache = {}

    @property
    def vocab_size(self):
        return FIRST_MERGE_ID + len(self.merges)

    @property
    def eot_id(self):
        return self.special_to_id["<|endoftext|>"]

    @property
    def pad_id(self):
        return self.special_to_id["<|pad|>"]

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, texts, vocab_size, verbose=True):
        assert vocab_size > FIRST_MERGE_ID
        chunk_counts = Counter()
        for t in texts:
            chunk_counts.update(PRETOKENIZE.findall(t))
        words = [[BYTE_OFFSET + b for b in w.encode("utf-8")] for w in chunk_counts]
        freqs = list(chunk_counts.values())

        pair_counts = Counter()
        where = defaultdict(set)              # pair -> indices of words containing it
        for wi, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_counts[p] += freqs[wi]
                where[p].add(wi)

        merges, merge_counts = [], []           # merge_counts: corpus frequency of each pair when merged
        n_merges = vocab_size - FIRST_MERGE_ID
        for step in range(n_merges):
            if not pair_counts:
                break
            # most frequent pair; ties broken by smallest ids for determinism
            best = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
            if pair_counts[best] < 2:
                break
            new_id = FIRST_MERGE_ID + len(merges)
            merges.append(best)
            merge_counts.append(pair_counts[best])
            for wi in list(where[best]):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:]):                 # remove this word's old pairs
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    where[p].discard(wi)
                out, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                words[wi] = out
                for p in zip(out, out[1:]):             # add its new pairs
                    pair_counts[p] += f
                    where[p].add(wi)
            where.pop(best, None)
            if verbose and (step + 1) % 500 == 0:
                print(f"  merge {step + 1}/{n_merges}")
        tok = cls(merges)
        tok.merge_counts = merge_counts
        return tok

    # ------------------------------------------------------------------ encode / decode
    def _bpe(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        ids = [BYTE_OFFSET + b for b in chunk.encode("utf-8")]
        while len(ids) > 1:
            pairs = [(self.ranks.get(p, 1 << 30), i) for i, p in enumerate(zip(ids, ids[1:]))]
            rank, i = min(pairs)
            if rank == 1 << 30:
                break
            ids = ids[:i] + [FIRST_MERGE_ID + rank] + ids[i + 2:]
        self._cache[chunk] = ids
        return ids

    def encode(self, text, allow_special=False):
        """Text -> list of token ids. Special-token strings are only treated as special
        tokens when allow_special=True (so user text can never inject them)."""
        ids = []
        if allow_special:
            parts = re.split("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")", text)
        else:
            parts = [text]
        for part in parts:
            if allow_special and part in self.special_to_id:
                ids.append(self.special_to_id[part])
            elif part:
                for chunk in PRETOKENIZE.findall(part):
                    ids.extend(self._bpe(chunk))
        return ids

    def decode(self, ids, skip_special=False):
        out = b"".join(self.vocab[i] for i in ids if not (skip_special and i < N_SPECIAL))
        return out.decode("utf-8", errors="replace")

    def token_strings(self, ids):
        """Human-readable pieces, for showing how text is split ('·' marks a leading space)."""
        return [self.vocab[i].decode("utf-8", errors="replace").replace(" ", "·") for i in ids]

    # ------------------------------------------------------------------ save / load
    def save(self, path):
        data = {"type": "byte-level BPE", "version": 1, "special_tokens": SPECIAL_TOKENS,
                "byte_offset": BYTE_OFFSET, "pretokenize_pattern": PRETOKENIZE.pattern,
                "vocab_size": self.vocab_size, "merges": [list(m) for m in self.merges]}
        if getattr(self, "merge_counts", None):
            data["merge_counts"] = self.merge_counts      # informational only
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["special_tokens"] == SPECIAL_TOKENS and data["byte_offset"] == BYTE_OFFSET
        return cls(data["merges"])
