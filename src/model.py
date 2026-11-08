"""Decoder-only Transformer (GPT-style), written from basic PyTorch layers.

Spec checklist (section 4):
  token embeddings + positional information   -> GPT.tok_emb (V x d), GPT.pos_emb (T x d, learned)
  causal multi-head self-attention            -> CausalSelfAttention with separate W_Q, W_K, W_V, W_O
  causal masking                              -> lower-triangular mask; future positions get -inf
  decoder blocks: attention, FFN, residual,   -> Block: x + Attn(LN(x)); x + FFN(LN(x))   (pre-LayerNorm)
                  normalization
  LM head with one logit per vocab token      -> GPT.lm_head (d -> V), weights tied to tok_emb
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 2048
    context_length: int = 256      # maximum sequence length T
    n_layer: int = 6               # number of decoder blocks
    n_head: int = 6                # attention heads per block
    d_model: int = 384             # embedding / residual stream width
    d_ff: int = 1536               # feed-forward hidden width (4 * d_model)
    dropout: float = 0.1
    tie_weights: bool = True       # LM head shares the token embedding matrix

    def to_dict(self):
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head, self.d_head = cfg.n_head, cfg.d_model // cfg.n_head
        # Learned projection matrices. Each is d_model x d_model and holds all heads side by side.
        self.W_Q = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_K = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_V = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_O = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # mask[i, j] = True where position i may attend to position j (j <= i)
        mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.context_length, cfg.context_length), persistent=False)
        self.last_attn = None          # set when store_attn=True (for analysis / figures)
        self.store_attn = False

    def forward(self, x):
        B, T, D = x.shape
        # (B, T, D) -> (B, n_head, T, d_head)
        q = self.W_Q(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.W_K(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.W_V(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        # scaled dot-product scores: (B, n_head, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
        if self.store_attn:
            self.last_attn = attn.detach()
        attn = self.attn_drop(attn)
        out = attn @ v                                         # (B, n_head, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, D)   # concatenate heads
        return self.resid_drop(self.W_O(out))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """Pre-LayerNorm decoder block: x = x + Attn(LN(x)); x = x + FFN(LN(x))."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))     # residual connection around attention
        x = x + self.ffn(self.ln2(x))      # residual connection around feed-forward
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)       # V x d
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)   # T x d (learned positions)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # d -> V logits
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight                  # same V x d matrix
        self.apply(self._init_weights)
        # GPT-2 style: scale residual output projections by 1/sqrt(2 * n_layer)
        for name, p in self.named_parameters():
            if name.endswith("W_O.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, by_component=False):
        total = sum(p.numel() for p in self.parameters())     # tied weights counted once
        if not by_component:
            return total
        c = self.cfg
        per_block = sum(p.numel() for p in self.blocks[0].parameters())
        attn = sum(p.numel() for p in self.blocks[0].attn.parameters())
        ffn = sum(p.numel() for p in self.blocks[0].ffn.parameters())
        return {
            "token_embedding (V x d, shared with LM head)": c.vocab_size * c.d_model,
            "position_embedding (T x d)": c.context_length * c.d_model,
            f"attention W_Q/W_K/W_V/W_O per block (x{c.n_layer})": attn,
            f"feed-forward per block (x{c.n_layer})": ffn,
            f"layer norms per block (x{c.n_layer})": per_block - attn - ffn,
            "all decoder blocks": per_block * c.n_layer,
            "final layer norm": sum(p.numel() for p in self.ln_f.parameters()),
            "LM head (tied: 0 extra)" if c.tie_weights else "LM head (d x V)": 0 if c.tie_weights else c.vocab_size * c.d_model,
            "total": total,
        }

    def forward(self, idx, targets=None, loss_mask=None):
        """idx: (B, T) token ids. targets: (B, T) next-token ids. loss_mask: (B, T) 1 where the
        loss counts (used in fine-tuning to train only on answer tokens). Returns (logits, loss)."""
        B, T = idx.shape
        assert T <= self.cfg.context_length, f"sequence length {T} > context length {self.cfg.context_length}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))      # (B, T, d)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))                       # (B, T, V): one logit per vocab token
        loss = None
        if targets is not None:
            per_tok = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1), reduction="none")
            if loss_mask is None:
                loss = per_tok.mean()
            else:
                m = loss_mask.view(-1).float()
                loss = (per_tok * m).sum() / m.sum().clamp(min=1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, stop_ids=(), generator=None):
        """Autoregressive sampling. temperature=0 means greedy decoding."""
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.context_length:]               # crop to the context window
            logits, _ = self(ctx)
            logits = logits[:, -1, :].float()                     # next-token logits
            if temperature == 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove = cum - torch.softmax(sorted_logits, dim=-1) > top_p   # keep the first token over p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_ids and nxt.item() in stop_ids:
                break
        return idx


def load_checkpoint(path, device="cpu"):
    """Load a saved model for inference. Returns (model, config dict, full checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt
