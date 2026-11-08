"""Model tests. Run: python -m tests.test_model"""
import math

import torch

from src.model import GPT, GPTConfig

CFG = GPTConfig(vocab_size=2048, context_length=32, n_layer=2, n_head=4, d_model=64, d_ff=256, dropout=0.0)


def test_shapes():
    m = GPT(CFG)
    x = torch.randint(0, CFG.vocab_size, (3, 20))
    logits, loss = m(x, x)
    assert logits.shape == (3, 20, CFG.vocab_size)        # one logit per vocabulary token, per position
    assert loss.ndim == 0


def test_causal_mask_blocks_future():
    """Changing tokens at positions >= t must not change the logits at positions < t."""
    torch.manual_seed(0)
    m = GPT(CFG).eval()
    x = torch.randint(0, CFG.vocab_size, (1, 16))
    y = x.clone()
    y[0, 10:] = torch.randint(0, CFG.vocab_size, (6,))
    a, _ = m(x)
    b, _ = m(y)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-6)
    assert not torch.allclose(a[0, 10:], b[0, 10:])


def test_attention_weights_are_lower_triangular():
    m = GPT(CFG).eval()
    att = m.blocks[0].attn
    att.store_attn = True
    m(torch.randint(0, CFG.vocab_size, (1, 12)))
    w = att.last_attn[0]                                     # (heads, T, T)
    assert torch.all(torch.triu(w, diagonal=1) == 0)         # nothing above the diagonal
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)


def test_initial_loss_near_uniform():
    torch.manual_seed(0)
    m = GPT(CFG)
    x = torch.randint(0, CFG.vocab_size, (8, 32))
    y = torch.randint(0, CFG.vocab_size, (8, 32))   # independent targets (x as its own target is biased by tied weights)
    _, loss = m(x, y)
    assert abs(loss.item() - math.log(CFG.vocab_size)) < 0.3, loss.item()


def test_weight_tying_and_projections_exist():
    m = GPT(CFG)
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()
    a = m.blocks[0].attn
    for W in (a.W_Q, a.W_K, a.W_V, a.W_O):
        assert W.weight.shape == (CFG.d_model, CFG.d_model)


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    m = GPT(CFG)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, CFG.vocab_size, (4, 32))
    y = torch.roll(x, -1, dims=1)
    for _ in range(150):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, loss.item()


def test_loss_mask_only_counts_masked_positions():
    m = GPT(CFG).eval()
    x = torch.randint(0, CFG.vocab_size, (1, 8))
    mask = torch.zeros(1, 8)
    mask[0, 5:] = 1
    logits, masked = m(x, x, loss_mask=mask)
    full = torch.nn.functional.cross_entropy(logits[0, 5:], x[0, 5:])
    assert torch.allclose(masked, full, atol=1e-5)


def test_generate_greedy_deterministic():
    m = GPT(CFG)
    x = torch.randint(0, CFG.vocab_size, (1, 5))
    a = m.generate(x, 10, temperature=0)
    b = m.generate(x, 10, temperature=0)
    assert torch.equal(a, b) and a.shape == (1, 15)


def test_kv_cache_matches_full_recompute():
    """Generation with the key/value cache must equal recomputing the whole sequence each step,
    including past the context window (sliding-window fallback)."""
    torch.manual_seed(0)
    m = GPT(CFG).eval()
    x = torch.randint(0, CFG.vocab_size, (1, 6))
    fast = m.generate(x, 40, temperature=0)                    # 46 tokens > context 32
    slow = x
    for _ in range(40):
        logits, _ = m(slow[:, -CFG.context_length:])
        slow = torch.cat([slow, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(fast, slow)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
