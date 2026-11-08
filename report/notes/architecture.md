# Model Architecture (report notes, spec section 4)

The model is a decoder-only Transformer in the GPT style. It is built from basic PyTorch layers (`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`) in [src/model.py](../../src/model.py), and is trained from random initialization with no pretrained weights.

## Configuration (chosen for a free Colab T4 GPU)

| Hyperparameter | Value | Reason |
|---|---|---|
| Vocabulary size V | 2,048 | from the tokenizer step |
| Context length T | 256 tokens | covers most documents and every QA pair; attention cost grows with T² |
| Decoder blocks | 6 | |
| Attention heads | 6 (each d_head = 64) | |
| Model width d_model | 384 | |
| Feed-forward width | 1,536 (4 × d_model) | the standard ratio |
| Dropout | 0.1 | the model is large relative to a 1.2M-token corpus |
| Positional information | learned position embeddings (T × d) | simple, and enough for a fixed maximum length |
| Normalization | pre-LayerNorm | trains more stably than post-norm, with no learning-rate tricks needed |
| Activation | GELU | |
| LM head | tied to the token embedding | saves V·d = 786k parameters; input and output share one token representation |

## Data flow for a sequence of T token IDs

1. **Embedding:** x = TokEmb[ids] + PosEmb[0..T−1], giving a (T × 384) matrix, followed by dropout.
2. **Six decoder blocks.** Each block applies:
   - x = x + MultiHeadCausalAttention(LayerNorm(x)), the first residual connection
   - x = x + FFN(LayerNorm(x)), the second residual connection
3. **Final LayerNorm, then the LM head:** logits = LN(x) · Eᵀ, with shape (T × 2048). That is one logit per vocabulary token at every position.

## Causal multi-head self-attention

- **Projections.** For input X (T × d):
  - Q = X·W_Q, K = X·W_K, V = X·W_V
  - Each W is a separate 384 × 384 learned matrix, `nn.Linear` with bias.
  - Each result is split into 6 heads of 64 dimensions.
- **Per head:** scores = Q_h·K_hᵀ / √64, a (T × T) matrix.
- **Causal mask:** a lower-triangular boolean matrix. Positions j > i (the future) are set to −∞ before the softmax, so after the softmax they get exactly zero weight. Token i can attend only to tokens 0..i.
- **Weights and outputs:** weights = softmax(scores), row by row; head_h = weights·V_h.
- **Combining heads:** concatenate the 6 heads back to (T × 384) and project with W_O (384 × 384).
- **Implementation choice:** attention is written out explicitly rather than with PyTorch's fused attention function, so every step above appears in the code. Attention weights can be stored for visualization (`store_attn=True`).

## Parameter count (6 layers, d = 384, V = 2,048, T = 256)

| Component | Parameters |
|---|---:|
| Token embedding (V × d), shared with LM head | 786,432 |
| Position embedding (T × d) | 98,304 |
| Attention W_Q, W_K, W_V, W_O per block (4 × (384·384 + 384)) | 591,360 |
| Feed-forward per block (384·1536 + 1536 + 1536·384 + 384) | 1,181,568 |
| LayerNorms per block (2 × 2·384) | 1,536 |
| One decoder block | 1,774,464 |
| All 6 blocks | 10,646,784 |
| Final LayerNorm | 768 |
| LM head | 0 extra (tied) |
| **Total** | **11,532,288 (≈ 11.5M)** |

## Vocabulary and model dimensions

V = 2,048 fixes two shapes:
- the **embedding matrix** at 2,048 × 384 = 786,432 parameters, one row per token
- the **LM head output** at 2,048 logits per position, which the cross-entropy loss is computed over

With weight tying, the LM head's 384 × 2,048 weight matrix is the transposed embedding matrix, so the same 786k parameters do both jobs.

## Verified behaviour

[tests/test_model.py](../../tests/test_model.py) checks the following:

1. Changing future tokens does not change earlier logits (causal mask).
2. Attention weights are exactly zero above the diagonal, and each row sums to 1.
3. Logits have shape (B, T, V).
4. Initial loss is about ln V = 7.62, i.e. a near-uniform prediction.
5. The LM head shares memory with the embedding, and W_Q, W_K, W_V and W_O exist with shape d × d.
6. The model can overfit a single batch.
7. The fine-tuning loss mask counts only the masked positions.
8. Greedy generation is deterministic.
