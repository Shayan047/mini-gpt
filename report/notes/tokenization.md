# Tokenization and Vocabulary (report notes, spec section 3)

## Strategy: byte-level BPE, implemented from scratch

- **Why sub-word (BPE) and not words or characters?**
  - **Word-level** maps any word not in the vocabulary to `<unk>`. That would make every question about an unseen item unanswerable, and this project's research question is exactly about unseen items.
  - **Character-level** never meets an unknown symbol, but makes sequences about 4× longer, so the same context window covers far less text.
  - **BPE** keeps frequent words as single tokens and splits rare or unseen words into smaller pieces.
- **Why byte-level?** Text is first converted to UTF-8 bytes, so the 256 byte tokens can represent any string: typos, emoji, accented letters. There is no `<unk>` token at all.
- **Implementation:** [src/tokenizer.py](../../src/tokenizer.py), plain Python with no tokenizer library.
- **Training data:** the training split only, so validation and test text does not influence the vocabulary.

## Building the vocabulary and assigning IDs

1. **Pre-tokenization.** A GPT-2-style regex splits text into chunks: words with their leading space, numbers, runs of punctuation, and whitespace. Merges never cross chunk boundaries, so a token never spans two words.
2. **Starting vocabulary.** IDs 0–3 are the special tokens, and IDs 4–259 are the 256 single bytes.
3. **Merge loop.** Count every adjacent token pair across the corpus, weighted by how often each chunk occurs. Merge the most frequent pair into a new token and give it the next free ID. Repeat until the vocabulary reaches 2,048, which takes 1,788 merges. Ties are broken deterministically, so training is reproducible.
4. **Saved file.** The ordered list of merges is stored in `data/tokenizer/tokenizer.json` and is all that is needed to rebuild the vocabulary. A readable `vocab.txt` lists each ID next to its token.

| IDs | Content |
|---|---|
| 0 | `<\|endoftext\|>`: document separator and end of an answer |
| 1 | `<\|pad\|>`: padding for fine-tuning batches |
| 2 | `<\|user\|>`: marks a question in fine-tuning |
| 3 | `<\|assistant\|>`: marks an answer in fine-tuning |
| 4–259 | the 256 single bytes |
| 260–2047 | the 1,788 learned merges, in the order they were learned |

The first merges learned are common letter pairs and short words (`·t`, `in`, `he`, `·the`). The last are whole domain words. The full list is in `results/samples/tokenization_examples.md`.

## Encoding and decoding

- **`encode(text)`**
  1. Split the text into chunks.
  2. Convert each chunk to byte IDs.
  3. Repeatedly apply the learned merge with the lowest rank (the earliest learned) until no merge applies.
  4. Cache the result for each distinct chunk.
- **Special-token safety:** strings like `<|assistant|>` become special IDs only when `allow_special=True`. That flag is used when building the fine-tuning data, never on user input, so a user cannot inject them.
- **`decode(ids)`:** concatenate the bytes of each token and decode as UTF-8.
- **Verified:** decode(encode(x)) == x for all 6,789 documents, plus unusual strings (typos, Unicode, emoji, repeated whitespace). Tests are in [tests/test_tokenizer.py](../../tests/test_tokenizer.py).

## Corpus size in tokens (spec requirement)

| Split | Documents | Tokens |
|---|---:|---:|
| train | 6,135 | 1,184,188 |
| val | 327 | 62,115 |
| test | 327 | 64,219 |
| **total** | **6,789** | **1,310,522** |

- Each document ends with one `<|endoftext|>`, and those are included in the counts.
- Compression: about 3.9 characters per token, or 1.48 tokens per word.
- Source: `results/tables/token_stats.md`.

## Choosing the vocabulary size (2,048)

The plan started at about 4k. Measurement changed that: the training split contains only about 3,400 distinct words, so a 4,096 vocabulary becomes almost word-level.
- 90% of word types become single tokens.
- About 1,000 tokens occur fewer than 50 times.
- Their embeddings would receive very few training updates.

| Vocab | Train tokens | Rarest merge seen | Val word types that are one token |
|---:|---:|---:|---:|
| 1,024 | 1,469,671 | 553× | 13% |
| **2,048** | **1,178,053** | **147×** | **37%** |
| 3,072 | 1,083,089 | 52× | 66% |
| 4,096 | 1,049,247 | 19× | 90% |

(The train-token figure for 2,048 differs slightly from the 1,184,188 in the table above, because the counts above also include one `<|endoftext|>` per document.)

At 2,048, every learned token occurs at least 147 times, and related words share pieces. The cost is sequences about 12% longer. Source: `results/tables/vocab_size_comparison.md`.

**Observation for the research question:** held-out item names never appear in training, so they break into unfamiliar fragments.

| Item name | Pieces |
|---|---|
| avocado pits | `·a v o c ad o ·p it s` (9 pieces) |
| wireless routers | `·w i re less ·r out ers` |
| plastic hummus tubs | `·plastic ·h um mu s ·tubs` |

The model can generalize to these items only through familiar pieces such as `·plastic`, `·glass`, `·batteries`, `·milk` and `·cartons`. These are the material words the material rules are built on.

## How vocabulary size sets the embedding matrix and the LM head

Let V = vocabulary size (2,048) and d = model width (`d_model`, chosen in the model-design step).

- **Token embedding matrix E has shape V × d.** Row *i* is the learned vector for token ID *i*. A token ID simply selects one row. One row per token means the number of rows equals V.
- **LM head W_out has shape d × V**, plus an optional bias of length V. It maps each position's final d-dimensional hidden state to **V logits**, one score per vocabulary token. The softmax over these V scores is the next-token distribution, so the number of outputs must equal V. Cross-entropy loss compares it to the true next token ID.
- **Parameter count:** each of the two matrices has V·d parameters.

| d | V·d (each matrix) |
|---:|---:|
| 256 | 524,288 |
| 384 | 786,432 |

- **Weight tying.** If the LM head reuses the embedding matrix (W_out = Eᵀ), the two share these parameters.
- **Effect of vocabulary size:** doubling V doubles both matrices and the cost of the final softmax. This is another reason a modest V suits a small model.
