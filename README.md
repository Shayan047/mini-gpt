# Mini GPT: Tidewell Residential Waste Guide

CS 5370 (Fall 2026) group project. This is a small **decoder-only Transformer (11.5M parameters)**, built and trained from scratch:

1. It was **pretrained** on a synthetic corpus about the residential waste guide of a fictional town, *Tidewell*.
2. It was then **fine-tuned** to answer residents' questions, such as *"Where do pizza boxes go?"*.

Both trained models are included in this repository, so **no training is needed to run them**.

---

## 1. Quick start: run the trained models

### Requirements

| | |
|---|---|
| Python | 3.10 or newer (tested with 3.12 on Colab and 3.14 on Windows) |
| Packages | `torch` (≥ 2.1, CPU build is enough), `numpy`; `matplotlib` only for plots |
| Hardware for inference | **any CPU** with about 1 GB of free RAM. No GPU needed; an answer takes about 0.5 s on a laptop CPU |
| Hardware used for training | Google Colab, free tier, NVIDIA **Tesla T4** (15 GB). Pretraining took 7.5 min, fine-tuning 1 min |

```bash
git clone https://github.com/Shayan047/mini-gpt.git
cd mini-gpt
pip install torch numpy matplotlib        # CPU-only build: add  --index-url https://download.pytorch.org/whl/cpu
```

All commands are run from the repository root.

### Ask the fine-tuned model a question

```bash
python -m src.generate --ckpt models/finetuned.pt --qa --temperature 0 --prompt "Where do pizza boxes go?"
```

```text
They are accepted in the green organics cart. Scrape them clean of food.
```

### Interactive session (type questions, then an empty line or `quit` to exit)

```bash
python -m src.generate --ckpt models/finetuned.pt --qa --temperature 0
```

### Continue text with the pretrained model

```bash
python -m src.generate --ckpt models/pretrained.pt --temperature 0 --max-new-tokens 50 --prompt "Glass bottles and jars go in the"
```

```text
Glass bottles and jars go in the blue cart. Empty them fully first. Rinse them with a little water. Put the cap or lid back on. ...
```

### Generation options

| Option | Meaning | Default |
|---|---|---|
| `--ckpt` | `models/finetuned.pt` (question answering) or `models/pretrained.pt` (text continuation) | required |
| `--qa` | wrap the prompt as a question in the fine-tuning format. **Use it with the fine-tuned model** | off |
| `--prompt` | prompt text; leave it out for an interactive session | none |
| `--temperature` | 0 = greedy (deterministic); higher values give more varied output | 0.7 |
| `--top-k`, `--top-p` | sampling filters | 50, off |
| `--max-new-tokens` | maximum length of the generated text | 120 |
| `--seed` | fix the random seed for reproducible sampling | none |
| `--device` | `cpu` or `cuda` | auto |

### Example inputs

All outputs below come from the fine-tuned model with `--temperature 0`.

| Question | Answer |
|---|---|
| Where do pizza boxes go? | They are accepted in the green organics cart. Scrape them clean of food. |
| Can I put batteries in the garbage? | Batteries of every kind go to a household hazardous waste drop-off, never in a cart. For example: household batteries, car batteries. |
| What is the capital of France? | I can only answer questions about household waste, recycling and collection in Tidewell. |

Other questions to try:
- *How often is the green cart collected?*
- *Where do plastic bags go?*
- *What do I do with old paint?*
- *I live in an apartment. What do I do with old furniture?*
- *What counts as household hazardous waste?*

**Tips for best results:**
- Ask **one question at a time**.
- Use ordinary spelling.
- The model only knows the fictional Tidewell guide. Questions outside it, or about items it never saw, are often declined or answered wrongly (see section 4).

### Required files

| File | Purpose |
|---|---|
| `models/finetuned.pt` | fine-tuned model weights and configuration (training step 1,000) |
| `models/pretrained.pt` | pretrained model weights and configuration (training step 1,800) |
| `models/model_info.json` | configurations, prompt formats and SHA-256 checksums of both models |
| `data/tokenizer/tokenizer.json` | byte-level BPE tokenizer (vocabulary of 2,048), required by both models |
| `src/model.py`, `src/tokenizer.py`, `src/generate.py` | model definition, tokenizer and the command-line tool |

Each checkpoint stores its model configuration (layers, heads, widths, vocabulary size, context length), so `src.model.load_checkpoint(path)` rebuilds the exact architecture. Checkpoints saved on a GPU are mapped to the CPU automatically.

To use the models from Python:

```python
from src.model import load_checkpoint
from src.tokenizer import BPETokenizer
import torch

tok = BPETokenizer.load("data/tokenizer/tokenizer.json")
model, cfg, _ = load_checkpoint("models/finetuned.pt", device="cpu")
ids = torch.tensor([tok.encode("<|user|> Where do glass jars go?<|assistant|>", allow_special=True)])
out = model.generate(ids, max_new_tokens=60, temperature=0, stop_ids=(tok.eot_id,))
print(tok.decode(out[0, ids.shape[1]:].tolist(), skip_special=True))
```

---

## 2. What is in this project

| Stage | What was done | Where |
|---|---|---|
| Fictional domain | *Tidewell Waste & Recycling Services*, limited to its **Residential Waste Guide**. It covers 8 disposal streams, 20 material rules, 167 items (20 held out for testing) and 55 facts on 13 topics. Facts are deliberately generic: no addresses, phone numbers or times | `data/kb/` |
| Synthetic corpus | 6,789 documents in 17 generated document types plus 149 hand-written ones, for **1,310,522 tokens**. Generated by Claude Code and checked for consistency with the knowledge base and for leakage | `data/corpus/`, `data/DATA_GENERATION.md` |
| Tokenizer | byte-level BPE written from scratch, vocabulary 2,048 | `src/tokenizer.py` |
| Model | GPT-style decoder: learned token and position embeddings, 6 blocks, 6 attention heads with explicit W_Q/W_K/W_V/W_O, causal mask, pre-LayerNorm, GELU feed-forward, LM head tied to the embeddings. 11,532,288 parameters, context of 256 tokens | `src/model.py` |
| Pretraining | next-token prediction, 3,000 steps (24.6M tokens); best validation loss 0.859 (perplexity 2.36) at step 1,800 | `src/train.py`, `notebooks/01_pretrain.ipynb` |
| Fine-tuning | 11,056 question–answer pairs; loss on answer tokens only | `src/finetune.py`, `notebooks/02_finetune.ipynb` |
| Evaluation | 654 held-out questions, identical for both models, plus knowledge probes across 11 pretraining checkpoints | `src/evaluate_pretrain.py`, `src/evaluate_qa.py` |
| Research question | *How well does a small Transformer generalize waste-sorting rules to unseen items and unseen question wordings, and how does supervised fine-tuning change that?* Includes an ablation without the decline examples | `report/notes/`, `notebooks/03_sft_ablation.ipynb` |

### Main results

| | Pretrained (`Q:/A:` prompt) | Fine-tuned |
|---|---:|---:|
| All 654 evaluation questions | 76.3% | **82.3%** |
| Known items, unseen question wording | 100% | 99.7% |
| Yes/no questions | 66.7% | **85.0%** |
| Items never seen in training | 50.0% | 38.3% |
| Off-topic questions politely declined | 0% | **75%** |

Full tables and figures are in `results/tables/` and `results/figures/`.

---

## 3. Reproducing everything from scratch

Data generation and the tokenizer need only the Python standard library. Training needs a GPU; the notebooks are written for **Google Colab (free T4)**.

```bash
# 1. data (deterministic, a few seconds)
python src/kb.py                              # validate the knowledge base
python -m src.datagen.generate_corpus         # corpus -> data/corpus/*.jsonl
python -m src.datagen.check_corpus            # leakage / consistency / duplicate checks
python -m src.train_tokenizer                 # BPE -> data/tokenizer/, data/tokens/*.bin
python -m src.datagen.build_sft               # fine-tuning + evaluation sets -> data/sft/, data/eval/

# 2. training (GPU): notebooks/01_pretrain.ipynb, then notebooks/02_finetune.ipynb
#    or directly:
python -m src.train --out-dir checkpoints/pretrain
python -m src.finetune --init checkpoints/pretrain/ckpt_best.pt --out-dir checkpoints/sft

# 3. evaluation (CPU is fine)
python -m src.evaluate_pretrain               # loss/perplexity + knowledge probes over checkpoints
python -m src.evaluate_qa --pretrained models/pretrained.pt --finetuned models/finetuned.pt
python -m src.significance                    # paired McNemar tests

# tests
python -m tests.test_tokenizer && python -m tests.test_model && python -m tests.test_probes && python -m tests.test_finetune
```

With the fixed seeds, training is deterministic: two pretraining runs on a T4 produced identical losses.

---

## 4. Known limitations

- **Domain only.** The model knows only the fictional Tidewell guide and has no general world knowledge.
- **Unseen items.** Items that never appeared in training are answered by analogy with known items that share words with them, which is often wrong. Examples: "watermelon rinds" → *hot water tanks*; "bookshelves" → *old books*.
- **Over-cautious declines.** About 20% of casually phrased questions are declined. An ablation shows these declines mostly fall on questions the model would have answered wrongly anyway.
- **Fragile inputs.** Typos (*"coffe grounds"*) and two-part questions can lead to wrong answers.
- **Yes/no slips.** Yes/no questions get the right lead word about 85% of the time. Example: *"Can I put plastic bags in the blue cart?"* → *"**Yes.** Plastic bags are not recyclable, so they go in the black garbage cart"*. The bin is right, but the yes is wrong.
- **Forgetting.** Fine-tuning made the model worse at general document text; its loss on the pretraining corpus rose by 53%.

---

## 5. Repository layout

| Path | Contents |
|---|---|
| `models/` | **the two submitted models** + `model_info.json` |
| `data/kb/` | fictional knowledge base (`knowledge_base.json`, readable `knowledge_base.md`) |
| `data/seed_docs/` | 149 hand-written seed documents |
| `data/corpus/`, `data/tokens/` | pretraining corpus (JSONL) and its tokenized form |
| `data/tokenizer/` | BPE tokenizer (`tokenizer.json`, readable `vocab.txt`) |
| `data/sft/`, `data/eval/` | fine-tuning pairs and the 654-question evaluation set |
| `data/DATA_GENERATION.md` | data-generation methodology |
| `src/` | all code: data generation, tokenizer, model, training, evaluation, generation |
| `notebooks/` | Colab notebooks: 01 pretraining, 02 fine-tuning, 03 ablation |
| `tests/` | tests for the tokenizer, model, probes and fine-tuning batches |
| `results/` | loss curves, training logs, generated samples, evaluation tables, screenshots |
| `report/` | report drafts and notes |
