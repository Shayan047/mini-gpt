# CS 5370 Term Project — Mini GPT Tracker

Source: `Term Project.pdf` (CS 5370, Fall 2026 — "Building and Investigating a Small Transformer Language Model").
Fresh redo — do not copy from the earlier Claude-assisted attempt.

Legend: `[ ]` todo · `[x]` done · **(OBS)** = something to observe, measure, or record along the way (feeds the report/results)

---

## 0. Setup & environment
- [x] Create project structure (`data/`, `src/`, `notebooks/`, `checkpoints/`, `results/`, `report/`)
- [x] Initialize git repo
- [ ] Set up Python virtual env + `requirements.txt` (torch, numpy, matplotlib, tqdm; tokenizer lib if BPE)
- [ ] Training/eval code targets **Google Colab (free T4 GPU)** — PyTorch is preinstalled there; local machine only runs data generation (pure Python, no torch needed)
- [ ] **(OBS)** Record Colab hardware actually used (GPU type, VRAM, session length) — drives model size (spec §4) and goes in README "hardware requirements" (§9g). Local laptop: no GPU, 16 GB RAM
- [ ] Checkpoint-resume support (free Colab sessions can disconnect)
- [ ] Fix random seeds everywhere for reproducibility

## 1. Fictional domain & knowledge base (spec §2)
- [x] Invent a fictional organization → **Tidewell Waste & Recycling Services** (fictional town of Tidewell)
- [x] Narrow to ONE department → **Residential Waste Guide**: how residents sort, prepare and dispose of household waste
- [x] Write a structured knowledge base → [data/kb/knowledge_base.json](data/kb/knowledge_base.json), validator [src/kb.py](src/kb.py)
- [x] User review of the KB content (bin colours, schedule, rules) — approved
- [x] Render a readable markdown version of the KB → [data/kb/knowledge_base.md](data/kb/knowledge_base.md) (`python src/kb_to_md.py`)
- [x] Keep fact count small and fixed — every item, rule (R01–R20) and topic fact (S/P/M/A/D/B/F/U/H/O/V/N/W ids) has an ID
- [x] **(OBS)** KB size: 8 streams, 20 material rules, 27 prep codes, 167 items (147 core + 20 held-out), 541 item names/aliases, 13 topics with 55 facts

## 2. Synthetic corpus generation (spec §2)
- [x] Write the data-generation methodology → [data/DATA_GENERATION.md](data/DATA_GENERATION.md)
- [x] Generate documents in varied types/styles → 17 generated doc types ([src/datagen/](src/datagen/)) + 149 hand-written seed docs ([data/seed_docs/](data/seed_docs/))
- [x] Each fact repeated many times in different wording and contexts (balanced cyclers; paraphrase banks in [lexicon.py](src/datagen/lexicon.py), [questions.py](src/datagen/questions.py))
- [x] Consistency check → [src/datagen/check_corpus.py](src/datagen/check_corpus.py): 0 held-out leaks, 0 specific details, 0 seed-doc conflicts, 0 duplicates
- [x] Clean/normalize text (tidy() in expand.py; one doc per JSONL line)
- [x] **(OBS)** Corpus stats: train 6,135 docs / 802k words; val 327 / 42k; test 327 / 43k → `results/tables/corpus_stats.md`
- [x] **(OBS)** Per-fact coverage: items 203–689 mentions (median 245); topic facts 82–200 docs; rules 124–193 docs → `results/tables/item_coverage.csv`, `fact_coverage.csv`
- [x] Generation sources saved (paraphrase banks + generator + seed docs are the "prompts"; corpus reproducible from fixed seeds)

## 3. Data splits (spec §2)
- [x] Split into train / validation / test by document (generated per split with separate seeds; seeds docs 90/5/5)
- [x] Verify no evaluation document is duplicated in training (exact-duplicate check = 0; ~15% of templates eval-only)
- [ ] **(OBS)** 72–75% of val/test *sentences* also occur in train (short fact sentences recur) → val loss ≈ in-domain fluency; generalization is measured by the QA eval set. Mention in report limitations
- [ ] **(OBS)** Record split sizes (docs ✔ above; tokens after tokenizer)

## 4. Tokenization & vocabulary (spec §3)
- [x] Choose tokenizer strategy → byte-level BPE, implemented from scratch ([src/tokenizer.py](src/tokenizer.py)); justification in [report/notes/tokenization.md](report/notes/tokenization.md)
- [x] Build vocabulary, assign token IDs, add special tokens: 0 `<|endoftext|>`, 1 `<|pad|>`, 2 `<|user|>`, 3 `<|assistant|>`; 4–259 bytes; 260–2047 merges
- [x] Implement `encode()` / `decode()`; round-trip verified on all 6,789 docs + tests ([tests/test_tokenizer.py](tests/test_tokenizer.py), 5/5 pass)
- [x] Save tokenizer/vocab → `data/tokenizer/tokenizer.json`, `vocab.txt`; encoded splits → `data/tokens/{train,val,test}.bin` (uint16) + `meta.json`
- [x] **(OBS)** Vocab size **2,048** (changed from planned ~4k after measurement, user approved). **Corpus size: 1,310,522 tokens** (train 1,184,188 / val 62,115 / test 64,219)
- [x] Write explanation: vocab size → embedding matrix V×d and LM head d×V (V logits) → in tokenization notes; fill in final d once model config is chosen
- [ ] SFT prompt format fixed as `<|user|> question<|assistant|> answer<|endoftext|>` (no space before special tokens — a space produces a stray `·` token)

## 5. Transformer implementation (spec §4) — from scratch in PyTorch
- [ ] Token embedding layer
- [ ] Positional information (learned or sinusoidal) — justify choice
- [ ] Causal multi-head self-attention with explicit W_Q, W_K, W_V, W_O
- [ ] Causal mask (lower-triangular) — unit test that future tokens don't affect earlier outputs
- [ ] Feed-forward network (MLP) per block
- [ ] Residual connections + LayerNorm (pre-norm or post-norm — justify)
- [ ] Stack N decoder blocks
- [ ] LM head → one logit per vocab token
- [ ] Config object (n_layers, n_heads, d_model, context_len, dropout, vocab_size) saved alongside checkpoints
- [ ] Choose hyperparameters suited to CPU-only hardware
- [ ] **(OBS)** Record parameter count (total and per component)
- [ ] Sanity check: initial loss ≈ ln(V); model can overfit a tiny batch

## 6. Pretraining — next-token prediction (spec §5)
- [ ] Data loader producing (input, target-shifted-by-one) windows
- [ ] Cross-entropy loss, optimizer (AdamW), LR schedule (warmup + decay), grad clipping
- [ ] Train from random init
- [ ] Log training and validation loss at regular intervals (to file — CSV/JSON)
- [ ] Save checkpoints periodically (several intermediate ones help the research question) + best/final
- [ ] **(OBS)** Record training time, steps/epochs, tokens seen, final train/val loss, val perplexity
- [ ] **(OBS)** Watch for overfitting (val loss diverging from train) — note when/if it happens
- [ ] **(OBS)** Save sample generations at each checkpoint during training
- [ ] Plot train vs. validation loss curves

## 7. Text generation & evaluation (spec §6)
- [ ] Autoregressive generation function (greedy, temperature, top-k / top-p)
- [ ] Evaluate on held-out test set: test loss / perplexity
- [ ] Prompt tests: plausible domain text, completion of familiar statements, reproduction of KB relationships
- [ ] Build a fact-completion probe set from the KB (e.g. "Bags must be reported within ___") and score accuracy
- [ ] **(OBS)** Document successful outputs
- [ ] **(OBS)** Document unsuccessful outputs (hallucinations, wrong facts, repetition, incoherence) with possible explanations

## 8. Supervised fine-tuning for QA (spec §7)
- [ ] Create QA dataset derived from the same KB (multiple phrasings per fact)
- [ ] Define QA prompt format (e.g. `Question: ... Answer: ...<eos>`)
- [ ] Reserve a separate QA **evaluation set** incl. unseen question phrasings — never used in fine-tuning
- [ ] Verify no overlap between SFT train questions and eval questions
- [ ] Fine-tune starting from the pretrained checkpoint (short run, lower LR; optionally mask loss to answer tokens only)
- [ ] Log SFT train/val loss; save SFT checkpoint
- [ ] Run the SAME eval questions on the pretrained and the fine-tuned model
- [ ] Define scoring (exact match / key-fact match / manual rubric)
- [ ] **(OBS)** Before vs. after comparison table (accuracy, sample answers)
- [ ] **(OBS)** What improved, what didn't, limitations (e.g. catastrophic forgetting of general text, memorized phrasing, failures on unseen formulations)

## 9. Research investigation (spec §8)
- [x] Pick ONE research question → generalization to unseen items & phrasings, before vs after SFT
- [ ] Design the experiment using saved checkpoints / held-out data / controlled generation settings
- [ ] Define metrics and run the experiment
- [ ] **(OBS)** Collect evidence (tables, plots, sample outputs)
- [ ] Discuss findings, limitations, possible explanations

## 10. Deliverables (spec §9)
- [ ] **a) Research report (5–7 pages):** dataset, research question, architecture, training procedure, evaluation, findings, limitations
- [ ] **b) Synthetic dataset:** corpus, knowledge base, data-generation methodology
- [ ] **c) Source code:** data processing, model, training, evaluation, generation
- [ ] **d) Two checkpoints:** after pretraining + after SFT, plus tokenizer/vocab and config to load both
- [ ] **e) Results:** loss curves, training logs, generated samples, experimental findings
- [ ] **f) Demo video (3–5 min):** live screen recording loading the model and generating from several prompts, at least one prompt NOT from training examples
- [ ] **g) README:** install requirements, hardware requirements, inference commands, required files, example inputs — runnable without retraining
- [ ] **h) Slide presentation + Q&A prep:** dataset, architecture, training, research question, results, conclusions; all 4 members must be able to explain every part
- [ ] **i) AI Use Statement:** tools used and how (code, design, analysis, writing) — includes Claude Code
- [ ] **j) Reproducibility check:** fresh clone/env → load both checkpoints → generate from new prompts; confirm code/tokenizer/config compatibility
- [ ] **k) Member contributions statement** for each of the 4 members

## 11. Final verification
- [ ] Run inference from a clean environment following only the README
- [ ] Confirm checkpoint file sizes are submittable
- [ ] Cross-check report numbers against logs/results files
- [ ] Everything in the report is verified (spec: you are responsible for AI-produced claims)

---

## Decisions (agreed with user, 2026-11-07)
| Decision | Choice |
|---|---|
| Compute | Google Colab, free tier (T4). Code written for Colab; checkpoint-resume required |
| Organization / department | Tidewell Waste & Recycling Services → Residential Waste Guide |
| Topics | Item sorting + preparation, collection schedule, drop-off depot, bulky item pickup, rules & fines, plus apartments, reduce/reuse, home composting, moving, seasonal, why sorting matters |
| Fact style | **Generic, not exact** — no addresses, phone numbers, staff names or clock times (earlier attempt kept repeating one memorized address). E.g. bulky items: "book a pickup, or use your building's designated bulky item area — ask your property manager" |
| Data generation | Claude Code writes the KB + large template/paraphrase banks + Python generator (free, reproducible, guaranteed KB-consistent) |
| Tokenizer | Byte-level BPE from scratch, **vocab 2,048** (revised from ~4k after measuring: corpus has only ~3.4k distinct words, 4k would be near word-level with ~1,000 rarely-seen tokens) |
| Research question | How well does a small Transformer generalize waste-sorting rules to unseen items and unseen question phrasings, and how does supervised fine-tuning change that? |
| Report assets | Every script/notebook saves its outputs for the report: `results/figures/` (plots, PNG), `results/tables/` (CSV + markdown tables), `results/samples/` (generated text), `results/logs/` (training logs), `results/screenshots/` (Colab/demo screenshots). Each asset is listed in the Report assets index below |

## Data plan (target numbers)
- **Pretraining corpus:** ~4,000 documents, ~1.2–1.5M BPE tokens (avg ~350 tokens/doc), ~14 document types (guide pages, A–Z entries, FAQs, newsletters, staff memos/training notes, customer-service emails, chat transcripts, social posts, flyers/notices, forum Q&A, school handouts, building notices, contamination reports, seasonal bulletins)
- **Exposure target:** each core item's disposal fact ~60–100 mentions in varied wording; each rule / topic fact ~40–60 mentions; frequencies balanced so no single answer dominates
- **Material rules stated explicitly and often** (e.g. "rigid plastic containers → blue cart") so the model can map unseen items through material words
- **Splits:** 90 / 5 / 5 train/val/test by document; a share of template variants reserved for val/test only, so test loss measures unseen phrasing, not memorized sentences
- **SFT set:** ~12,000 QA pairs (≈50 question phrasings per core item + topic questions), in a QA format not used in the pretraining corpus
- **Eval QA set (never used in training), same set for pretrained vs fine-tuned:** ~600 questions —
  - core items asked with question templates never used in SFT
  - topic questions with unseen phrasings
  - the 20 **held-out items** (never in corpus or SFT) → tests generalization to out-of-data items
  - ~50 hand-written free-form questions (casual wording, typos, multi-part)
- **Model size (T4):** ~10–15M params; corpus × ~15–20 epochs ≈ 25M tokens ≈ well under one Colab session
- **Honest limitation to report:** a model trained only on this corpus has no outside world knowledge; it can generalize to new items only via words it has seen (plastic, tub, glass, battery, cord…). Items with no familiar material words will be guesses.

## Report assets index
_Every figure/table/sample saved for the report gets a line here (file — what it shows — report section)._

- `results/tables/corpus_stats.md` / `.csv` — docs, words, chars per split and per document type — Dataset
- `results/tables/item_coverage.csv` — mentions of each core item in train — Dataset / factual recall analysis
- `results/tables/fact_coverage.csv` — docs using each topic fact and material rule — Dataset
- `results/tables/corpus_checks.md` — leakage, consistency, duplicate and overlap checks — Dataset (quality control)
- `results/samples/corpus_samples.md` — one example document per type — Dataset / appendix
- `data/kb/knowledge_base.md` — readable knowledge base — Dataset / appendix
- `data/DATA_GENERATION.md` — data-generation methodology — Dataset; AI Use Statement
- `results/tables/token_stats.md` / `.csv` — tokens per split, chars/token — Dataset / Tokenization (corpus size in tokens)
- `results/tables/vocab_size_comparison.md` — vocab 1k/2k/3k/4k trade-off — Tokenization (justifies 2,048)
- `results/samples/tokenization_examples.md` — how sentences, core and held-out item names are split; first/last merges — Tokenization / Research question
- `report/notes/tokenization.md` — draft text for the tokenization section incl. embedding/LM-head explanation — Tokenization

---

## Observations log
_Record findings here as work progresses (date — observation)._

- 2026-11-07 — Environment: Python 3.14.3, PyTorch not installed, no NVIDIA GPU detected, 16 GB RAM → plan for CPU-only training with a small model.
- 2026-11-07 — KB built and validated (147 core + 20 held-out items). Watch-out: core alias "plates" (ceramic → garbage) vs held-out "used paper plates" (→ organics) — a good probe of whether the model uses the material word "paper".
- 2026-11-07 — Data generation decision: option C (Claude Code writes paraphrase banks + seeded generator, plus hand-written seed docs). User approved.
- 2026-11-07 — Corpus v1 generated: 6,789 docs total (149 hand-written seeds), ~886k words. All checks pass.
- 2026-11-07 — Consistency checker caught 2 ambiguous seed sentences (mistake described without negation, e.g. "Plastic bags in the blue cart.") → reworded. Good example for report of why automated KB checks matter.
- 2026-11-07 — Manual review removed 3 seed sentences that introduced non-KB facts (balloons, pool chemical buckets, dog bed) or hinted at held-out items (bookshelf, corn, wireless). Lesson: LLM-written free text drifts from the KB even when instructed not to.
- 2026-11-07 — Exact-duplicate documents appeared at scale (short topic pages) incl. one test doc duplicating train → added cross-split dedup in generator.
- 2026-11-07 — Tokenizer: corpus has only 3,358 distinct pre-token chunks → 4,096 vocab ≈ word-level (90% of val word types single-token, rarest merge seen 19×). Switched to 2,048 (rarest merge 147×). User approved.
- 2026-11-07 — Held-out item names fragment into many pieces (e.g. "avocado pits" → 9 tokens) while material words (plastic, glass, batteries, milk, cartons) stay whole → generalization must come through those material tokens. Good evidence for the research question.
- 2026-11-07 — Data-generation docs updated: final corpus = 1,310,522 tokens.
