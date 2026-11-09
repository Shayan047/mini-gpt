# CS 5370 Term Project — Mini GPT Tracker

Source: `Term Project.pdf` (CS 5370, Fall 2026 — "Building and Investigating a Small Transformer Language Model").
Fresh redo — do not copy from the earlier Claude-assisted attempt.

Legend: `[ ]` todo · `[x]` done · **(OBS)** = something to observe, measure, or record along the way (feeds the report/results)

---

## ▶ Next session — start here (paused 2026-11-08)
1. **SFT ablation without decline examples** (user approved):
   - [x] `--exclude-categories` in `src/finetune.py` (no-decline set: 10,639 train pairs; everything else identical) + [notebooks/03_sft_ablation.ipynb](notebooks/03_sft_ablation.ipynb)
   - [x] `src/evaluate_qa.py --add LABEL CKPT` evaluates only the new model and merges it with saved answers (rescore verified identical)
   - [x] **USER:** `git push`, run notebook 03 on Colab (~1–2 min), 📸 log, download `sft_no_decline/ckpt_best.pt` → `checkpoints/sft_no_decline/ckpt_best.pt`, results → `results/`
   - [x] Evaluate, compare 4 systems (+ McNemar significance → `results/tables/qa_significance.md`, `src/significance.py`)
   - [ ] **USER DECISION:** which fine-tuned model to submit (with declines vs. no declines)
2. Then deliverables: README (install/hardware/inference/examples), demo script for the video, report draft from `report/notes/`, slides, AI Use Statement, member contributions, final reproducibility check.
- Local checkpoints: `checkpoints/pretrain/` (best = step 1800 + 10 snapshots, run 2), `checkpoints/sft/ckpt_best.pt`. Local CPU torch in `.venv`. All work committed up to the QA evaluation (not pushed by Claude).

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

## 5. Transformer implementation (spec §4) — from scratch in PyTorch → [src/model.py](src/model.py)
- [x] Token embedding layer (V×d = 2048×384)
- [x] Positional information → learned position embeddings (256×384)
- [x] Causal multi-head self-attention with explicit, separate W_Q, W_K, W_V, W_O (attention math written out, not fused)
- [x] Causal mask (lower-triangular) — tested: future tokens don't change earlier logits; attention weights zero above diagonal
- [x] Feed-forward network (384→1536→384, GELU)
- [x] Residual connections + LayerNorm → pre-LN
- [x] Stack 6 decoder blocks
- [x] LM head → 2048 logits per position, weight-tied to token embedding
- [x] Config object (GPTConfig) saved inside every checkpoint
- [x] Hyperparameters for free T4: 6 layers, 6 heads, d=384, context 256, dropout 0.1
- [x] **(OBS)** Parameter count: **11,532,288** (blocks 10.65M, token emb 0.79M, pos emb 0.10M) — matches hand calculation
- [x] Sanity checks: initial loss ≈ ln V (7.62); overfits one batch → [tests/test_model.py](tests/test_model.py), 8/8 pass (local CPU torch 2.15 in .venv)
- [x] Architecture notes for report → [report/notes/architecture.md](report/notes/architecture.md)

## 6. Pretraining — next-token prediction (spec §5) → [src/train.py](src/train.py), [notebooks/01_pretrain.ipynb](notebooks/01_pretrain.ipynb)
- [x] Data loader: random 256-token windows from uint16 memmap; targets shifted by one
- [x] Cross-entropy loss, AdamW (β 0.9/0.95, wd 0.1 on matrices only), warmup 150 + cosine 6e-4→6e-5, grad clip 1.0, fp16 AMP on GPU
- [x] Train/val loss on fixed eval windows every 150 steps → `log.csv`, `train_log.txt`
- [x] Checkpoints: `ckpt_last.pt` (resume, atomic save), `ckpt_best.pt` (lowest val), `snapshots/step_*.pt` every 500 steps
- [x] Sample generations from 6 fixed prompts (greedy + T=0.8) at every eval → `samples.jsonl` (checkpoint-evolution evidence)
- [x] Loss-curve plot script → [src/plot_losses.py](src/plot_losses.py)
- [x] CPU smoke test incl. disconnect/resume: OK
- [x] **USER:** push to GitHub, run `notebooks/01_pretrain.ipynb` on Colab T4, take 📸 screenshots, download `MyDrive/mini-gpt/results/` into repo `results/`
- [x] **(OBS)** Tesla T4, **7.3 min**, 3,000 steps (~20.7 epochs), 24.6M tokens seen. Best val **0.859 (ppl 2.36) at step 1800**; final train 0.281 / val 0.887
- [x] **(OBS)** Mild overfitting: val minimum at step 1800, then rises ~3% (0.859→0.887) while train keeps falling (0.36→0.28); train–val gap ≈0.6 nats = partial memorization of training docs. Best checkpoint (step 1800) kept as the official pretrained model
- [x] **(OBS)** Samples across checkpoints: step 150 fragments/blank lines/wrong facts → 450 grammatical but stream mix-ups (green cart described with blue-cart items) and repetition loops ("plastic straws") → 1800 fluent + factually correct + document structure → 3000 same, but Q/A prompt about batteries answered about *thermometers* (related item, not the asked one) ⇒ motivates SFT
- [x] Final loss-curve figure → `results/figures/pretrain_loss.png`

## 7. Text generation & evaluation (spec §6) → [src/evaluate_pretrain.py](src/evaluate_pretrain.py), [src/generate.py](src/generate.py), notes [report/notes/pretrain_evaluation.md](report/notes/pretrain_evaluation.md)
- [x] Autoregressive generation (greedy, temperature, top-k, top-p) + key/value cache (3× faster, identical output — tested) + CLI `src/generate.py` (single prompt / interactive / `--qa`)
- [x] Test loss / perplexity over full splits → best ckpt: val 0.863 (ppl 2.37), **test 0.801 (ppl 2.23)**
- [x] Prompt tests: familiar statements, novel framing, held-out items, off-domain → `results/samples/pretrain_generation_examples.md`
- [x] Fact probes: item→stream probe (3 frames × 147 core + 20 held-out items) and 25 fact clozes → `results/tables/pretrain_eval.md`, per-prompt CSVs; destination classifier tested (`tests/test_probes.py`)
- [x] **(OBS)** Successful outputs documented (see notes)
- [x] **(OBS)** Unsuccessful outputs documented: yes/no contradiction ("Yes! Plastic bags go in the black cart"), wrong-item answers, spurious subword associations, greedy repetition, narrative prompts, off-domain collapse
- [x] **(OBS)** Fact cloze 100% at all checkpoints; core items 90.7% (directory/FAQ 100%, "X belong…" frame 72% because of the fixed phrase "belong in the garbage, not the recycling" — prep steps still item-correct); held-out items 36.7% (chance 12.5%, majority 29.9%)
- [x] **(OBS)** Held-out generalization = nearest-seen-item substitution via shared tokens: works for material words (olive oil bottles→glass bottles), fails on spurious fragments (watermelon→hot *water* tanks, scarves→*sc*anners)
- [x] Dense snapshots (run 2, every 300 steps): facts learned between steps 300→600 (core 39→89%, cloze 64→100%), saturated by 900; held-out peaks 48% @900 then 35–43%; best-val ckpt ≠ best probe ckpt → `results/figures/pretrain_checkpoints.png`

## 8. Supervised fine-tuning for QA (spec §7) → [src/datagen/build_sft.py](src/datagen/build_sft.py), [src/finetune.py](src/finetune.py), [notebooks/02_finetune.ipynb](notebooks/02_finetune.ipynb)
- [x] QA dataset from the same KB → `data/sft/train.jsonl` **11,056 pairs**, `val.jsonl` 460 (items open/yes-no/prep, topic facts, streams, material rules, ~4% out-of-scope declines — user approved)
- [x] QA format `<|user|> question<|assistant|> answer<|endoftext|>`; loss only on answer tokens (+eot) — mask tested (`tests/test_finetune.py`)
- [x] Separate eval set `data/eval/qa_eval.jsonl` **654 questions**: core items with unseen wordings (441 incl. yes/no), topic facts / streams / rules with held-out wording (83), 20 held-out items (60), 50 hand-written free-form (typos, casual), 20 out-of-scope
- [x] Leakage checks: 0 eval questions in SFT data (one accidental collision found and fixed), 0 held-out items in SFT, 0 examples > context → `results/tables/sft_data_stats.md`
- [x] Fine-tune script: init from pretrained best, lr 1e-4 warmup+cosine, 1,000 steps (~2.9 epochs), best by SFT val loss; logs pretraining-corpus val loss to measure forgetting; step-0 row = pretrained baseline. CPU smoke test OK
- [x] **USER:** push, run `notebooks/02_finetune.ipynb` on Colab, 📸 screenshots, download `sft/ckpt_best.pt` → `checkpoints/sft/`, results → `results/`
- [x] **(OBS)** SFT: 1.0 min on T4; SFT val 0.969 → 0.268 (best at last step, no overfitting); **forgetting: pretraining-corpus val loss 0.876 → 1.339 (+53%)**
- [x] Run the SAME eval questions on the pretrained and the fine-tuned model → [src/evaluate_qa.py](src/evaluate_qa.py) (`--rescore` re-scores saved answers)
- [x] Scoring: item questions → destination classifier (+ yes/no lead); topic facts → key words; out-of-scope → decline phrase; declines never count as a destination
- [x] **(OBS)** Before vs. after: overall 76.3% → **82.3%**; yes/no 66.7 → 85.0; bins/rules 46.4 → 67.9; out-of-scope declines 0 → 75; known items 100 → 99.7; topic facts 80 → 76; **held-out items 50.0 → 38.3**; free-form ≈47 both → `results/tables/qa_eval.md`, `results/figures/qa_before_after.png`, `results/samples/qa_examples.md`
- [x] **(OBS)** Improved: answers the asked item (names other item 13.2% → 6.0%), yes/no logic, stopping, chat format, declines. Not improved: knowledge itself (destination 90.2 vs 88.7%). **New failure: over-generalized declines** — 0% on familiar wording but 11.7% of held-out-item and 20% of free-form questions declined ("painting the bedroom" → construction decline). Off-topic lexical triggers ("sky blue" → blue cart, "cook rice" → organics). Typos break item recognition. Notes: [report/notes/sft_evaluation.md](report/notes/sft_evaluation.md)

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
- `report/notes/architecture.md` — architecture description, hyperparameter table, parameter breakdown — Model architecture
- `results/figures/pretrain_loss.png` — train/val loss curves, best val marked — Training procedure / Results
- `results/logs/pretrain_log.csv`, `pretrain_train_log.txt`, `pretrain_summary.json` — full pretraining logs — Results (training logs deliverable)
- `results/samples/pretrain_samples_during_training.jsonl` — fixed-prompt generations at every 150 steps — Evaluation / checkpoint evolution
- `results/screenshots/colab_gpu_nvidia_smi.png` — T4 GPU used — Hardware / README
- `results/screenshots/model_config_param_count.png` — config + parameter breakdown — Architecture
- `results/screenshots/pretrain_training_log_end.png` — end of training log + summary — Training procedure
- `results/tables/pretrain_eval.md` / `.csv` — loss/ppl + probe accuracies per checkpoint — Evaluation
- `results/tables/pretrain_item_probe.csv`, `pretrain_fact_cloze.csv` — every probe prompt, completion, verdict — Evaluation / appendix
- `results/samples/pretrain_generation_examples.md` — curated prompts × greedy/sampled — Evaluation (successful/unsuccessful outputs)
- `results/figures/pretrain_checkpoints.png` — probe accuracy + held-out loss across checkpoints — Evaluation (thin: 3 points)
- `report/notes/pretrain_evaluation.md` — draft text for the evaluation section — Evaluation
- `results/tables/sft_data_stats.md` — SFT/eval set sizes per category + leakage checks — Fine-tuning
- `results/samples/sft_examples.md` — example SFT pairs and eval questions — Fine-tuning / appendix
- `results/figures/sft_loss.png` — SFT loss + pretraining-corpus loss (forgetting) — Fine-tuning
- `results/logs/sft_log.csv`, `sft_train_log.txt`, `sft_summary.json`; `results/samples/sft_samples_during_training.jsonl` — SFT logs — Results
- `results/screenshots/sft_data_summary.png`, `sft_training_log.png`, `sft_loss_curves.png` — Colab SFT run — Fine-tuning
- `results/tables/qa_eval.md` / `.csv` — **main before/after table** + behaviour table — Fine-tuning evaluation / Research question
- `results/tables/qa_eval_answers.csv` — every answer from every system with verdicts — appendix
- `results/figures/qa_before_after.png` — grouped bar chart before vs after — Fine-tuning evaluation
- `results/samples/qa_examples.md` — side-by-side answers (fixed by SFT / still wrong) — Fine-tuning evaluation
- `report/notes/sft_evaluation.md` — draft text: setup, results, improved / not improved / limitations, ablation, significance, research-question summary — Fine-tuning + Research
- `results/tables/qa_significance.md` — exact McNemar tests between systems — Fine-tuning evaluation / Research
- `results/figures/sft_no_decline_loss.png`, `results/logs/sft_no_decline_*`, `results/screenshots/sft_no_decline_training_log.png` — ablation run — Research (ablation)

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
- 2026-11-07 — Model: 11,532,288 params. Test caught a subtle point worth explaining in Q&A: with tied embeddings, loss for 'predict the input token itself' starts below ln V because the residual stream already contains that token's embedding.
- 2026-11-08 — Pretraining done on Colab T4 (7.3 min). Best val 0.859 @1800, mild overfitting after. Pretrained model answers Q/A-formatted prompts fluently but about the wrong (related) item → good before/after SFT contrast.
- 2026-11-08 — Snapshot bug: only steps 1500 and 3000 saved (step % 500 with evals every 150). Fixed for future runs.
- 2026-11-08 — Probe scoring bug caught by inspecting failures: classifier read only the first sentence, missing answers after "That's right." → now first destination within 2 sentences; added tests/test_probes.py.
- 2026-11-08 — Pretrained eval: facts memorized (cloze 100%), item recall frame-dependent (72–100%), held-out 36.7% via token-overlap substitution. Lowest-val-loss checkpoint ≠ best fact recall.
- 2026-11-08 — Pretraining run 2 (dense snapshots) reproduced run 1 exactly (identical losses to 4 decimals) → fixed seeds make training deterministic on T4. Good reproducibility evidence.
- 2026-11-08 — SFT data built (11,056 pairs). Pretrained model given the chat format (never seen special tokens) produces garbled continuations ("s go to an electronics drop-off… - What do I do with moving boxes?") — baseline for the SFT comparison.
- 2026-11-08 — QA eval: pretrained model already answers known items 100% in Q:/A: format (corpus had FAQ-style docs) → SFT changed the interface, not the knowledge. SFT's decline behaviour over-generalizes to unfamiliar phrasing (held-out 50→38%). Candidate ablation: SFT without decline examples.
- 2026-11-09 — Ablation (SFT without declines): false declines → 0, casual known-item questions 48.6→62.9% (p=0.06), but held-out only 38.3→41.7% and off-topic questions get confident nonsense. Of 23 questions the decline model declined, the no-decline model answers only 9 correctly → declines mostly land on questions the model can't answer (uncertainty signal). Held-out drop vs pretrained is NOT significant (p=0.14). Significant: overall SFT gain, yes/no, declines (all p<0.001).
- 2026-11-09 — Note for README: checkpoints saved on GPU must be loaded with map_location (load_checkpoint does this) — plain torch.load fails on CPU-only machines.
