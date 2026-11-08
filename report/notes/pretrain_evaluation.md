# Pretrained Model: Generation and Evaluation (report notes, spec section 6)

**Official pretrained checkpoint:** `ckpt_best.pt`, step 1,800, which has the lowest validation loss.

**Reproduce:**
- `python -m src.evaluate_pretrain` runs all evaluations, about 5 minutes on a laptop CPU.
- `python -m src.generate` produces free-form generation.

## 1. Quantitative results

Source: `results/tables/pretrain_eval.md`.

| Checkpoint | Val loss | Val ppl | Test loss | Test ppl | Core-item probe | Held-out-item probe | Fact cloze |
|---|---:|---:|---:|---:|---:|---:|---:|
| step 1500 | 0.872 | 2.39 | 0.813 | 2.26 | 94.3% | 41.7% | 100% |
| **step 1800 (best val)** | **0.863** | **2.37** | **0.801** | **2.23** | **90.7%** | **36.7%** | **100%** |
| step 3000 (final) | 0.893 | 2.44 | 0.825 | 2.28 | 95.0% | 40.0% | 100% |

**How the metrics were measured:**
- **Loss and perplexity** are computed over the whole validation or test split, in non-overlapping 256-token windows.
- **Item probe.** For each item there are three prompt "frames" that stop just before the destination:
  - statement: "Pizza boxes belong…"
  - directory entry: "Pizza boxes⏎Where:"
  - FAQ: "Q: Where do pizza boxes go?⏎A:"

  The greedy completion is classified by the first disposal stream it names. Tests are in `tests/test_probes.py`.
- **Baselines for the item probe:** chance with 8 streams is 12.5%; always answering "garbage", the most common stream, scores 29.9%.
- **Fact cloze:** 25 familiar knowledge-base statements cut just before the key word, e.g. "The green organics cart is collected every ___".

**Per-frame accuracy on core items** at the best checkpoint:

| Frame | Accuracy |
|---|---:|
| Statement | 72.1% |
| Directory entry | 100% |
| FAQ | 100% |

### Findings

1. **Familiar statements are reproduced perfectly.** All 25 cloze statements were completed correctly at every checkpoint, so the model has memorized the knowledge base's core facts.
2. **Item→stream relationships are learned, but recall depends on the prompt.** Every core-item failure came from the "X belong…" frame. The model completes it with *"…belong in the garbage, not the recycling"*, a high-frequency fixed phrase in the corpus, even for cans, bottles and cartons. In the same wrong completions, it still produces the correct **item-specific** preparation steps ("Rinse them with a little water. Leave the lid on."). So the item is recognized, but a strong phrase-level habit overrides the destination. The same items score 100% in the directory and FAQ frames.
3. **Unseen items are well below core items (36.7% vs 90.7%), but above both baselines.** The model substitutes a known item that shares tokens with the unseen name:
   - glass olive oil bottles → *glass bottles* ✅
   - almond milk cartons → *milk cartons* ✅
   - electric kettles → *electric shavers* ✅
   - watermelon rinds → *hot **water** tanks* ❌
   - scarves → *scanners* ❌, sharing only the subword "sc"
   - bookshelves → *delivery boxes* ❌

   Generalization works when the shared token is a **material word**, and fails when it is an unrelated fragment. This links to the tokenizer: unseen names are split into unfamiliar pieces, as shown in `tokenization_examples.md`.
4. **Checkpoints differ little.** By step 1,500, the knowledge probes are already saturated. The lowest-validation-loss checkpoint (1,800) is not the best on the item probe, because validation loss measures fluency on all text, not recall of specific facts.
5. **Test perplexity (2.23) is slightly lower than validation (2.37).** Both splits use held-out phrasing, so this is sampling variation between two small splits, not a sign of leakage.

## 2. Qualitative results

All examples are in `results/samples/pretrain_generation_examples.md`. Generation used greedy decoding and temperature 0.8.

### Successful outputs
- **Plausible domain text in the right document style.** "Pizza boxes" → a correct A–Z directory entry: *"Where: green cart ⏎ How: Scrape them clean of food. ⏎ Note: Greasy pizza boxes go in organics. A clean, grease-free lid can be torn off and recycled."*
- **Familiar statements completed correctly.** "Glass bottles and jars go in the" → *"blue cart. Empty them fully first. Rinse them with a little water. Put the cap or lid back on."*
- **Reproduces relationships in a new framing.** "Dear resident, thank you for your question about cooking grease." → *"It goes in the green cart… never pour grease down the drain… Large amounts of used cooking oil go to a drop-off depot."* All of this matches the knowledge base.
- **Sampling adds variety without breaking facts.** "Here is a quick tip about batteries:" (T = 0.8) → *"lithium-ion batteries are household hazardous waste… Cover the terminals with clear tape… a leading cause of fires in collection trucks."*
- **Soft-plastic generalization.** The unseen item "Bubble wrap" → *"is garbage and goes in the black cart."* Correct, through the material word "wrap".

### Unsuccessful outputs
- **Yes/no logic not learned.** "Question: Can I put plastic bags in the blue cart? Answer:" → *"**Yes!** Plastic bags go in the black garbage cart."* The destination is right, but the yes/no lead contradicts it. The model learned how often answers start with "Yes", not the logic behind it.
- **Answers about the wrong item.** "Q: Where do wireless routers go?" → *"In short, **drills** count as electronic waste."* The stream is right, but for a different item. In the training-time samples, a battery question was answered about thermometers.
- **Spurious subword associations.** "Nail polish remover" → *"food: it is collected in the blue recycling cart. Make sure it is dry; wet paper is hard to recycle."* This is wrong: it should go to hazardous waste.
- **Repetition at greedy decoding.** "Plastic hummus tubs" → *"plastic hangers, plastic hangers, plastic hangers, plastic toys, plastic toys"*. Greedy output also repeats list lines such as "Household batteries: Cover the terminals…" twice.
- **Narrative prompts break it.** "My neighbour asked what to do with her old couch, so I told her" → *"bicides? ⏎ Property Management"*. First-person stories are rare in the corpus.
- **No world knowledge; everything collapses into the domain.** "The capital of France is" → *"collected with the garbage in the black cart."* "Once upon a time," → a sentence about drop-off depots.

### What this shows before fine-tuning
The pretrained model has learned the fictional domain's vocabulary, document styles and most facts. However:
- it does not reliably **answer the question that was asked**
- it often drifts to a related item or continues into the next FAQ entry
- it has no notion of when to stop

These are the behaviours supervised fine-tuning targets.
