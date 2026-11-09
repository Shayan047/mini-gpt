# Supervised Fine-Tuning and QA Evaluation (report notes, spec section 7 and research question)

## Setup

- **Fine-tuning data:** 11,056 question–answer pairs plus 460 for validation, all built from the same knowledge base (`results/tables/sft_data_stats.md`):
  - item questions (open, yes/no and preparation)
  - topic facts
  - bin ("stream") and material-rule questions
  - about 4% out-of-scope questions answered with a polite decline
- **Format:** `<|user|> question<|assistant|> answer<|endoftext|>`. The loss is computed on the answer tokens only.
- **Training:** starts from the pretrained best checkpoint (step 1,800); AdamW, lr 1e-4 with warmup and cosine decay, 1,000 steps (about 2.9 epochs), 1.0 minute on a T4.
- **Loss (answer tokens):** 0.969 for the pretrained model → **0.268** after fine-tuning. The lowest validation loss came at the last step and was still decreasing slowly, so there is no overfitting on the fine-tuning distribution.
- **Forgetting:** validation loss on the *pretraining* corpus rose from **0.876 to 1.339 (+53%)**. The model gave up general document modelling in exchange for answering questions (`results/figures/sft_loss.png`).

### Evaluation set: 654 questions, never used in fine-tuning, identical for both models

| Group | Questions |
|---|---:|
| Known items, unseen question wordings (open) | 294 |
| Known items, unseen question wordings (yes/no) | 147 |
| Held-out items (never in the corpus or fine-tuning data) | 60 |
| Hand-written free-form questions (casual, typos, descriptions), known items | 35 |
| Hand-written free-form questions, held-out items | 15 |
| Topic facts with a held-out question wording | 55 |
| Bin and material-rule questions with a held-out wording | 28 |
| Out-of-scope questions | 20 |

**How each system was prompted** (greedy decoding):
- **Pretrained, Q/A prompt:** `Q: …⏎A:`, the FAQ format it saw in its corpus. Its answer is cut at the first blank line, where it would start the next FAQ entry.
- **Pretrained, chat prompt:** shown as a sanity row; it uses tokens the model has never seen.
- **Fine-tuned:** the chat format it was trained on.

## Results

Source: `results/tables/qa_eval.md` and `results/figures/qa_before_after.png`.

| Question group | n | Pretrained (Q/A prompt) | Pretrained (chat prompt) | **Fine-tuned** |
|---|---:|---:|---:|---:|
| Known items, unseen wording | 294 | 100.0% | 27.9% | 99.7% |
| Known items, yes/no | 147 | 66.7% | 0.0% | **85.0%** |
| Held-out items | 60 | **50.0%** | 23.3% | 38.3% |
| Free-form, known items | 35 | 45.7% | 17.1% | 48.6% |
| Free-form, held-out items | 15 | 26.7% | 13.3% | 26.7% |
| Topic facts, unseen wording | 55 | 80.0% | 21.8% | 76.4% |
| Bins & material rules | 28 | 46.4% | 21.4% | **67.9%** |
| Out-of-scope (should decline) | 20 | 0.0% | 0.0% | **75.0%** |
| **All** | 654 | 76.3% | 18.7% | **82.3%** |

| Behaviour | Pretrained (Q/A) | Fine-tuned |
|---|---:|---:|
| Destination correct (all item questions) | 90.2% | 88.7% |
| Yes/no lead correct | 64.4% | **82.2%** |
| Answer names a *different* item first | 13.2% | **6.0%** |
| False declines: known items, unseen wording | 0% | 0% |
| False declines: held-out items | 0% | **11.7%** |
| False declines: free-form questions | 0% | **20.0%** |
| Mean answer length (words) | 22.9 (after cutting) | 19.3 (stops on its own) |

## What improved

1. **Answering the question that was asked.** Answers that name a different item fell from 13.2% to 6.0%.
   - Before: *"Where should used needles go?"* → *"In short, used **motor oil** should be brought to a hazardous waste drop-off."*
   - After: *"…used medicine and needles never go in a cart; take it to a pharmacy."*
2. **Yes/no logic** (lead word correct: 64% → 82%).
   - Before: *"Can I put my old electric kettle in the garbage?"* → *"**Yes!** … televisions and monitors go to an electronics drop-off."*
   - After: *"**No, sorry.** They count as electronic waste."*
   - The pretrained model had learned only how often answers start with "Yes", not when a yes is correct.
3. **Bin and material-rule questions** (46% → 68%). These are general questions such as *"Which things can I put in my blue bin?"*, which pretraining text rarely phrased as questions.
4. **Knowing when to stop.** The fine-tuned model ends its answer with `<|endoftext|>` (mean 19 words). The pretrained model only gives a usable answer if we cut its output by hand. In the chat format, 412 of its 634 in-domain answers ran to the 80-token limit.
5. **Declining off-topic questions** (0% → 75%), e.g. *"What is the capital of Italy?"* → *"Sorry, that's outside what I can help with…"*.
6. **Following the chat format at all.** The same pretrained weights given the chat format score 18.7%: they start mid-word and drift into unrelated FAQ entries. Fine-tuning taught the model what `<|user|>` and `<|assistant|>` mean.

## What did not improve

1. **Known items were already solved.** The pretraining corpus contained FAQs, chats and emails in Q&A form, so the pretrained model already answered 100% of open known-item questions in the `Q:/A:` format. Fine-tuning changed the *interface*, not the knowledge. **Fine-tuning adds no new facts:** destination accuracy is 90.2% before and 88.7% after.
2. **Held-out items got worse (50.0% → 38.3%), mainly because of over-generalized declines.** 11.7% of held-out-item questions were declined, e.g. *"Where does an old baby crib go?"*, *"how do I get rid of weed killer I don't need"* and *"where do corn cobs go after a bbq"*. Questions that look unlike any fine-tuning question are treated as out of scope.
3. **Free-form questions stayed at about 47%,** with 20% declined:
   - *"my kid broke a drinking glass, which bin?"*
   - *"my old fridge died"*
   - *"I have a broken couch, what now?"*
   - *"I have half a can of paint left over from **painting the bedroom**"* got the *commercial/construction* decline. The words about painting a room were apparently associated with renovation.
4. **Typos.** *"where do coffe grounds go"* → the fine-tuned model answered about *cords and cables*, and *"is cardbord recyclable"* failed for both models. The misspelled words split into unfamiliar sub-word pieces, and the model latches onto the wrong item.
5. **Topic facts stayed flat** (80% → 76%). Errors are usually a *neighbouring* fact from the same topic: *"Where can I see my collection schedule?"* → the cart set-out time. The automatic key-word scoring is strict, and a few answers it marks wrong are reasonable, e.g. *"Can I leave stuff beside the shared bins?"* → *"Never leave items … beside a full bin."*

## Limitations of the fine-tuned model

- **Over-cautious declines:** about 1 in 5 unfamiliar or casually phrased in-domain questions is declined. Decline training and coverage of unfamiliar phrasing pull against each other.
- **Single words still trigger answers to off-topic questions:**
  - *"Why is the sky **blue**?"* → a blue-cart answer
  - *"How do I cook **rice**?"* → *"It is accepted in the green organics cart"*
  - *"What do factories do with their scrap metal?"* → pharmacy take-back
- **Forgetting:** pretraining-corpus loss rose 53%, so the fine-tuned model is a worse general text generator than the pretrained one.
- **No reasoning about unseen items:** held-out items are still handled by token-overlap substitution, as in pretraining.
  - *"Where do wool scarves go?"* → a bulky item, by analogy with *wardrobes*.
  - *"I'm getting rid of an old bookshelf"* → *old **books***, recycling.
- **Evaluation caveats:**
  - Automatic scoring checks the bin, the yes/no lead and key words, not the full correctness of the answer.
  - The free-form and held-out groups are small (15–60 questions), so differences of a few questions are not significant.
  - The pretrained model is given a hand-written answer cut-off that the fine-tuned model does not need.

## Ablation: fine-tuning without the decline examples

- **Hypothesis:** the roughly 4% out-of-scope "decline" examples taught the model to decline anything unfamiliar, and that caused the held-out and free-form problems.
- **Experiment:** a second fine-tuning run with everything identical (same pretrained start, seed, 1,000 steps, learning rate), except that the 417 out-of-scope examples were removed, leaving 10,639 pairs. Run on a T4 in 1.2 minutes. Files: [notebooks/03_sft_ablation.ipynb](../../notebooks/03_sft_ablation.ipynb), `results/logs/sft_no_decline_*`.

| Question group | Pretrained (Q/A) | Fine-tuned | **Fine-tuned, no declines** |
|---|---:|---:|---:|
| Known items, unseen wording | 100.0% | 99.7% | 100.0% |
| Known items, yes/no | 66.7% | 85.0% | 83.7% |
| Held-out items | 50.0% | 38.3% | 41.7% |
| Free-form, known items | 45.7% | 48.6% | **62.9%** |
| Free-form, held-out items | 26.7% | 26.7% | 20.0% |
| Topic facts | 80.0% | 76.4% | 74.5% |
| Bins & material rules | 46.4% | 67.9% | 71.4% |
| Out-of-scope (declines) | 0% | 75.0% | **0%** |
| **All** | 76.3% | **82.3%** | 80.7% |
| False declines on in-domain questions | 0% | 3.6% | 0% |
| Forgetting (pretraining-corpus loss after fine-tuning) | 0.876 | 1.339 | 1.322 |

### What the ablation shows

1. **The decline examples cause the false declines, but not the held-out drop.**
   - Without them, false declines fall to 0%, yet held-out accuracy recovers only from 38.3% to 41.7%. It stays below the pretrained model's 50%.
   - Of the 23 in-domain questions the original fine-tuned model declined, the no-decline model answers only **9 correctly (39%)**. The other 14 get confidently wrong answers, e.g. *"How do I dispose of corn cobs?"* → *"If you live in an apartment, you share bins with your building."*
   - So the model mostly declined questions it could not answer anyway. Its declines partly act as an **uncertainty signal**.
2. **The cost of declining is concentrated in casual wording of known items:** 48.6% → 62.9% when declines are removed. Five questions flipped to correct and none flipped to wrong, a trend at p = 0.06. Examples: *"my kid broke a drinking glass, which bin?"* and *"a fluorescent tube broke, where does it go"*.
3. **Without decline training, off-topic questions get confident nonsense:**
   - *"What is the capital of Italy?"* → *"…leaves are collected in the green organics cart."*
   - *"What's a good exercise routine?"* → *"Exercise equipment goes out through bulky item collection."*
4. **Forgetting is the same with and without declines** (1.34 vs 1.32), so it comes from fine-tuning itself.

## Statistical significance

Source: `results/tables/qa_significance.md`; exact McNemar tests on the same questions.

**Significant** (p < 0.05):
- the overall improvement from fine-tuning (+74 / −35 questions, p < 0.001)
- yes/no logic (+35 / −8, p < 0.001)
- declining off-topic questions (+15 / −0, p < 0.001)

**Not significant** with these group sizes:
- the held-out drop (p = 0.14)
- topic facts (p = 0.79)
- the bins & rules gain (p = 0.07)
- free-form changes (p ≥ 0.06)

The report should present these as tendencies, not effects.

## Research question: summary

*How well does a small Transformer generalize waste-sorting rules to unseen items and unseen question wordings, and how does supervised fine-tuning change that?*

- **Unseen wordings of known facts:** generalization is excellent before and after fine-tuning: 100% and 99.7% on open questions, with 0% false declines.
- **Unseen items:** generalization is weak, at 37–50% across the probe and QA tasks (chance 12.5%, majority 30%). It works only when the item name shares a **material word** with known items (glass, milk cartons, tins, batteries).
- **What fine-tuning changed:** it taught the model *how to answer* — the chat format, answering the asked item, yes/no logic (significant), stopping and declining (significant). It did not make the model better at *what* it knows about unseen items.
  - Held-out accuracy fell from 50% to 38–42%, which is not significant.
  - The ablation shows this drop is **not** mainly caused by decline training.
  - Decline training does trade about 14 points of casual-question coverage for the ability to refuse off-topic questions. Its declines mostly land on questions the model would have answered wrongly anyway.
