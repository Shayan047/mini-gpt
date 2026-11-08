# Data Generation Methodology

This document describes how the synthetic pretraining corpus for the Tidewell Residential Waste Guide was created. It covers deliverable (b) of the term project: the generated corpus, the fictional knowledge base, and the method used to generate it.

## 1. Fictional domain

- **Organization:** Tidewell Waste & Recycling Services, the fictional waste service of the fictional town of Tidewell.
- **Department modelled:** Residential Waste Guidance only. This covers how residents sort, prepare and dispose of everyday household waste. Commercial, industrial and construction waste are out of scope.
- **Design choice: generic facts.** The knowledge base contains no street addresses, phone numbers, staff names or clock times. For example, the guide says "use your building's designated bulky item area; ask your property manager" instead of naming a location. This stops the model from learning one memorized detail and repeating it as the answer to unrelated questions.

## 2. Knowledge base

- **Files:**
  - [kb/knowledge_base.json](kb/knowledge_base.json) is the machine-readable source of truth.
  - [kb/knowledge_base.md](kb/knowledge_base.md) is the readable version.
- **Validator:** `python src/kb.py`.

| Component | Count |
|---|---:|
| Disposal streams (blue recycling cart, green organics cart, black garbage cart, hazardous waste drop-off, electronics drop-off, bulky item pickup, textile drop-off, pharmacy take-back) | 8 |
| Material rules, e.g. "rigid plastic containers go in the blue cart" | 20 |
| Preparation codes (empty, rinse, flatten, tape terminals, ...) | 27 |
| Household items: 147 core + 20 held-out | 167 |
| Item names and aliases | 541 |
| Topics (collection schedule, cart placement, missed collection, apartments, drop-off depots, bulky items, rules and fines, reduce/reuse, home composting, organics tips, moving, seasonal, why sorting matters) | 13 |
| Topic facts | 55 |

Each item records:
- its stream
- the material rule that explains it, or `null` for an exception such as shredded paper going to organics
- its preparation codes
- an optional note
- its split: `core` or `heldout`

**Held-out items** (20, such as hummus tubs, bubble wrap, wireless routers, nail polish remover and bookshelves) never appear in the corpus or the fine-tuning data. They are used only to test whether the model can apply the material rules to items it has never seen. Each one was chosen so that a material rule explains it.

## 3. Generation method

The corpus was generated with **Claude Code** (an AI coding agent, model Claude Opus 5.5). Two parts were combined.

### 3.1 Template generator (bulk of the corpus)

Claude Code wrote paraphrase banks for every fact in the knowledge base, along with a Python generator that assembles documents from them. All files are in `src/datagen/`.

| Paraphrase bank | Wordings |
|---|---:|
| Ways to say where a stream goes ("go in the blue cart", "belong in the blue recycling bin", ...) | 82 |
| Preparation step wordings | 101 |
| Material rule wordings, in addition to the knowledge base text | 80 |
| Topic fact wordings, in addition to the knowledge base text | 165 |
| Sentence frames for stating where an item goes (statement and direct-answer styles) | 25 |
| Question wordings: open item questions, yes/no, preparation, topic | 38 + 11 + 6 + 174 |

Templates use a small mini-language: `[a|b]` picks one alternative, `{slot}` fills in a value, and `{v:go}` conjugates the verb to agree with the item ("cans go" / "foil goes"). This multiplies the number of distinct surface forms.

The generator produces **17 document types**:
- guide pages
- A–Z directory entries
- FAQs
- newsletters
- internal staff memos
- staff training notes
- customer-service emails
- chat transcripts
- social media posts
- topic pages
- public notices
- community forum threads
- school handouts
- apartment building notices
- contamination reports
- "common mistakes" sheets
- "rules of thumb" sheets

**Balanced exposure.** Items, topic facts and rules are drawn from shuffled cycles, so each one appears about equally often and no single answer dominates.

**Unseen phrasing for evaluation.** In every paraphrase bank with 5 or more wordings, about 15% of the wordings are reserved for validation and test documents only, chosen by a fixed hash. Validation and test documents therefore contain sentence structures that never occur in training.

### 3.2 Hand-written seed documents (149)

Claude Code also wrote 149 complete free-form documents in [seed_docs/](seed_docs/). They use styles the templates cannot produce:
- blog posts and advice columns
- radio announcements
- podcast excerpts
- phone-call transcripts and conversations between neighbours
- policies, a handbook and training manuals
- lesson plans and event recaps
- item spotlights
- letters, diary entries and scenario walkthroughs

They are split 90/5/5 across train, validation and test, the same as the generated documents.

## 4. Consistency and leakage checks

Run with `python -m src.datagen.check_corpus`. The report is saved to `results/tables/corpus_checks.md`.

1. **Held-out leakage.** No name or alias of a held-out item, in singular or plural form, appears anywhere in the corpus. Result: **0**.
2. **Generic facts.** No phone numbers, clock times or street addresses appear. Result: **0**.
3. **Knowledge-base consistency.** A sentence that names exactly one item and exactly one destination, without negation, must name that item's correct destination. 13,411 sentences were checked.
   - **Seed documents: 0 conflicts.** The check caught two ambiguous sentences while the seed documents were being written, and both were rewritten.
   - **Generated documents:** 213 flags, all false positives. These are sentences about part of an item, such as "remove the plastic liner and put it in the garbage" for cereal boxes.
4. **Duplicates.** No document text appears twice, within or across splits. Result: **0**; the generator regenerates on any collision.
5. **Manual review.** Seed documents were also reviewed by hand. Three sentences that introduced facts not in the knowledge base, or hinted at held-out items, were removed or reworded.

## 5. Resulting corpus

Regenerate with `python -m src.datagen.generate_corpus`; the fixed seeds make it fully reproducible. Per-type statistics are in `results/tables/corpus_stats.md`.

| Split | Documents | Words | Characters |
|---|---:|---:|---:|
| train | 6,135 | 802,227 | 4,659,607 |
| val | 327 | 41,784 | 242,747 |
| test | 327 | 42,839 | 250,532 |

- **Token count:** 1,310,522 tokens after tokenization with the byte-level BPE tokenizer (vocabulary 2,048): train 1,184,188, val 62,115, test 64,219. See `results/tables/token_stats.md`.
- **Exposure in training:** each core item is mentioned 203–689 times, with a median of 245. Each topic fact is used in 82–200 documents and each material rule in 124–193 documents. Per-item and per-fact counts are in `results/tables/item_coverage.csv` and `results/tables/fact_coverage.csv`.
- **Phrasing overlap:** 72–75% of validation and test sentences also appear verbatim in training. This is expected for a corpus built from a fixed knowledge base, since short fact sentences recur. It means validation loss mostly measures in-domain fluency. Generalization is measured separately, with unseen question wordings and held-out items in the QA evaluation set.

## 6. Output format

`data/corpus/{train,val,test}.jsonl` contains one document per line, with these fields:
- `id`
- `split`
- `type`
- `source`: `generator` or `seed:<file>:<index>`
- `text`
- `items`, `facts`, `rules`: the knowledge-base IDs the document was built from
