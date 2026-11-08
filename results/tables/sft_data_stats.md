# Fine-tuning and evaluation data

Format: `<|user|> question<|assistant|> answer<|endoftext|>`; loss on answer tokens only.

**SFT train:** 11,056 pairs · **SFT val:** 460 pairs · tokens per example: mean 47, max 118 (context 256) · total SFT train tokens ≈ 524,157

| SFT category | train pairs |
|---|---:|
| item_open | 5,726 |
| item_prep | 1,079 |
| item_yesno | 2,239 |
| out_of_scope | 417 |
| rule | 374 |
| stream | 221 |
| topic_fact | 1,000 |

**Evaluation set:** 654 questions (never used in fine-tuning; same questions for both models)

| evaluation category | questions |
|---|---:|
| core_item_unseen_wording | 294 |
| core_item_unseen_wording_yesno | 147 |
| freeform_core | 35 |
| freeform_heldout | 15 |
| heldout_item | 40 |
| heldout_item_yesno | 20 |
| out_of_scope | 20 |
| rule_unseen_wording | 20 |
| stream_unseen_wording | 8 |
| topic_fact_unseen_wording | 55 |

## Leakage checks

- Evaluation questions that also appear (normalized) in the SFT data: **0**
- SFT pairs mentioning a held-out item: **0**
- SFT examples longer than the 256-token context: **0**
