# Fine-tuning and evaluation data

Format: `<|user|> question<|assistant|> answer<|endoftext|>`; loss on answer tokens only.

**SFT train:** 10,954 pairs · **SFT val:** 456 pairs · tokens per example: mean 47, max 130 (context 256) · total SFT train tokens ≈ 518,339

| SFT category | train pairs |
|---|---:|
| item_open | 4,998 |
| item_prep | 982 |
| item_yesno | 1,992 |
| material | 450 |
| out_of_scope | 428 |
| rule | 879 |
| stream | 227 |
| topic_fact | 998 |

**Evaluation set:** 732 questions (never used in fine-tuning; same questions for both models)

| evaluation category | questions |
|---|---:|
| core_item_unseen_wording | 346 |
| core_item_unseen_wording_yesno | 173 |
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
