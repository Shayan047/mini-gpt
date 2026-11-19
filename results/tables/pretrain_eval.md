# Pretrained model evaluation

Official checkpoint = **best validation** (`ckpt_best.pt`, step 4,000), not the final step.
Full item-probe / cloze sweeps across every snapshot were not re-exported here; validation/train losses below are from the training log (`results/logs/pretrain_log.csv`). QA generalization numbers are in `qa_eval.md`.

| checkpoint | val loss | val ppl | train loss | notes |
|---|---:|---:|---:|---|
| step 2,000 | 0.8650 | 2.375 | 0.4083 | still improving |
| step 3,000 | 0.8662 | 2.378 | 0.3754 | near the valley |
| **step 4,000 (best val)** | **0.8418** | **2.320** | **0.3522** | **official pretrained** |
| step 6,000 | 0.8652 | 2.376 | 0.3074 | val rising |
| step 8,000 (final) | 0.8899 | 2.435 | 0.2794 | more memorization |

Chance with 8 streams = 12.5%. On the held-out QA set the pretrained model scores 40.0% on never-seen items and 99.7% on known items with new wording (`results/tables/qa_eval.md`).
