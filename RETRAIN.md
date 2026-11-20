# Retrain recipe

Optional guide to regenerate data and retrain the models in this repository.

Changes relative to an earlier smaller setup:

1. More core items on thin material rules (held-out names stay out of train/SFT)
2. Larger pretrain corpus, including `material_sort_drill` documents
3. More rule/material pairs in SFT
4. Pretrain for **8,000** steps (keep the **best validation** checkpoint)
5. Fine-tune for **1,500** steps with **15%** pretrain replay

Architecture stays **6 layers / 6 heads / d_model=384** (~11.5M parameters).

## Already in the repo

- Corpus, SFT data, KB, eval: `data/corpus`, `data/sft`, `data/kb`, `data/eval`
- Tokenizer and token files: `data/tokenizer/`, `data/tokens/`
- Released checkpoints: `models/pretrained.pt` (best val, step 4,000), `models/finetuned.pt` (step 1,500)
- Local training outputs: `checkpoints/pretrain/`, `checkpoints/sft/`

Use the wall-clock times recorded in the training logs; do not substitute estimated times.

## 1. Rebuild data (optional)

```bash
python -m src.datagen.add_core_items
python src/kb.py
python src/kb_to_md.py
python -m src.datagen.generate_corpus          # train=12000, val=500, test=500
python -m src.datagen.check_corpus
python -m src.train_tokenizer                  # vocab still 2048
python -m src.datagen.build_sft
```

## 2. Pretrain (GPU / Colab T4)

```bash
python -m src.train --out-dir checkpoints/pretrain \
  --max-steps 8000 --warmup-steps 300 --eval-every 200 --dropout 0.12
```

Use `checkpoints/pretrain/ckpt_best.pt` (lowest validation loss), not necessarily step 8000.

## 3. Fine-tune (GPU)

```bash
python -m src.finetune \
  --init checkpoints/pretrain/ckpt_best.pt \
  --out-dir checkpoints/sft \
  --max-steps 1500 --replay-ratio 0.15 --lr 8e-5
```

## 4. Evaluate (CPU is fine)

```bash
python -m src.evaluate_pretrain --ckpt-dir checkpoints/pretrain
python -m src.evaluate_qa \
  --pretrained checkpoints/pretrain/ckpt_best.pt \
  --finetuned checkpoints/sft/ckpt_best.pt
python -m src.significance
python -m src.plot_losses --log checkpoints/pretrain/log.csv \
  --out results/figures/pretrain_loss.png --title "Pretraining: next-token prediction"
python -m src.plot_sft --log checkpoints/sft/log.csv --out results/figures/sft_loss.png
```

When the run looks good, copy the best checkpoints into `models/` and update `models/model_info.json`.

## What to check after a retrain

| Signal | Typical check |
|---|---|
| Pretrain GPU time | Compare against the logged Colab time for this schedule |
| Validation curve | Prefer the best-validation checkpoint over the final step |
| Held-out item QA | Compare to the numbers in `results/tables/qa_eval.md` |
| Corpus val during SFT | Should rise only slightly when replay is enabled |

Cite numbers only from the logs and tables produced by that run.
