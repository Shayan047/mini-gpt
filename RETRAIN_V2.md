# Retrain recipe (KB / corpus / SFT v1.1)

Senior-ML changes for **better held-out transfer** and **honestly longer GPU time**:

1. More core items on thin material rules (no held-out name leaks)
2. ~2× pretrain corpus + new `material_sort_drill` docs
3. More rule/material SFT pairs
4. Pretrain default **8,000 steps** (still save **best validation**)
5. SFT with **15% pretrain replay** (cuts forgetting) for **1,500 steps**

Architecture stays **6L / 6H / d_model=384** (~11.5M) so the research comparison stays clean.

## Important

- Expanded corpus / SFT / KB are already in the repo (`data/corpus`, `data/sft`, `data/kb`, `data/eval`).
- Matching v1.1 BPE files are in **`data/tokenizer_v2/`** and **`data/tokens_v2/`**.
- Active `data/tokenizer/` + `data/tokens/` stay on **v1.0** so `models/pretrained.pt` / `models/finetuned.pt` demos keep working until you finish retrain.
- Do **not** invent wall-clock times; log whatever Colab reports.

## 1. Activate v1.1 tokens (before GPU train)

```bash
# Option A (recommended): copy the already-built v1.1 artifacts
cp -r data/tokenizer_v2/* data/tokenizer/
cp -r data/tokens_v2/* data/tokens/

# Option B: rebuild from the current corpus
python -m src.train_tokenizer
```

On Windows PowerShell:

```powershell
Copy-Item data/tokenizer_v2/* data/tokenizer/ -Force
Copy-Item data/tokens_v2/* data/tokens/ -Force
```

If you still need to regenerate everything from scratch:

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
python -m src.train --out-dir checkpoints/pretrain_v2 \
  --max-steps 8000 --warmup-steps 300 --eval-every 200 --dropout 0.12
```

Use `checkpoints/pretrain_v2/ckpt_best.pt` (lowest val loss), not necessarily step 8000.

## 3. Fine-tune (GPU)

```bash
python -m src.finetune \
  --init checkpoints/pretrain_v2/ckpt_best.pt \
  --out-dir checkpoints/sft_v2 \
  --max-steps 1500 --replay-ratio 0.15 --lr 8e-5
```

## 4. Evaluate (CPU fine)

```bash
python -m src.evaluate_pretrain --ckpt-dir checkpoints/pretrain_v2
python -m src.evaluate_qa \
  --pretrained checkpoints/pretrain_v2/ckpt_best.pt \
  --finetuned checkpoints/sft_v2/ckpt_best.pt
python -m src.significance
```

After you are happy with the numbers, copy the best checkpoints into `models/` and refresh `models/model_info.json` (leave `data/tokenizer/` as the matching v1.1 tokenizer).

## What we expect (hypotheses, not claims)

| Signal | Hope |
|---|---|
| Pretrain GPU time | Clearly past 15 minutes on a free T4 |
| Val loss curve | Bottom later than step 1,800 because the corpus is larger |
| Held-out item QA | Up vs 38–50% if material→stream transfer improved |
| Corpus val during SFT | Smaller rise than +53% thanks to replay |

Report only numbers from the new logs/tables.
