# Mini GPT: Tidewell Residential Waste Guide

CS 5370 (Fall 2026) group project. A small decoder-only Transformer, trained from scratch on a synthetic corpus about a fictional town's residential waste guide, then fine-tuned for question answering.

> Work in progress. Full installation, inference instructions and checkpoint downloads will be added here before submission.

## Repository layout

| Path | Contents |
|---|---|
| `data/kb/` | Fictional knowledge base (`knowledge_base.json`, readable `knowledge_base.md`) |
| `data/seed_docs/` | 149 hand-written seed documents |
| `data/corpus/` | Generated pretraining corpus (`train/val/test.jsonl`) |
| `data/tokenizer/` | Byte-level BPE tokenizer (`tokenizer.json`, `vocab.txt`) |
| `data/tokens/` | Tokenized splits (`*.bin`, uint16) + `meta.json` |
| `data/DATA_GENERATION.md` | Data-generation methodology |
| `src/` | Knowledge base tools, data generator, tokenizer, model and training code |
| `tests/` | Tests |
| `results/` | Tables, figures, samples and logs used in the report |
| `report/` | Report drafts and notes |

## Rebuilding the data (no dependencies, Python 3.10+)

```bash
python src/kb.py                          # validate the knowledge base
python -m src.datagen.generate_corpus     # generate data/corpus/*.jsonl (deterministic seeds)
python -m src.datagen.check_corpus        # leakage / consistency / duplicate checks
python -m src.train_tokenizer             # train BPE (vocab 2048) and write data/tokens/*.bin
python -m tests.test_tokenizer
```
