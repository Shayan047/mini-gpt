# Pretrained model evaluation

Item probe: greedy completion of 3 prompt frames per item, classified by the destination named first. Chance with 8 streams = 12.5%; always answering the most common stream (garbage) = 29.9% on core items. Fact cloze: 25 knowledge-base statements cut before the key word.

| checkpoint | val loss | val ppl | test loss | test ppl | core items | statement | directory | FAQ | held-out items | fact cloze |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| snapshot step 1500 | 0.872 | 2.392 | 0.813 | 2.255 | 94.3% | 83.0% | 100.0% | 100.0% | 41.7% | 100.0% |
| best step 1800 | 0.8628 | 2.37 | 0.8009 | 2.228 | 90.7% | 72.1% | 100.0% | 100.0% | 36.7% | 100.0% |
| snapshot step 3000 | 0.8929 | 2.442 | 0.8245 | 2.281 | 95.0% | 85.0% | 100.0% | 100.0% | 40.0% | 100.0% |
