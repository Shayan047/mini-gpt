# Pretrained model evaluation

Item probe: greedy completion of 3 prompt frames per item, classified by the destination named first. Chance with 8 streams = 12.5%; always answering the most common stream (garbage) = 29.9% on core items. Fact cloze: 25 knowledge-base statements cut before the key word.

| checkpoint | val loss | val ppl | test loss | test ppl | core items | statement | directory | FAQ | held-out items | fact cloze |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| snapshot step 300 | 1.6806 | 5.369 | 1.6311 | 5.11 | 39.2% | 54.4% | 42.2% | 21.1% | 23.3% | 64.0% |
| snapshot step 600 | 0.979 | 2.662 | 0.9161 | 2.5 | 89.1% | 85.0% | 95.2% | 87.1% | 40.0% | 100.0% |
| snapshot step 900 | 0.9009 | 2.462 | 0.8393 | 2.315 | 92.3% | 77.6% | 100.0% | 99.3% | 48.3% | 96.0% |
| snapshot step 1200 | 0.8681 | 2.382 | 0.8099 | 2.248 | 91.2% | 73.5% | 100.0% | 100.0% | 41.7% | 100.0% |
| snapshot step 1500 | 0.872 | 2.392 | 0.813 | 2.255 | 94.3% | 83.0% | 100.0% | 100.0% | 41.7% | 100.0% |
| best step 1800 | 0.8628 | 2.37 | 0.8009 | 2.228 | 90.7% | 72.1% | 100.0% | 100.0% | 36.7% | 100.0% |
| snapshot step 1800 | 0.8628 | 2.37 | 0.8009 | 2.228 | 90.7% | 72.1% | 100.0% | 100.0% | 36.7% | 100.0% |
| snapshot step 2100 | 0.8658 | 2.377 | 0.8034 | 2.233 | 93.2% | 80.3% | 100.0% | 99.3% | 43.3% | 100.0% |
| snapshot step 2400 | 0.8826 | 2.417 | 0.8179 | 2.266 | 93.9% | 82.3% | 100.0% | 99.3% | 35.0% | 100.0% |
| snapshot step 2700 | 0.8877 | 2.429 | 0.8209 | 2.273 | 93.7% | 81.0% | 100.0% | 100.0% | 40.0% | 100.0% |
| snapshot step 3000 | 0.8929 | 2.442 | 0.8245 | 2.281 | 95.0% | 85.0% | 100.0% | 100.0% | 40.0% | 100.0% |
