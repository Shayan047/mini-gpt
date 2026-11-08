# Question answering: pretrained vs. fine-tuned

Same 654 evaluation questions for every system (none used in fine-tuning). Greedy decoding. Accuracy = destination correct (and yes/no lead correct where asked); topic facts = key word present; out-of-scope = declined.

| question group | n | pretrained (Q/A prompt) | pretrained (chat prompt) | fine-tuned (chat prompt) |
|---|---:|---:|---:|---:|
| Known items, unseen wording | 294 | 100.0% | 27.9% | 99.7% |
| Known items, yes/no | 147 | 66.7% | 0.0% | 85.0% |
| Held-out items (never seen) | 60 | 50.0% | 23.3% | 38.3% |
| Free-form, known items | 35 | 45.7% | 17.1% | 48.6% |
| Free-form, held-out items | 15 | 26.7% | 13.3% | 26.7% |
| Topic facts, unseen wording | 55 | 80.0% | 21.8% | 76.4% |
| Bins & material rules | 28 | 46.4% | 21.4% | 67.9% |
| Out-of-scope (declines) | 20 | 0.0% | 0.0% | 75.0% |
| **All questions** | 654 | 76.3% | 18.7% | 82.3% |

### Behaviour

| measure | pretrained (Q/A prompt) | pretrained (chat prompt) | fine-tuned (chat prompt) |
|---|---:|---:|---:|
| destination correct (all item questions) | 90.2% | 23.0% | 88.7% |
| yes/no lead correct (yes/no questions) | 64.4% | 0.0% | 82.2% |
| names a different item first (item questions) | 13.2% | 68.2% | 6.0% |
| false declines: all in-domain questions | 0.0% | 0.0% | 3.6% |
| false declines: known items, unseen wording | 0.0% | 0.0% | 0.0% |
| false declines: held-out items | 0.0% | 0.0% | 11.7% |
| false declines: free-form questions | 0.0% | 0.0% | 20.0% |
| no destination stated, not a decline (item questions) | 1.1% | 50.3% | 0.7% |
