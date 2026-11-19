# Question answering: pretrained vs. fine-tuned

Same 732 evaluation questions for every system (none used in fine-tuning). Greedy decoding. Accuracy = destination correct (and yes/no lead correct where asked); topic facts = key word present; out-of-scope = declined.

| question group | n | pretrained (Q/A prompt) | pretrained (chat prompt) | fine-tuned (chat prompt) |
|---|---:|---:|---:|---:|
| Known items, unseen wording | 346 | 99.7% | 65.3% | 96.5% |
| Known items, yes/no | 173 | 71.1% | 0.0% | 94.8% |
| Held-out items (never seen) | 60 | 40.0% | 30.0% | 51.7% |
| Free-form, known items | 35 | 62.9% | 28.6% | 74.3% |
| Free-form, held-out items | 15 | 40.0% | 26.7% | 40.0% |
| Topic facts, unseen wording | 55 | 98.2% | 14.5% | 90.9% |
| Bins & material rules | 28 | 71.4% | 50.0% | 89.3% |
| Out-of-scope (declines) | 20 | 0.0% | 0.0% | 75.0% |
| **All questions** | 732 | 81.1% | 38.3% | 88.9% |

### Behaviour

| measure | pretrained (Q/A prompt) | pretrained (chat prompt) | fine-tuned (chat prompt) |
|---|---:|---:|---:|
| destination correct (all item questions) | 91.7% | 57.7% | 90.8% |
| yes/no lead correct (yes/no questions) | 69.9% | 0.0% | 93.2% |
| names a different item first (item questions) | 8.9% | 79.2% | 9.1% |
| false declines: all in-domain questions | 0.0% | 0.0% | 1.5% |
| false declines: known items, unseen wording | 0.0% | 0.0% | 0.6% |
| false declines: held-out items | 0.0% | 0.0% | 1.7% |
| false declines: free-form questions | 0.0% | 0.0% | 14.0% |
| no destination stated, not a decline (item questions) | 0.8% | 8.7% | 0.6% |
