# Vocabulary size comparison

All sizes trained on the same training split. 'Rarest merge frequency' is how often the last learned token's pair occurred in the corpus when it was merged: a proxy for how many training updates the rarest token's embedding receives.

| vocab size | train tokens | val tokens | rarest merge frequency | val word types encoded as one token | chosen |
|---:|---:|---:|---:|---:|:---:|
| 1,024 | 1,469,671 | 77,298 | 553 | 13% |  |
| 2,048 | 1,178,053 | 61,788 | 147 | 37% | yes |
| 3,072 | 1,083,089 | 56,667 | 52 | 66% |  |
| 4,096 | 1,049,247 | 54,812 | 19 | 90% |  |

The training split has only 3,358 distinct pre-token chunks (words with their leading space, punctuation runs). At 4,096 the vocabulary becomes nearly word-level and ~1,000 tokens are seen fewer than 50 times; at 2,048 every token is seen 100+ times and related words share sub-word pieces, at the cost of ~12% longer sequences.
