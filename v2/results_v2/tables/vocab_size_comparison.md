# Vocabulary size comparison

All sizes trained on the same training split. 'Rarest merge frequency' is how often the last learned token's pair occurred in the corpus when it was merged: a proxy for how many training updates the rarest token's embedding receives.

| vocab size | train tokens | val tokens | rarest merge frequency | val word types encoded as one token | chosen |
|---:|---:|---:|---:|---:|:---:|
| 1,024 | 3,099,703 | 130,883 | 1167 | 11% |  |
| 2,048 | 2,488,984 | 105,093 | 310 | 33% | yes |
| 3,072 | 2,286,895 | 96,426 | 109 | 57% |  |
| 4,096 | 2,216,849 | 93,216 | 44 | 82% |  |

The training split has only 3,575 distinct pre-token chunks (words with their leading space, punctuation runs). At 4,096 the vocabulary becomes nearly word-level and ~1,000 tokens are seen fewer than 50 times; at 2,048 every token is seen 100+ times and related words share sub-word pieces, at the cost of ~12% longer sequences.
