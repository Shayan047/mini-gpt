# Paired significance tests (exact McNemar)

Each cell: `+gained / −lost` = questions only the second system / only the first system answered correctly, and the two-sided p-value. **Bold** = p < 0.05. Small groups (n ≤ 60) can only detect large differences.

| question group | n | pretrained → fine-tuned |
|---|---:|---|
| Known items, unseen wording | 346 | **+1 / −12, p=0.003** |
| Known items, yes/no | 173 | **+43 / −2, p=0.000** |
| Held-out items | 60 | **+7 / −0, p=0.016** |
| Free-form, known items | 35 | +6 / −2, p=0.289 |
| Free-form, held-out items | 15 | +2 / −2, p=1.000 |
| Topic facts | 55 | +1 / −5, p=0.219 |
| Bins & material rules | 28 | +6 / −1, p=0.125 |
| Out-of-scope | 20 | **+15 / −0, p=0.000** |
| All questions | 732 | **+81 / −24, p=0.000** |
