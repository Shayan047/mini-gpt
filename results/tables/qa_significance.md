# Paired significance tests (exact McNemar)

Each cell: `+gained / −lost` = questions only the second system / only the first system answered correctly, and the two-sided p-value. **Bold** = p < 0.05. Small groups (n ≤ 60) can only detect large differences.

| question group | n | pretrained → fine-tuned | pretrained → fine-tuned (no declines) | fine-tuned → no declines |
|---|---:|---|---|---|
| Known items, unseen wording | 294 | +0 / −1, p=1.000 | +0 / −0, p=1.000 | +1 / −0, p=1.000 |
| Known items, yes/no | 147 | **+35 / −8, p=0.000** | **+35 / −10, p=0.000** | +3 / −5, p=0.727 |
| Held-out items | 60 | +5 / −12, p=0.143 | +3 / −8, p=0.227 | +4 / −2, p=0.688 |
| Free-form, known items | 35 | +4 / −3, p=1.000 | +7 / −1, p=0.070 | +5 / −0, p=0.062 |
| Free-form, held-out items | 15 | +2 / −2, p=1.000 | +1 / −2, p=1.000 | +0 / −1, p=1.000 |
| Topic facts | 55 | +6 / −8, p=0.791 | +6 / −9, p=0.607 | +3 / −4, p=1.000 |
| Bins & material rules | 28 | +7 / −1, p=0.070 | +9 / −2, p=0.065 | +3 / −2, p=1.000 |
| Out-of-scope | 20 | **+15 / −0, p=0.000** | +0 / −0, p=1.000 | **+0 / −15, p=0.000** |
| All questions | 654 | **+74 / −35, p=0.000** | **+61 / −32, p=0.003** | +19 / −29, p=0.193 |
