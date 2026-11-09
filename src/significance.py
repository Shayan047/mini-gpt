"""Paired significance tests between QA systems (exact McNemar test on the same questions).

    python -m src.significance

Reads results/tables/qa_eval_answers.csv (written by src.evaluate_qa) and writes
results/tables/qa_significance.md. For each question group and each pair of systems it counts the
questions only system B got right (+) and only system A got right (-); the exact two-sided McNemar
p-value says how likely such an imbalance is if both systems were equally good.
"""
import csv
from collections import defaultdict
from math import comb

from src.kb import ROOT

GROUPS = {
    "Known items, unseen wording": ("core_item_unseen_wording",),
    "Known items, yes/no": ("core_item_unseen_wording_yesno",),
    "Held-out items": ("heldout_item", "heldout_item_yesno"),
    "Free-form, known items": ("freeform_core",),
    "Free-form, held-out items": ("freeform_heldout",),
    "Topic facts": ("topic_fact_unseen_wording",),
    "Bins & material rules": ("stream_unseen_wording", "rule_unseen_wording"),
    "Out-of-scope": ("out_of_scope",),
    "All questions": None,
}
PAIRS = [("pretrained (Q/A prompt)", "fine-tuned (chat prompt)", "pretrained → fine-tuned"),
         ("pretrained (Q/A prompt)", "fine-tuned, no declines", "pretrained → fine-tuned (no declines)"),
         ("fine-tuned (chat prompt)", "fine-tuned, no declines", "fine-tuned → no declines")]


def mcnemar(a, b):
    gained = sum(1 for x, y in zip(a, b) if not x and y)
    lost = sum(1 for x, y in zip(a, b) if x and not y)
    n = gained + lost
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(gained, lost) + 1)) / 2 ** n)
    return gained, lost, p


def main():
    rows = list(csv.DictReader(open(ROOT / "results" / "tables" / "qa_eval_answers.csv", encoding="utf-8")))
    by = defaultdict(dict)
    for r in rows:
        by[r["id"]][r["system"]] = r
    systems = {r["system"] for r in rows}
    pairs = [p for p in PAIRS if p[0] in systems and p[1] in systems]
    md = ["# Paired significance tests (exact McNemar)", "",
          "Each cell: `+gained / −lost` = questions only the second system / only the first system answered correctly, "
          "and the two-sided p-value. **Bold** = p < 0.05. Small groups (n ≤ 60) can only detect large differences.", "",
          "| question group | n | " + " | ".join(p[2] for p in pairs) + " |", "|---|---:|" + "---|" * len(pairs)]
    for label, cats in GROUPS.items():
        ids = [i for i, d in by.items() if cats is None or next(iter(d.values()))["category"] in cats]
        cells = []
        for a_sys, b_sys, _ in pairs:
            a = [by[i][a_sys]["correct"] == "1" for i in ids]
            b = [by[i][b_sys]["correct"] == "1" for i in ids]
            g, l, p = mcnemar(a, b)
            cell = f"+{g} / −{l}, p={p:.3f}"
            cells.append(f"**{cell}**" if p < 0.05 else cell)
        md.append(f"| {label} | {len(ids)} | " + " | ".join(cells) + " |")
    out = ROOT / "results" / "tables" / "qa_significance.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
