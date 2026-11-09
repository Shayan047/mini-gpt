"""Question-answering evaluation: the SAME 654 questions for the pretrained and the fine-tuned model
(spec section 7 and the research question).

    python -m src.evaluate_qa [--pretrained checkpoints/pretrain/ckpt_best.pt] [--finetuned checkpoints/sft/ckpt_best.pt]

Systems compared (greedy decoding, up to 80 new tokens)
  pretrained (Q/A prompt)   "Q: <question>\\nA:"  - the FAQ format the pretrained model saw in its corpus;
                            the answer is cut at the first blank line / next "Q:" (it would continue the FAQ)
  pretrained (chat prompt)  "<|user|> <question><|assistant|>" - sanity row; tokens it has never seen
  fine-tuned (chat prompt)  "<|user|> <question><|assistant|>" - the format it was fine-tuned on

Scoring (automatic, see src/probes.py)
  item / stream / rule questions   destination named first in the answer == expected stream
  yes/no questions                 also: the answer's lead word (yes/no) is correct
  topic facts                      answer contains one of the fact's key words
  out-of-scope                     answer is a decline; in-domain answers that decline count as false declines
  'names another item'             the first knowledge-base item named in the answer is not the one asked about

Outputs: results/tables/qa_eval.{md,csv}, results/tables/qa_eval_answers.csv,
         results/figures/qa_before_after.png, results/samples/qa_examples.md
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

from src.datagen.sft_content import DECLINE_MARKERS
from src.kb import ROOT, item_surface_forms, load_kb
from src.model import load_checkpoint
from src.probes import classify_destination
from src.tokenizer import BPETokenizer

TABLES, FIGURES, SAMPLES = ROOT / "results" / "tables", ROOT / "results" / "figures", ROOT / "results" / "samples"
GROUPS = [  # (label, categories)
    ("Known items, unseen wording", ["core_item_unseen_wording"]),
    ("Known items, yes/no", ["core_item_unseen_wording_yesno"]),
    ("Held-out items (never seen)", ["heldout_item", "heldout_item_yesno"]),
    ("Free-form, known items", ["freeform_core"]),
    ("Free-form, held-out items", ["freeform_heldout"]),
    ("Topic facts, unseen wording", ["topic_fact_unseen_wording"]),
    ("Bins & material rules", ["stream_unseen_wording", "rule_unseen_wording"]),
    ("Out-of-scope (declines)", ["out_of_scope"]),
]
YES = re.compile(r"^\W*(yes|yep|yeah|correct|that's right|that is right|sure)\b", re.I)
NO = re.compile(r"^\W*(no|nope|not quite|actually,? no|sorry,? no)\b", re.I)


def pct(v):
    return "–" if v is None else f"{v:.1%}"


def yesno_of(answer):
    if YES.search(answer):
        return "yes"
    if NO.search(answer):
        return "no"
    return None


def build_item_matcher(kb):
    lookup = {f.lower(): it["id"] for it in kb["items"] for f in item_surface_forms(it)}
    forms = sorted(lookup, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(f) for f in forms) + r")\b"), lookup


def first_item(text, matcher):
    pat, lookup = matcher
    m = pat.search(text.lower())
    return lookup[m.group(1)] if m else None


@torch.no_grad()
def answer(model, tok, question, system, max_new_tokens=80):
    if system == "qa_prompt":
        prompt, special = f"Q: {question}\nA:", False
    else:
        prompt, special = f"<|user|> {question}<|assistant|>", True
    ids = torch.tensor([tok.encode(prompt, allow_special=special)])
    out = model.generate(ids, max_new_tokens, temperature=0, stop_ids=(tok.eot_id,))
    text = tok.decode(out[0, ids.shape[1]:].tolist(), skip_special=True)
    if system == "qa_prompt":                                 # cut where the FAQ moves to the next entry
        text = re.split(r"\n\s*\n|\nQ:|\nQuestion:", text)[0]
    return text.strip()


def score(e, ans, matcher):
    r = {"declined": int(any(m in ans.lower() for m in DECLINE_MARKERS))}
    if e.get("expect_decline"):
        r["correct"] = r["declined"]
        return r
    if "keywords" in e:
        r["correct"] = int(any(k in ans.lower() for k in e["keywords"]))
        return r
    pred = "declined" if r["declined"] else classify_destination(ans)   # a decline mentioning "recycling" is not an answer
    r["predicted_stream"] = pred or "none"
    r["destination_correct"] = int(pred == e["expected_stream"])
    r["correct"] = r["destination_correct"]
    if "expected_yesno" in e:
        yn = yesno_of(ans)
        r["yesno_given"] = yn or "none"
        r["yesno_correct"] = int(yn == e["expected_yesno"])
        r["correct"] = int(r["destination_correct"] and r["yesno_correct"])
    if "item_id" in e:
        fi = first_item(ans, matcher)
        r["first_item_named"] = fi or ""
        r["names_other_item"] = int(fi is not None and fi != e["item_id"])
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default=str(ROOT / "checkpoints" / "pretrain" / "ckpt_best.pt"))
    ap.add_argument("--finetuned", default=str(ROOT / "checkpoints" / "sft" / "ckpt_best.pt"))
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions (debugging)")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score the answers saved in results/tables/qa_eval_answers.csv instead of generating")
    ap.add_argument("--add", nargs=2, action="append", metavar=("LABEL", "CHECKPOINT"), default=[],
                    help="evaluate an extra fine-tuned model (chat prompt) and merge it with the saved answers of "
                         "the other systems, e.g. --add \"fine-tuned, no declines\" checkpoints/sft_no_decline/ckpt_best.pt")
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    kb = load_kb()
    tok = BPETokenizer.load(ROOT / "data" / "tokenizer" / "tokenizer.json")
    ev = [json.loads(l) for l in open(ROOT / "data" / "eval" / "qa_eval.jsonl", encoding="utf-8")][: a.limit]
    matcher = build_item_matcher(kb)
    systems = [("pretrained (Q/A prompt)", a.pretrained, "qa_prompt"),
               ("pretrained (chat prompt)", a.pretrained, "chat"),
               ("fine-tuned (chat prompt)", a.finetuned, "chat")]
    def row(name, e, ans):
        return {"system": name, "id": e["id"], "category": e["category"], "item_id": e.get("item_id", ""),
                "question": e["question"], "answer": ans, "reference": e.get("reference", ""), **score(e, ans, matcher)}

    rows = []
    added = [(label, path, "chat") for label, path in a.add]
    if a.rescore or added:                    # reuse saved answers; regenerate only the added systems
        by_id = {e["id"]: e for e in ev}
        for old in csv.DictReader(open(TABLES / "qa_eval_answers.csv", encoding="utf-8")):
            if old["id"] in by_id and old["system"] not in {s[0] for s in added}:
                rows.append(row(old["system"], by_id[old["id"]], old["answer"]))
    to_generate = added if (a.rescore or added) else systems
    if to_generate:
        for name, path, fmt in to_generate:
            model, _, ck = load_checkpoint(path, "cpu")
            print(f"== {name}: {path} (step {ck['step']})")
            for i, e in enumerate(ev):
                rows.append(row(name, e, answer(model, tok, e["question"], fmt)))
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(ev)}")

    # ---- summary table
    names = list(dict.fromkeys(r["system"] for r in rows))     # every system present, in order
    def rate(sel, key):
        vals = [r[key] for r in sel if key in r]
        return sum(vals) / len(vals) if vals else None
    table, md = [], ["# Question answering: pretrained vs. fine-tuned", "",
                     f"Same {len(ev)} evaluation questions for every system (none used in fine-tuning). Greedy decoding. "
                     "Accuracy = destination correct (and yes/no lead correct where asked); topic facts = key word present; "
                     "out-of-scope = declined.", "",
                     "| question group | n | " + " | ".join(names) + " |", "|---|---:|" + "---:|" * len(names)]
    for label, cats in GROUPS + [("**All questions**", None)]:
        line = {"group": label.strip("*")}
        for n in names:
            sel = [r for r in rows if r["system"] == n and (cats is None or r["category"] in cats)]
            line["n"] = len(sel)
            line[n] = rate(sel, "correct")
        table.append(line)
        md.append(f"| {label} | {line['n']} | " + " | ".join(pct(line[n]) for n in names) + " |")
    md += ["", "### Behaviour", "", "| measure | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    behaviour = [
        ("destination correct (all item questions)", lambda r: "destination_correct" in r and r["item_id"], "destination_correct"),
        ("yes/no lead correct (yes/no questions)", lambda r: "yesno_correct" in r, "yesno_correct"),
        ("names a different item first (item questions)", lambda r: "names_other_item" in r, "names_other_item"),
        ("false declines: all in-domain questions", lambda r: r["category"] != "out_of_scope", "declined"),
        ("false declines: known items, unseen wording", lambda r: r["category"].startswith("core_item"), "declined"),
        ("false declines: held-out items", lambda r: r["category"].startswith("heldout"), "declined"),
        ("false declines: free-form questions", lambda r: r["category"].startswith("freeform"), "declined"),
        ("no destination stated, not a decline (item questions)", lambda r: "predicted_stream" in r and r["item_id"],
         None),
    ]
    for label, filt, key in behaviour:
        vals = []
        for n in names:
            sel = [r for r in rows if r["system"] == n and filt(r)]
            v = (sum(r["predicted_stream"] == "none" for r in sel) / len(sel) if sel else None) if key is None else rate(sel, key)
            vals.append(v)
            table.append({"group": label, "n": len(sel), n: v})
        md.append(f"| {label} | " + " | ".join(pct(v) for v in vals) + " |")
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "qa_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(TABLES / "qa_eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "n"] + names)
        w.writeheader()
        merged = defaultdict(dict)
        for t in table:
            merged[t["group"]].update(t)
        w.writerows(merged.values())
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("system", "id", "category", "question", "answer"), k))
    with open(TABLES / "qa_eval_answers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [g for g, _ in GROUPS]
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    colors = ["#9aa5b1", "#d0d7de", "#1f6feb", "#bf8700", "#2da44e"]
    width = 0.8 / len(names)
    for k, n in enumerate(names):
        vals = [next(t[n] for t in table if t["group"] == g) or 0 for g in labels]
        bars = ax.bar([i + (k - (len(names) - 1) / 2) * width for i in range(len(labels))], vals, width, label=n, color=colors[k],
                      edgecolor="#57606a", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(", ", ",\n").replace(" (", "\n(") for l in labels], fontsize=7)
    ax.set_ylim(0, 1.1); ax.set_ylabel("accuracy"); ax.legend(frameon=False, fontsize=8, ncol=len(names), loc="upper left")
    ax.set_title("Same evaluation questions before and after supervised fine-tuning"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "qa_before_after.png")

    # ---- side-by-side examples: per group, first question where verdicts differ + one where both fail
    by_id = defaultdict(dict)
    for r in rows:
        by_id[r["id"]][r["system"]] = r
    ex = ["# QA examples: same question, pretrained vs. fine-tuned", "",
          "Greedy decoding. ✅/❌ = automatic verdict. Pretrained shown with its native `Q:/A:` prompt.", ""]
    pre, fin = "pretrained (Q/A prompt)", "fine-tuned (chat prompt)"
    for label, cats in GROUPS:
        ids = [i for i, d in by_id.items() if d[pre]["category"] in cats]
        improved = [i for i in ids if by_id[i][fin]["correct"] and not by_id[i][pre]["correct"]][:3]
        both_bad = [i for i in ids if not by_id[i][fin]["correct"]][:2]
        ex += [f"## {label}", ""]
        for tag, sel in (("Fixed by fine-tuning", improved), ("Fine-tuned model still wrong", both_bad)):
            for i in sel:
                d = by_id[i]
                ex += [f"**{tag}** — Q: *{d[fin]['question']}*  ",
                       f"- pretrained {'✅' if d[pre]['correct'] else '❌'}: {d[pre]['answer'][:220]}  ",
                       f"- fine-tuned {'✅' if d[fin]['correct'] else '❌'}: {d[fin]['answer'][:220]}", ""]
    (SAMPLES / "qa_examples.md").write_text("\n".join(ex), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
