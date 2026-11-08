"""Build the supervised fine-tuning (SFT) set and the QA evaluation set from the knowledge base.

    python -m src.datagen.build_sft

Outputs
    data/sft/train.jsonl, data/sft/val.jsonl   {"question", "answer", "category", ...}
    data/eval/qa_eval.jsonl                     evaluation questions with expected answers (never trained on)
    results/tables/sft_data_stats.md            counts per category + leakage checks
    results/samples/sft_examples.md             example training pairs and evaluation questions

Separation between fine-tuning and evaluation questions
  * item questions: evaluation uses only the eval-only question templates (expand.Bank), which appear in
    neither the training corpus nor the SFT set
  * topic / stream / rule questions: the last wording of every list is reserved for evaluation
  * the 20 held-out items, the hand-written free-form questions and the evaluation off-topic questions
    are never used in fine-tuning
"""
import json
import random
import re
from collections import Counter

from src.kb import ROOT, load_kb
from src.tokenizer import BPETokenizer
from . import lexicon as L
from . import questions as Q
from . import sft_content as C
from .doctypes import DocGen
from .expand import Bank, cap

SFT_DIR, EVAL_DIR = ROOT / "data" / "sft", ROOT / "data" / "eval"
N_PER_ITEM, N_PER_FACT, N_PER_STREAM, N_PER_RULE = 64, 22, 30, 20
N_OOS_GENERAL, N_OOS_COMMERCIAL = 10, 15
VAL_FRACTION = 0.04
SFT_ANSWER = {"p_note": 0.45, "p_reason": 0.15}     # keep fine-tuning answers short and consistent
CURBSIDE = list(Q.ASKED_BIN)


def vary(q, rng):
    """Natural, typo-free variation of a training question."""
    p = rng.choice(C.QUESTION_PREFIXES)
    if p and not p.rstrip().endswith((".", "!", ":")):
        q = q[:1].lower() + q[1:] if q[1:2].islower() else q
    q = p + q
    r = rng.random()
    if r < 0.10:
        q = q.lower()
    elif r < 0.15:
        q = q.rstrip("?")
    return q


def sft_format(question, answer):
    return f"<|user|> {question}<|assistant|> {answer}<|endoftext|>"


def item_example(gen, item, rng, kind):
    ctx = gen.ictx(item)
    if kind == "yesno":
        if item["stream"] in CURBSIDE and rng.random() < 0.5:
            asked = item["stream"]
        else:
            asked = rng.choice([s for s in CURBSIDE if s != item["stream"]])
        c = dict(ctx, asked=rng.choice(Q.ASKED_BIN[asked]))
        q = gen.r("q_item_yesno", Q.ITEM_YESNO_QUESTIONS, c)
        lead = rng.choice(Q.YES_LEADS if asked == item["stream"] else Q.NO_LEADS)
        return q, gen.answer(item, ctx, lead, **SFT_ANSWER), ("yes" if asked == item["stream"] else "no")
    bank = ("q_item_prep", Q.ITEM_PREP_QUESTIONS) if kind == "prep" else ("q_item", Q.ITEM_QUESTIONS)
    return gen.r(*bank, ctx), gen.answer(item, ctx, **SFT_ANSWER), None


def build_sft(kb, rng):
    gen = DocGen(kb, rng, mode="train")             # draws only from training templates
    core = [i for i in kb["items"] if i["split"] == "core"]
    ex = []
    for it in core:
        for _ in range(N_PER_ITEM):
            kind = rng.choices(["open", "yesno", "prep"], [0.63, 0.25, 0.12])[0]
            q, a, yn = item_example(gen, it, rng, kind)
            ex.append({"category": f"item_{kind}", "question": vary(q, rng), "answer": a, "item_id": it["id"]})
    for fid in gen.facts:
        for _ in range(N_PER_FACT):
            q = rng.choice(Q.FACT_QUESTIONS[fid][:-1])
            ex.append({"category": "topic_fact", "question": vary(q, rng), "answer": gen.fact_text(fid), "fact_id": fid})
    for s, qs in C.STREAM_QUESTIONS.items():
        items = [i for i in core if i["stream"] == s]
        for _ in range(N_PER_STREAM):
            parts = [rng.choice(L.STREAM_INTRO[s])]
            if rng.random() < 0.5:
                parts.append(rng.choice(kb["streams"][s]["rules"]))
            if rng.random() < 0.6:
                parts.append("For example: " + ", ".join(i["name"] for i in rng.sample(items, min(3, len(items)))) + ".")
            ex.append({"category": "stream", "question": vary(rng.choice(qs[:-1]), rng), "answer": " ".join(parts), "stream": s})
    for rid, qs in C.RULE_QUESTIONS.items():
        exs = [i for i in core if i["rule"] == rid]
        for _ in range(N_PER_RULE):
            a = gen.rule_text(rid)
            if exs and rng.random() < 0.6:
                a += " For example: " + ", ".join(i["name"] for i in rng.sample(exs, min(len(exs), rng.randint(1, 3)))) + "."
            ex.append({"category": "rule", "question": vary(rng.choice(qs[:-1]), rng), "answer": a, "rule_id": rid})
    for q in C.OOS_GENERAL_TRAIN:
        for _ in range(N_OOS_GENERAL):
            ex.append({"category": "out_of_scope", "question": vary(q, rng), "answer": rng.choice(C.DECLINE_GENERAL)})
    for q in C.OOS_COMMERCIAL_TRAIN:
        for _ in range(N_OOS_COMMERCIAL):
            ex.append({"category": "out_of_scope", "question": vary(q, rng), "answer": rng.choice(C.DECLINE_COMMERCIAL)})
    # de-duplicate identical pairs
    seen, out = set(), []
    for e in ex:
        key = (e["question"], e["answer"])
        if key not in seen:
            seen.add(key)
            out.append(e)
    rng.shuffle(out)
    n_val = int(len(out) * VAL_FRACTION)
    return out[n_val:], out[:n_val]


def build_eval(kb, rng):
    gen = DocGen(kb, rng, mode="eval")
    items = {i["id"]: i for i in kb["items"]}
    core = [i for i in kb["items"] if i["split"] == "core"]
    held = [i for i in kb["items"] if i["split"] == "heldout"]
    q_open = Bank("q_item", Q.ITEM_QUESTIONS)
    q_yesno = Bank("q_item_yesno", Q.ITEM_YESNO_QUESTIONS)
    from .expand import expand
    ev = []

    def ref(item, ctx, lead=""):
        return gen.answer(item, ctx, lead, **SFT_ANSWER)

    def add_item_questions(it, category, open_templates, n_open, yesno_templates):
        ctx = gen.ictx(it, it["name"])
        for t in rng.sample(open_templates, min(n_open, len(open_templates))):
            ev.append({"category": category, "question": expand(t, rng, ctx), "item_id": it["id"],
                       "expected_stream": it["stream"], "reference": ref(it, ctx)})
        if yesno_templates:
            asked = it["stream"] if (it["stream"] in CURBSIDE and rng.random() < 0.5) \
                else rng.choice([s for s in CURBSIDE if s != it["stream"]])
            c = dict(ctx, asked=rng.choice(Q.ASKED_BIN[asked]))
            yn = "yes" if asked == it["stream"] else "no"
            ev.append({"category": category + "_yesno", "question": expand(rng.choice(yesno_templates), rng, c),
                       "item_id": it["id"], "expected_stream": it["stream"], "expected_yesno": yn,
                       "reference": ref(it, ctx, rng.choice(Q.YES_LEADS if yn == "yes" else Q.NO_LEADS))})

    for it in core:                                   # unseen question formulations, known items
        add_item_questions(it, "core_item_unseen_wording", q_open.eval_only, 2, q_yesno.eval_only)
    for it in held:                                   # items never seen in any training data
        add_item_questions(it, "heldout_item", q_open.eval_only + q_open.train, 2, q_yesno.templates)
    for fid in gen.facts:
        ev.append({"category": "topic_fact_unseen_wording", "question": Q.FACT_QUESTIONS[fid][-1], "fact_id": fid,
                   "keywords": C.FACT_KEYWORDS[fid], "reference": gen.fact_text(fid)})
    for s, qs in C.STREAM_QUESTIONS.items():
        ev.append({"category": "stream_unseen_wording", "question": qs[-1], "expected_stream": s,
                   "reference": rng.choice(L.STREAM_INTRO[s])})
    for rid, qs in C.RULE_QUESTIONS.items():
        r = next(x for x in kb["material_rules"] if x["id"] == rid)
        ev.append({"category": "rule_unseen_wording", "question": qs[-1], "rule_id": rid,
                   "expected_stream": r["stream"], "reference": r["text"]})
    for q, iid, yn in C.FREEFORM_EVAL:
        it = items[iid]
        e = {"category": "freeform_" + ("heldout" if it["split"] == "heldout" else "core"), "question": q,
             "item_id": iid, "expected_stream": it["stream"], "reference": ref(it, gen.ictx(it, it["name"]))}
        if yn:
            e["expected_yesno"] = yn
        ev.append(e)
    for q in C.OOS_GENERAL_EVAL:
        ev.append({"category": "out_of_scope", "question": q, "expect_decline": True, "reference": C.DECLINE_GENERAL[0]})
    for q in C.OOS_COMMERCIAL_EVAL:
        ev.append({"category": "out_of_scope", "question": q, "expect_decline": True, "reference": C.DECLINE_COMMERCIAL[0]})
    for i, e in enumerate(ev):
        e["id"] = f"qa-{i:04d}"
    return ev


def check_fact_keywords(kb):
    facts = {f["id"]: f["text"] for t in kb["topics"].values() for f in t["facts"]}
    bad = []
    for fid, text in facts.items():
        for wording in [text] + L.FACTS[fid]:
            if not any(k in wording.lower() for k in C.FACT_KEYWORDS[fid]):
                bad.append(f"{fid}: {wording}")
    assert not bad, "fact keywords miss these wordings:\n" + "\n".join(bad)


def main():
    kb = load_kb()
    check_fact_keywords(kb)
    tok = BPETokenizer.load(ROOT / "data" / "tokenizer" / "tokenizer.json")
    train, val = build_sft(kb, random.Random(2024))
    ev = build_eval(kb, random.Random(7))

    # ---- leakage checks
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    sft_qs = {norm(e["question"]) for e in train + val}
    overlap = [e for e in ev if norm(e["question"]) in sft_qs]
    held_ids = {i["id"] for i in kb["items"] if i["split"] == "heldout"}
    held_forms = [f.lower() for i in kb["items"] if i["split"] == "heldout" for f in [i["name"], *i["aliases"]]]
    held_leak = [e for e in train + val if any(re.search(r"\b" + re.escape(f) + r"\b", (e["question"] + " " + e["answer"]).lower())
                                               for f in held_forms)]
    lengths = [len(tok.encode(sft_format(e["question"], e["answer"]), allow_special=True)) for e in train + val]

    SFT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with open(SFT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(EVAL_DIR / "qa_eval.jsonl", "w", encoding="utf-8") as f:
        for e in ev:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # ---- stats
    ctr, ectr = Counter(e["category"] for e in train), Counter(e["category"] for e in ev)
    md = ["# Fine-tuning and evaluation data", "",
          f"Format: `<|user|> question<|assistant|> answer<|endoftext|>`; loss on answer tokens only.", "",
          f"**SFT train:** {len(train):,} pairs · **SFT val:** {len(val):,} pairs · "
          f"tokens per example: mean {sum(lengths) / len(lengths):.0f}, max {max(lengths)} (context 256) · "
          f"total SFT train tokens ≈ {sum(lengths[:len(train)]):,}", "",
          "| SFT category | train pairs |", "|---|---:|"] + [f"| {k} | {v:,} |" for k, v in sorted(ctr.items())]
    md += ["", f"**Evaluation set:** {len(ev)} questions (never used in fine-tuning; same questions for both models)", "",
           "| evaluation category | questions |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in sorted(ectr.items())]
    md += ["", "## Leakage checks", "",
           f"- Evaluation questions that also appear (normalized) in the SFT data: **{len(overlap)}**",
           f"- SFT pairs mentioning a held-out item: **{len(held_leak)}**",
           f"- SFT examples longer than the 256-token context: **{sum(l > 256 for l in lengths)}**"]
    (ROOT / "results" / "tables" / "sft_data_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    ex_md = ["# Fine-tuning examples", ""]
    shown = set()
    for e in train:
        if e["category"] not in shown:
            shown.add(e["category"])
            ex_md += [f"**{e['category']}**  ", f"Q: {e['question']}  ", f"A: {e['answer']}", ""]
    ex_md += ["# Evaluation questions (one per category)", ""]
    shown = set()
    for e in ev:
        if e["category"] not in shown:
            shown.add(e["category"])
            exp = {k: e[k] for k in ("expected_stream", "expected_yesno", "keywords", "expect_decline") if k in e}
            ex_md += [f"**{e['category']}**  ", f"Q: {e['question']}  ", f"Expected: `{exp}`", ""]
    (ROOT / "results" / "samples" / "sft_examples.md").write_text("\n".join(ex_md), encoding="utf-8")

    print("\n".join(md))
    assert not overlap and not held_leak and max(lengths) <= 256, "leakage or length check failed"


if __name__ == "__main__":
    main()
