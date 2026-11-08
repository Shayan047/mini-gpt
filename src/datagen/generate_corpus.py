"""Generate the pretraining corpus (train / val / test) from the knowledge base.

    python -m src.datagen.generate_corpus            # from the project root

Outputs
    data/corpus/{train,val,test}.jsonl      one document per line: id, split, type, source, text, items, facts, rules
    results/tables/corpus_stats.{csv,md}    documents / words / characters per split and per type
    results/tables/item_coverage.csv        how often each core item is mentioned in the training split
    results/tables/fact_coverage.csv        how often each topic fact / rule is used in the training split
    results/samples/corpus_samples.md       one example document per type (for the report)
"""
import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.kb import ROOT, item_surface_forms, load_kb
from .doctypes import DOC_TYPES, DocGen

SEED_DIR = ROOT / "data" / "seed_docs"
OUT_DIR = ROOT / "data" / "corpus"
TABLES = ROOT / "results" / "tables"
SAMPLES = ROOT / "results" / "samples"

DEFAULT_SIZES = {"train": 6000, "val": 320, "test": 320}
SEEDS = {"train": 1234, "val": 5678, "test": 9012}


def load_seed_docs():
    """Hand-written documents: data/seed_docs/*.txt, documents separated by a line '=====' ,
    first line of each document is 'type: <doctype>'."""
    docs = []
    for path in sorted(SEED_DIR.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        for i, block in enumerate(b.strip() for b in raw.split("\n=====\n")):
            if not block:
                continue
            first, _, body = block.partition("\n")
            dtype = first.split(":", 1)[1].strip() if first.lower().startswith("type:") else "seed"
            docs.append({"type": dtype, "text": body.strip(), "source": f"seed:{path.stem}:{i}",
                         "items": [], "facts": [], "rules": []})
    return docs


def split_seed_docs(seed_docs, rng):
    docs = seed_docs[:]
    rng.shuffle(docs)
    n = len(docs)
    n_eval = max(1, round(n * 0.05)) if n >= 20 else 0
    return {"val": docs[:n_eval], "test": docs[n_eval:2 * n_eval], "train": docs[2 * n_eval:]}


def generate(sizes, seed_docs):
    kb = load_kb()
    types, weights = zip(*DOC_TYPES.items())
    seed_split = split_seed_docs(seed_docs, random.Random(42))
    corpus = {}
    seen = {d["text"] for d in seed_docs}   # no document text may appear twice, within or across splits
    for split, n in sizes.items():
        rng = random.Random(SEEDS[split])
        gen = DocGen(kb, rng, mode="train" if split == "train" else "eval")
        docs = []
        while len(docs) < n:
            dtype = rng.choices(types, weights)[0]
            d = getattr(gen, dtype)()
            if d["text"] in seen:
                continue
            seen.add(d["text"])
            d["source"] = "generator"
            docs.append(d)
        docs += seed_split[split]
        rng.shuffle(docs)
        for i, d in enumerate(docs):
            d["id"] = f"{split}-{i:05d}"
            d["split"] = split
        corpus[split] = docs
    return kb, corpus


def mention_counts(kb, docs):
    """Count documents mentioning each core item by any surface form (word-boundary match)."""
    counts = Counter()
    texts = [d["text"].lower() for d in docs]
    for it in kb["items"]:
        if it["split"] != "core":
            continue
        pat = re.compile(r"\b(" + "|".join(re.escape(f.lower()) for f in item_surface_forms(it)) + r")\b")
        counts[it["id"]] = sum(len(pat.findall(t)) for t in texts)
    return counts


def write_outputs(kb, corpus):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    keys = ["id", "split", "type", "source", "text", "items", "facts", "rules"]
    for split, docs in corpus.items():
        with open(OUT_DIR / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps({k: d[k] for k in keys}, ensure_ascii=False) + "\n")

    # ---- per split / per type stats
    rows = []
    for split, docs in corpus.items():
        by_type = defaultdict(list)
        for d in docs:
            by_type[d["type"]].append(d)
        for t, ds in sorted(by_type.items()):
            rows.append({"split": split, "type": t, "docs": len(ds),
                         "words": sum(len(d["text"].split()) for d in ds),
                         "chars": sum(len(d["text"]) for d in ds)})
        rows.append({"split": split, "type": "ALL", "docs": len(docs),
                     "words": sum(len(d["text"].split()) for d in docs),
                     "chars": sum(len(d["text"]) for d in docs)})
    with open(TABLES / "corpus_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "type", "docs", "words", "chars"])
        w.writeheader()
        w.writerows(rows)
    md = ["| split | type | docs | words | chars |", "|---|---|---:|---:|---:|"]
    md += [f"| {r['split']} | {r['type']} | {r['docs']} | {r['words']:,} | {r['chars']:,} |" for r in rows]
    (TABLES / "corpus_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- coverage (training split)
    train = corpus["train"]
    mentions = mention_counts(kb, train)
    meta_items = Counter(i for d in train for i in d["items"])
    with open(TABLES / "item_coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "name", "stream", "docs_featuring", "text_mentions"])
        for it in kb["items"]:
            if it["split"] == "core":
                w.writerow([it["id"], it["name"], it["stream"], meta_items[it["id"]], mentions[it["id"]]])
    facts = Counter(f for d in train for f in d["facts"])
    rules = Counter(r for d in train for r in d["rules"])
    with open(TABLES / "fact_coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "kind", "docs_using"])
        for t in kb["topics"].values():
            for fa in t["facts"]:
                w.writerow([fa["id"], "topic_fact", facts[fa["id"]]])
        for r in kb["material_rules"]:
            w.writerow([r["id"], "material_rule", rules[r["id"]]])

    # ---- one sample per type
    seen, out = set(), ["# Corpus samples (one per document type, training split)", ""]
    for d in train:
        if d["type"] not in seen:
            seen.add(d["type"])
            out += [f"## {d['type']}  (`{d['id']}`, source: {d['source']})", "", "```text", d["text"], "```", ""]
    (SAMPLES / "corpus_samples.md").write_text("\n".join(out), encoding="utf-8")
    return rows, mentions, facts, rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=DEFAULT_SIZES["train"])
    ap.add_argument("--val", type=int, default=DEFAULT_SIZES["val"])
    ap.add_argument("--test", type=int, default=DEFAULT_SIZES["test"])
    args = ap.parse_args()
    seed_docs = load_seed_docs()
    kb, corpus = generate({"train": args.train, "val": args.val, "test": args.test}, seed_docs)
    rows, mentions, facts, rules = write_outputs(kb, corpus)
    for r in rows:
        if r["type"] == "ALL":
            print(f"{r['split']:5s}: {r['docs']:5d} docs  {r['words']:9,d} words  {r['chars']:10,d} chars")
    m = sorted(mentions.values())
    print(f"seed docs: {len(seed_docs)}")
    print(f"item text mentions (train): min {m[0]}, median {m[len(m)//2]}, max {m[-1]}")
    fv = sorted(facts.values()) or [0]
    rv = sorted(rules.values()) or [0]
    print(f"topic-fact docs (train): min {fv[0]}, median {fv[len(fv)//2]}, max {fv[-1]}")
    print(f"rule docs (train): min {rv[0]}, median {rv[len(rv)//2]}, max {rv[-1]}")


if __name__ == "__main__":
    main()
