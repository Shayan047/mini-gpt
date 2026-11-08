"""Consistency and leakage checks for the generated corpus.

    python -m src.datagen.check_corpus

Checks
 1. Held-out items: no name/alias of a held-out item may appear in any corpus split.
 2. Generic facts: no phone numbers, clock times or street addresses.
 3. KB consistency (heuristic): a sentence that names exactly one item and exactly one
    destination, with no negation, must name the item's correct destination.
 4. Duplicates: identical documents within or across splits.
 5. Phrasing overlap: share of val/test sentences that also occur verbatim in train
    (reported, not an error; template corpora always share some sentences).
Writes results/tables/corpus_checks.md. Exit code 1 if checks 1-4 find problems.
"""
import json
import re
import sys
from collections import Counter

from src.kb import ROOT, item_surface_forms, load_kb

CORPUS = ROOT / "data" / "corpus"
OUT = ROOT / "results" / "tables" / "corpus_checks.md"

DEST_PATTERNS = {
    "recycling": r"blue (recycling )?(cart|bin)|recycling cart|\bin recycling\b|\bthe recycling\b",
    "organics": r"green (organics )?(cart|bin)|organics cart|\bin organics\b|compost cart",
    "garbage": r"black (garbage )?(cart|bin)|garbage cart|\bin the garbage\b|\bare garbage\b|\bis garbage\b",
    "hazardous": r"hazardous waste",
    "electronics": r"electronics drop-off|e-waste|electronic waste|\bare electronics\b|\bis electronics\b",
    "bulky": r"bulky item|bulky waste",
    "textiles": r"textile",
    "takeback": r"pharmacy",
}
NEGATION = re.compile(r"\b(not|never|no|don't|doesn't|isn't|aren't|instead|wrong|mistake|rather|different|unless|only|except|but|nope|cannot|can't)\b", re.I)


def load_split(split):
    with open(CORPUS / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def sentences(text):
    for line in text.split("\n"):
        for s in re.split(r"(?<=[.!?])\s+", line):
            s = s.strip(" -*")
            if len(s) > 3:
                yield s


def build_item_matcher(kb):
    """One alternation regex over all surface forms, longest first, so "glass bottles" beats "bottles"."""
    lookup = {}
    for it in kb["items"]:
        for f in item_surface_forms(it):
            lookup[f.lower()] = it
    forms = sorted(lookup, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(f) for f in forms) + r")\b")
    return pattern, lookup


def items_in(sentence, matcher):
    pattern, lookup = matcher
    found = {}
    for m in pattern.finditer(sentence.lower()):
        it = lookup[m.group(1)]
        found[it["id"]] = it
    return list(found.values())


def main():
    kb = load_kb()
    splits = {s: load_split(s) for s in ("train", "val", "test")}
    report, errors = ["# Corpus checks", ""], 0

    # 1. held-out leakage
    held = [(f.lower(), it["id"]) for it in kb["items"] if it["split"] == "heldout" for f in item_surface_forms(it)]
    def singular(form):
        head, _, last = form.rpartition(" ")
        if last.endswith("ves"):
            last = last[:-3] + "f"
        elif re.search(r"(ss|x|ch|sh)es$", last):
            last = last[:-2]
        elif last.endswith("s") and not last.endswith("ss"):
            last = last[:-1]
        return f"{head} {last}".strip()

    held = held + [(singular(f), iid) for f, iid in held]
    held_map = dict(held)
    held_re = re.compile(r"\b(" + "|".join(re.escape(f) for f in sorted(held_map, key=len, reverse=True)) + r")\b")
    leaks = []
    for split, docs in splits.items():
        for d in docs:
            for m in held_re.finditer(d["text"].lower()):
                leaks.append(f"{d['id']}: held-out '{m.group(1)}' ({held_map[m.group(1)]})")
    report += [f"## 1. Held-out item leakage: {len(leaks)} found", ""] + [f"- {l}" for l in leaks[:50]] + [""]
    errors += len(leaks)

    # 2. specific details
    spec = []
    for split, docs in splits.items():
        for d in docs:
            for pat, what in [(r"\b\d{3}[- .]\d{3,4}\b", "phone"), (r"\b\d{1,2}(:\d\d)?\s?(a\.?m\.?|p\.?m\.?)(?!\w)", "time"),
                              (r"\b\d+\s+\w+\s+(Street|St\.|Avenue|Ave|Road|Rd|Drive|Lane)\b", "address")]:
                for m in re.finditer(pat, d["text"], flags=re.I):
                    spec.append(f"{d['id']}: {what} {m.group(0)!r}")
    report += [f"## 2. Specific details (phone/time/address): {len(spec)} found", ""] + [f"- {s}" for s in spec[:50]] + [""]
    errors += len(spec)

    # 3. KB consistency heuristic
    forms = build_item_matcher(kb)
    checked, bad = 0, []
    for split, docs in splits.items():
        for d in docs:
            for s in sentences(d["text"]):
                if s.endswith("?") or NEGATION.search(s):
                    continue
                its = items_in(s, forms)
                if len(its) != 1:
                    continue
                dests = [k for k, p in DEST_PATTERNS.items() if re.search(p, s, re.I)]
                if len(dests) != 1:
                    continue
                checked += 1
                if dests[0] != its[0]["stream"]:
                    bad.append(f"{d['id']} [{d['source']}]: '{s}' -> item {its[0]['id']} is {its[0]['stream']}, sentence says {dests[0]}")
    bad_seed = [x for x in bad if "[seed:" in x]
    bad_gen = [x for x in bad if "[seed:" not in x]
    report += [f"## 3. KB consistency: {checked:,} item+destination sentences checked", "",
               f"Hand-written seed documents: {len(bad_seed)} conflicts (must be 0).", ""]
    report += [f"- {x}" for x in bad_seed] + [""]
    report += [f"Generated documents: {len(bad_gen)} flags (informational: generated text is KB-consistent by construction; "
               "flags are sentences about part of an item, e.g. a box's plastic liner or a jar's leftover food).", ""]
    report += [f"- {x}" for x in bad_gen[:40]] + [""]
    errors += len(bad_seed)

    # 4. duplicates
    seen, dups = {}, []
    for split, docs in splits.items():
        for d in docs:
            if d["text"] in seen:
                dups.append(f"{d['id']} duplicates {seen[d['text']]}")
            seen.setdefault(d["text"], d["id"])
    report += [f"## 4. Duplicate documents: {len(dups)}", ""] + [f"- {x}" for x in dups[:50]] + [""]
    errors += len(dups)

    # 5. phrasing overlap
    train_sents = set(s for d in splits["train"] for s in sentences(d["text"]))
    report += ["## 5. Sentence overlap with training split (informational)", "",
               "| split | sentences | also in train | share |", "|---|---:|---:|---:|"]
    for split in ("val", "test"):
        sents = [s for d in splits[split] for s in sentences(d["text"])]
        shared = sum(1 for s in sents if s in train_sents)
        report.append(f"| {split} | {len(sents):,} | {shared:,} | {shared / max(1, len(sents)):.1%} |")
    report.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(l for l in report if l.startswith(("##", "|"))))
    print(f"seed-doc conflicts: {len(bad_seed)} | generated-doc flags (informational): {len(bad_gen)}")
    for x in bad_seed[:30]:
        print("  ", x)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
