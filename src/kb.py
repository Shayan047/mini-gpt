"""Load and validate the fictional knowledge base (data/kb/knowledge_base.json).

Run directly to validate and print statistics:
    python src/kb.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB_PATH = ROOT / "data" / "kb" / "knowledge_base.json"


def load_kb(path=KB_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def item_surface_forms(item):
    """All names a resident might use for an item (name + aliases)."""
    return [item["name"], *item.get("aliases", [])]


def validate(kb):
    """Return a list of error strings; empty list means the KB is consistent."""
    errors = []
    streams = kb["streams"]
    rules = {r["id"]: r for r in kb["material_rules"]}
    preps = kb["prep_codes"]

    ids = [it["id"] for it in kb["items"]]
    for dup, n in Counter(ids).items():
        if n > 1:
            errors.append(f"duplicate item id: {dup}")

    for it in kb["items"]:
        tag = it["id"]
        if it["stream"] not in streams:
            errors.append(f"{tag}: unknown stream {it['stream']}")
        if it["split"] not in ("core", "heldout"):
            errors.append(f"{tag}: bad split {it['split']}")
        rule = it.get("rule")
        if rule is not None:
            if rule not in rules:
                errors.append(f"{tag}: unknown rule {rule}")
            elif rules[rule]["stream"] != it["stream"]:
                errors.append(f"{tag}: rule {rule} points to {rules[rule]['stream']} but item is {it['stream']}")
        for p in it["prep"]:
            if p not in preps:
                errors.append(f"{tag}: unknown prep code {p}")

    # Held-out names must not leak into core items (they would then appear in the corpus).
    core_text = " | ".join(
        f.lower() for it in kb["items"] if it["split"] == "core" for f in item_surface_forms(it)
    )
    for it in kb["items"]:
        if it["split"] != "heldout":
            continue
        for form in item_surface_forms(it):
            if re.search(r"\b" + re.escape(form.lower()) + r"\b", core_text):
                errors.append(f"held-out form '{form}' ({it['id']}) also appears in a core item name/alias")

    # Generic-ness check: no phone numbers, street addresses or clock times.
    raw = json.dumps(kb)
    for pat, what in [(r"\b\d{3}[- .]\d{3,4}\b", "phone number"),
                      (r"\b\d{1,2}(:\d\d)?\s?(a\.?m\.?|p\.?m\.?)", "clock time"),
                      (r"\b\d+\s+\w+\s+(Street|St\.|Avenue|Ave|Road|Rd)\b", "street address")]:
        for m in re.finditer(pat, raw, flags=re.I):
            errors.append(f"specific {what} found: {m.group(0)!r}")
    return errors


def stats(kb):
    items = kb["items"]
    split = Counter(it["split"] for it in items)
    by_stream = Counter((it["stream"], it["split"]) for it in items)
    n_facts = sum(len(t["facts"]) for t in kb["topics"].values())
    lines = [
        f"streams: {len(kb['streams'])}",
        f"material rules: {len(kb['material_rules'])}",
        f"prep codes: {len(kb['prep_codes'])}",
        f"items: {len(items)} (core {split['core']}, held-out {split['heldout']})",
        f"surface forms (names + aliases): {sum(len(item_surface_forms(i)) for i in items)}",
        f"topics: {len(kb['topics'])}, topic facts: {n_facts}",
        "items per stream (core / held-out):",
    ]
    for s in kb["streams"]:
        lines.append(f"  {s:12s} {by_stream[(s, 'core')]:3d} / {by_stream[(s, 'heldout')]}")
    return "\n".join(lines)


if __name__ == "__main__":
    kb = load_kb()
    errs = validate(kb)
    print(stats(kb))
    if errs:
        print(f"\n{len(errs)} problem(s):")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print("\nKB valid.")
