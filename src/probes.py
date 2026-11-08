"""Shared probing utilities: knowledge-base fact probes and a destination classifier.

Used to score the pretrained model (fact completion) and, later, the fine-tuned model (QA).
"""
import re

import torch

# Patterns that identify which disposal stream a piece of generated text names.
DEST_PATTERNS = {
    "recycling": r"blue (recycling )?(cart|bin)|recycling cart|\brecycl(ed|able|ing)\b(?! machines)",
    "organics": r"green (organics )?(cart|bin)|organics cart|\borganics\b|\bcompost",
    "garbage": r"black (garbage )?(cart|bin)|garbage cart|\bgarbage\b|\btrash\b",
    "hazardous": r"hazardous",
    "electronics": r"electronic|e-waste",
    "bulky": r"bulky",
    "textiles": r"textile",
    "takeback": r"pharmac",
}
_DEST_RE = {k: re.compile(p, re.I) for k, p in DEST_PATTERNS.items()}
_NEGATED = re.compile(r"\b(not|never)\b(\s+[\w-]+){0,4}\s*$", re.I)    # "not the ...", "never go in the ..."


def _first_destination(sentence):
    best = None
    for stream, rx in _DEST_RE.items():
        for m in rx.finditer(sentence):
            if _NEGATED.search(sentence[: m.start()]):
                continue                               # e.g. "not the recycling"
            if best is None or m.start() < best[0]:
                best = (m.start(), stream)
            break
    return best[1] if best else None


def classify_destination(text, max_sentences=2):
    """Return the stream named first in the answer, or None. Looks at the first `max_sentences`
    sentences of the first line-block and uses the first one that names a destination, so short
    leads like "Yes." or "That's right." are skipped. Mentions right after 'not/never' are ignored."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n", text.strip()) if s.strip()][:max_sentences]
    for s in sentences:
        d = _first_destination(s)
        if d:
            return d
    return None


def item_plural(name):
    if " and " in name:
        return True
    last = name.split()[-1].lower()
    return last.endswith("s") and not last.endswith("ss")


def item_frames(name):
    """Completion prompts that stop right before the destination is stated. Each frame mirrors a
    document style in the corpus so the probe tests knowledge, not unfamiliar formatting."""
    pl = item_plural(name)
    cap = name[:1].upper() + name[1:]
    do = "do" if pl else "does"
    belong = "belong" if pl else "belongs"
    return {
        "statement": f"{cap} {belong}",                          # "Pizza boxes belong ..."
        "directory": f"{cap}\nWhere:",                           # A-Z directory entry
        "faq": f"Q: Where {do} {name} go?\nA:",                   # FAQ entry
    }


FACT_CLOZES = [
    # (prompt, accepted continuations, KB fact id)
    ("The green organics cart is collected every", ["week"], "S02"),
    ("The blue recycling cart and the black garbage cart are collected on", ["alternating"], "S03"),
    ("Set your carts out the evening before or early on the", ["morning"], "S04"),
    ("Report a missed collection to the waste service by the end of the", ["next day"], "M02"),
    ("In a week with a public holiday, collection may be delayed by", ["one day"], "S06"),
    ("Place carts at the curb with the wheels facing", ["home"], "P01"),
    ("In winter, place carts on a cleared surface, not on top of", ["snowbank"], "P04"),
    ("Apartment and condo residents should use their building's designated", ["bulky item area"], "B03"),
    ("Leaving bulky items at the curb without a booking is treated as", ["illegal dumping"], "B04"),
    ("Drop-off of household hazardous waste and electronics is free for residents in normal household", ["quantities"], "D03"),
    ("Repeated contamination after warnings can lead to a", ["fine"], "F04"),
    ("A contaminated cart may be left unemptied with a", ["warning tag"], "F02"),
    ("Line your kitchen pail with newspaper or a", ["paper bag"], "O02"),
    ("To reduce odours, wrap food scraps in newspaper or keep them in the", ["freezer"], "O01"),
    ("Keep meat, fish, bones, dairy and oils out of a backyard compost bin; put them in the", ["green"], "H03"),
    ("Anything with a plug, cord or battery is", ["electronic"], "R10"),
    ("Batteries of every kind go to a household", ["hazardous"], "R11"),
    ("Glass bottles and jars go in the", ["blue"], "R06"),
    ("When you are unsure about an ordinary item, put it in the", ["garbage"], "W04"),
    ("Unused or expired medications should be returned to a", ["pharmacy"], "R16"),
    ("Organics are turned into", ["compost"], "W02"),
    ("Transport hazardous products upright, sealed and separate from passengers, ideally in the", ["trunk"], "D04"),
    ("Flatten moving boxes for recycling, or give them to someone else who is", ["moving"], "V03"),
    ("Soft plastic such as bags, wrap, wrappers and pouches goes in the", ["garbage"], "R04"),
    ("Clean paper and cardboard go in the", ["blue"], "R01"),
]


@torch.no_grad()
def greedy(model, tok, prompt, max_new_tokens=40, device="cpu"):
    ids = torch.tensor([tok.encode(prompt, allow_special=True)], device=device)
    out = model.generate(ids, max_new_tokens, temperature=0, stop_ids=(tok.eot_id,))
    return tok.decode(out[0, ids.shape[1]:].tolist(), skip_special=True)
