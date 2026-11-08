"""Tiny template language used by the corpus generator.

Syntax inside a template string:
    [a|b|c]     choose one alternative at random (alternatives may be empty, may nest)
    {slot}      replaced with ctx[slot] (a string, or a callable(rng) -> str)
    {v:go}      verb agreeing with the current item: "go" if plural, "goes" if singular
    {Slot}      same as {slot} but with the first letter capitalized

Bank: a named list of templates. A fixed ~15% of each bank (chosen by hashing the
bank name and index) is reserved for evaluation documents only, so the validation and
test splits contain sentence structures the model never saw in training.
"""
import hashlib
import re

IRREGULAR = {"be": "is", "have": "has", "do": "does", "go": "goes"}


def conjugate(verb, plural):
    if plural:
        return {"be": "are"}.get(verb, verb)
    if verb in IRREGULAR:
        return IRREGULAR[verb]
    if re.search(r"(s|sh|ch|x|z|o)$", verb):
        return verb + "es"
    if re.search(r"[^aeiou]y$", verb):
        return verb[:-1] + "ies"
    return verb + "s"


def cap(s):
    return s[:1].upper() + s[1:] if s else s


def _choose(text, rng):
    """Resolve [a|b] alternations, innermost first."""
    pattern = re.compile(r"\[([^\[\]]*)\]")
    while True:
        m = pattern.search(text)
        if not m:
            return text
        options = m.group(1).split("|")
        text = text[: m.start()] + rng.choice(options) + text[m.end():]


def expand(template, rng, ctx):
    text = _choose(template, rng)

    def repl(m):
        key = m.group(1)
        if key[:2] in ("v:", "V:"):
            verb = conjugate(key[2:], ctx.get("plural", True))
            return cap(verb) if key[0] == "V" else verb
        lower = key[:1].lower() + key[1:]
        val = ctx.get(key, ctx.get(lower))
        if val is None:
            raise KeyError(f"missing slot {{{key}}} in template: {template!r}")
        if callable(val):
            val = val(rng)
        # a slot value may itself contain template syntax
        if "[" in val or "{" in val:
            val = expand(val, rng, ctx)
        return cap(val) if key[:1].isupper() and lower != key else val

    text = re.sub(r"\{([A-Za-z_]+(?::[a-z]+)?)\}", repl, text)
    return tidy(text)


def tidy(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\.\.(?!\.)", ".", text)
    return text.strip()


def _is_eval_only(name, idx, n):
    if n < 5:
        return False
    h = int(hashlib.md5(f"{name}:{idx}".encode()).hexdigest(), 16)
    return h % 7 == 0


class Bank:
    def __init__(self, name, templates):
        self.name = name
        self.templates = list(templates)
        n = len(self.templates)
        self.eval_only = [t for i, t in enumerate(self.templates) if _is_eval_only(name, i, n)]
        self.train = [t for i, t in enumerate(self.templates) if not _is_eval_only(name, i, n)]

    def pick(self, rng, mode="train"):
        if mode == "eval" and self.eval_only and rng.random() < 0.6:
            return rng.choice(self.eval_only)
        return rng.choice(self.train)

    def render(self, rng, ctx, mode="train"):
        return expand(self.pick(rng, mode), rng, ctx)
