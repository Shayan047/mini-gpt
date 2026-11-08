"""Document generators for the pretraining corpus.

Each generator returns a dict: {"type", "text", "items": [ids], "facts": [ids], "rules": [ids]}.
All factual content comes from the knowledge base; wording comes from lexicon.py / questions.py.
`mode` is "train" for training documents and "eval" for validation/test documents, which
then also draw on the eval-only template portions (unseen phrasing).
"""
import random

from . import lexicon as L
from . import questions as Q
from .expand import Bank, cap, expand, tidy


def is_plural(form):
    if " and " in form:
        return True
    last = form.split()[-1].lower()
    return last.endswith("s") and not last.endswith("ss")


class Cycler:
    """Yields elements in shuffled rounds so every element appears equally often."""

    def __init__(self, elems, rng):
        self.elems, self.rng, self.queue = list(elems), rng, []

    def next(self, exclude=()):
        for _ in range(3):
            if not self.queue:
                self.queue = self.elems[:]
                self.rng.shuffle(self.queue)
            e = self.queue.pop()
            if e not in exclude:
                return e
        return e

    def take(self, k):
        out = []
        while len(out) < min(k, len(self.elems)):
            e = self.next(exclude=out)
            if e not in out:
                out.append(e)
        return out


class DocGen:
    def __init__(self, kb, rng, mode="train"):
        self.kb, self.rng, self.mode = kb, rng, mode
        self.items = [it for it in kb["items"] if it["split"] == "core"]
        self.item_by_id = {it["id"]: it for it in kb["items"]}
        self.rules = {r["id"]: r for r in kb["material_rules"]}
        self.facts = {f["id"]: f for t in kb["topics"].values() for f in t["facts"]}
        self.topic_of = {f["id"]: k for k, t in kb["topics"].items() for f in t["facts"]}
        self.banks = {}
        self.item_cycle = Cycler(self.items, rng)
        self.stream_cycle = {s: Cycler([i for i in self.items if i["stream"] == s], rng) for s in kb["streams"]}
        self.fact_cycle = Cycler(list(self.facts), rng)
        self.topic_cycle = Cycler(list(kb["topics"]), rng)
        self.rule_cycle = Cycler(list(self.rules), rng)
        self.weighted_stream_cycle = Cycler(
            [s for s in kb["streams"] for _ in range(max(1, len(self.stream_cycle[s].elems) // 4))], rng)

    # ---------------------------------------------------------------- helpers
    def bank(self, name, templates):
        if name not in self.banks:
            self.banks[name] = Bank(name, templates)
        return self.banks[name]

    def r(self, name, templates, ctx=None):
        return self.bank(name, templates).render(self.rng, ctx or {}, self.mode)

    def pick(self, options):
        return self.rng.choice(options)

    def ictx(self, item, form=None):
        if form is None:
            form = item["name"] if self.rng.random() < 0.6 else self.rng.choice(item["aliases"] or [item["name"]])
        pl = is_plural(form)
        return {"item": form, "plural": pl, "pron": "they" if pl else "it", "obj": "them" if pl else "it",
                "poss": "their" if pl else "its", "dem": "these" if pl else "this"}

    def dest(self, item, ctx):
        return self.r(f"dest:{item['stream']}", L.DEST[item["stream"]], ctx)

    def statement(self, item, ctx):
        c = dict(ctx, dest=self.dest(item, ctx))
        return self.r("item_statement", L.ITEM_STATEMENT, c)

    def preps(self, item, ctx, limit=None):
        codes = item["prep"][:limit] if limit else item["prep"]
        return [self.r(f"prep:{c}", L.PREP[c], ctx) for c in codes]

    def note(self, item):
        if not item["note"]:
            return ""
        lead = self.pick(L.NOTE_LEADS)
        text = item["note"]
        if lead.endswith(", ") and text[1:2].islower():
            text = text[:1].lower() + text[1:]
        return lead + text

    def related_fact(self, item):
        return self.pick(L.RELATED_FACTS[item["stream"]])

    def rule_text(self, rid):
        return self.r(f"rule:{rid}", [self.rules[rid]["text"]] + L.RULES[rid])

    def reason(self, item, ctx):
        if item["rule"] and self.rng.random() < 0.6:
            return self.rule_text(item["rule"])
        reason = self.r(f"reason:{item['stream']}", L.STREAM_REASON[item["stream"]], ctx)
        return self.r("reason_frame", L.REASON_FRAME, {"reason": reason, "plural": ctx["plural"]})

    def fact_text(self, fid):
        return self.r(f"fact:{fid}", [self.facts[fid]["text"]] + L.FACTS[fid])

    def guidance(self, item, ctx=None, p_note=0.75, p_reason=0.4, direct=False):
        """Full answer for an item: destination + prep + optional note/reason.
        direct=True uses answer-style frames (for replies to a question naming the item)."""
        ctx = ctx or self.ictx(item)
        if direct:
            first = self.r("answer_statement", L.ANSWER_STATEMENT, dict(ctx, dest=self.dest(item, ctx)))
        else:
            first = self.statement(item, ctx)
        parts = [first] + self.preps(item, ctx)
        if self.rng.random() < p_note:
            parts.append(self.note(item))
        if self.rng.random() < p_reason:
            parts.append(self.reason(item, ctx))
        return " ".join(p for p in parts if p)

    def item_question(self, item, ctx):
        roll = self.rng.random()
        if roll < 0.2 and item["stream"] in Q.ASKED_BIN or (roll < 0.1):
            asked_stream = item["stream"] if (item["stream"] in Q.ASKED_BIN and self.rng.random() < 0.5) \
                else self.pick([s for s in Q.ASKED_BIN if s != item["stream"]])
            c = dict(ctx, asked=self.pick(Q.ASKED_BIN[asked_stream]))
            q = self.r("q_item_yesno", Q.ITEM_YESNO_QUESTIONS, c)
            lead = self.pick(Q.YES_LEADS if asked_stream == item["stream"] else Q.NO_LEADS)
            return q, lead
        if roll < 0.3:
            return self.r("q_item_prep", Q.ITEM_PREP_QUESTIONS, ctx), ""
        return self.r("q_item", Q.ITEM_QUESTIONS, ctx), ""

    def answer(self, item, ctx, lead=""):
        body = self.guidance(item, ctx, p_note=0.6, p_reason=0.35, direct=True)
        if lead.endswith(",") and body[1:2].islower() and not body.startswith(("I ", "Tidewell")):
            body = body[:1].lower() + body[1:]
        return (lead + " " + body).strip()

    def fact_question(self, fid):
        return self.r(f"q_fact:{fid}", Q.FACT_QUESTIONS[fid])

    def doc(self, dtype, lines, items=(), facts=(), rules=()):
        text = "\n".join(lines)
        text = "\n".join(tidy(l) if l.strip() else "" for l in text.split("\n"))
        return {"type": dtype, "text": text.strip(), "items": [i["id"] for i in items],
                "facts": list(facts), "rules": list(rules)}

    def org(self):
        return self.pick(L.ORG_NAMES)

    def org_cap(self):
        return cap(self.org())

    # ---------------------------------------------------------------- document types
    def guide_page(self):
        s = self.weighted_stream_cycle.next()
        st = self.kb["streams"][s]
        items = self.stream_cycle[s].take(self.rng.randint(4, 8))
        lines = [self.pick(L.STREAM_TITLE[s]), "", self.pick(L.STREAM_INTRO[s]), ""]
        lines.append(self.pick(["General rules:", "The basics:", "Before you start:", "Key rules:"]))
        rules = st["rules"][:]
        self.rng.shuffle(rules)
        lines += [f"- {r}" for r in rules]
        lines += ["", self.pick(["Common items:", "Examples:", "Items residents ask about:", "Item by item:"])]
        for it in items:
            ctx = self.ictx(it)
            if self.rng.random() < 0.5:
                lines.append(f"- {cap(ctx['item'])}: " + " ".join(self.preps(it, ctx) or [cap(self.pick(L.BIN_NAME[s])) + "."]))
            else:
                lines.append("- " + self.guidance(it, ctx, p_reason=0.2))
        rids = [r for r in self.rules if self.rules[r]["stream"] == s]
        if rids:
            rid = self.pick(rids)
            lines += ["", self.pick(["Rule of thumb: ", "Remember: ", "In short: "]) + self.rule_text(rid)]
        else:
            rid = None
        fid = self.fact_cycle.next()
        lines += ["", self.fact_text(fid)]
        return self.doc("guide_page", lines, items, [fid], [rid] if rid else [])

    def az_entries(self):
        items = self.item_cycle.take(self.rng.randint(5, 9))
        entries = []
        style = self.rng.randint(0, 2)
        for it in items:
            ctx = self.ictx(it, it["name"])
            bin_name = self.pick(L.BIN_NAME[it["stream"]])
            prep = " ".join(self.preps(it, ctx))
            note = it["note"] if self.rng.random() < 0.8 else ""
            if style == 0:
                e = [cap(it["name"]), f"Where: {bin_name}"]
                if prep:
                    e.append(f"How: {prep}")
                if note:
                    e.append(f"Note: {note}")
            elif style == 1:
                e = [f"{cap(it['name'])} - {bin_name}. {prep} {note}".strip()]
            else:
                e = [f"{cap(it['name'])}: {self.guidance(it, ctx, p_note=0.8, p_reason=0.0)}"]
            if it["aliases"] and self.rng.random() < 0.5:
                e.append(self.pick(["Also called: ", "See also: ", "Includes: "]) + ", ".join(it["aliases"][:3]))
            entries.append((it["name"].lower(), "\n".join(e)))
        entries.sort()
        title = self.pick(["A-Z Sorting Guide", "Waste Guide: A to Z", "Where Does It Go? A-Z Listing",
                           "Sorting Directory", "A to Z of Household Items"])
        lines = [title + self.pick(["", " (excerpt)", " - selected items"]), ""]
        for _, e in entries:
            lines += [e, ""]
        return self.doc("az_entries", lines, items)

    def faq(self):
        n_items, n_facts = self.rng.randint(2, 5), self.rng.randint(1, 3)
        items = self.item_cycle.take(n_items)
        fids = [self.fact_cycle.next() for _ in range(n_facts)]
        qa = []
        for it in items:
            ctx = self.ictx(it)
            q, lead = self.item_question(it, ctx)
            qa.append((q, self.answer(it, ctx, lead)))
        for fid in fids:
            qa.append((self.fact_question(fid), self.fact_text(fid)))
        self.rng.shuffle(qa)
        qfmt, afmt = self.pick([("Q: ", "A: "), ("Question: ", "Answer: "), ("", ""), ("- ", "  ")])
        lines = [self.pick(["Frequently Asked Questions", "Residents' FAQ", "Your Questions Answered",
                            "Sorting FAQ", "Common Questions About Waste in Tidewell"]), ""]
        for q, a in qa:
            lines += [qfmt + q, afmt + a, ""]
        return self.doc("faq", lines, items, fids)

    def newsletter(self):
        season = self.pick(L.SEASONS)
        spot = self.item_cycle.next()
        quick = self.item_cycle.take(3)
        fids = [self.fact_cycle.next() for _ in range(2)]
        if season in ("winter", "fall", "spring") and self.rng.random() < 0.5:
            fids.append(self.pick({"winter": ["N01", "N02", "P04"], "fall": ["N03", "W02"], "spring": ["N03", "U02"]}[season]))
        ctx = self.ictx(spot)
        lines = [self.pick([f"Waste Wise Newsletter - {cap(season)} Edition", f"Tidewell {cap(season)} Waste Update",
                            f"The Sorting Times: {cap(season)} Issue", f"{cap(season)} Newsletter from {self.org_cap()}"]), ""]
        lines += [self.pick(["Did you know?", "News and reminders", "This season"]), " ".join(self.fact_text(f) for f in fids), ""]
        lines += [self.pick(["Item spotlight: ", "Featured item: ", "In the spotlight: "]) + ctx["item"],
                  self.guidance(spot, ctx, p_note=0.9, p_reason=0.7), ""]
        lines.append(self.pick(["Quick sorting check:", "Test yourself:", "Sorting at a glance:"]))
        for it in quick:
            lines.append(f"- {cap(it['name'])}: {self.pick(L.BIN_NAME[it['stream']])}")
        lines += ["", self.pick(["Thanks for sorting right!", "See you next season.", f"From all of us at {self.org()}, thank you.",
                                 "Questions? Contact the waste service through the website, the app or customer service."])]
        return self.doc("newsletter", lines, [spot] + quick, fids)

    def staff_memo(self):
        items = self.item_cycle.take(self.rng.randint(2, 4))
        fids = [self.fact_cycle.next() for _ in range(self.rng.randint(1, 2))]
        names = ", ".join(i["name"] for i in items)
        lines = ["INTERNAL MEMO", f"To: {cap(self.pick(L.STAFF_ROLES))}", f"From: {self.pick(L.SENDER_ROLES)}",
                 f"Subject: {self.pick(['Resident questions about ', 'Reminder: sorting guidance for ', 'Common mix-ups: '])}{names}", ""]
        lines.append(self.pick(["Hi team,", "Hello everyone,", "Team,"]))
        lines.append(self.pick([f"We have had a lot of calls recently about {names}.",
                                f"Crews keep reporting confusion about {names}.",
                                f"Please review the guidance below on {names} before your next shift."]))
        lines.append(self.pick(["Please give residents the following guidance:", "Here is what to tell residents:",
                                "The correct guidance is:"]))
        for it in items:
            lines.append("- " + self.guidance(it, p_note=0.8, p_reason=0.5))
        lines += ["", self.pick(["Also remember: ", "One more reminder: ", "Related: "]) + " ".join(self.fact_text(f) for f in fids)]
        lines += ["", self.pick(["Thanks,", "Thank you,", "Regards,"]), self.pick(L.SENDER_ROLES)]
        return self.doc("staff_memo", lines, items, fids)

    def training_note(self):
        s = self.weighted_stream_cycle.next()
        items = self.stream_cycle[s].take(self.rng.randint(3, 6))
        rid = self.pick([r for r in self.rules if self.rules[r]["stream"] == s] or [None])
        lines = [f"Staff training note: {self.pick(L.STREAM_TITLE[s])}",
                 f"For: {self.pick(L.STAFF_ROLES)}", ""]
        lines.append(self.pick(L.STREAM_INTRO[s]))
        if rid:
            lines.append(self.pick(["The key rule to explain: ", "Teach residents this rule: ", "Rule to remember: "]) + self.rule_text(rid))
        lines += ["", self.pick(["Example calls:", "Typical questions and how to answer:", "Practice scenarios:"])]
        for it in items:
            ctx = self.ictx(it)
            lines.append(self.pick([f"If a resident asks about {ctx['item']}, explain: ",
                                    f"Caller has {ctx['item']}: ", f"Scenario - {ctx['item']}: "])
                         + self.guidance(it, ctx, p_note=0.8, p_reason=0.5))
        return self.doc("training_note", lines, items, [], [rid] if rid else [])

    def cs_email(self):
        it = self.item_cycle.next()
        ctx = self.ictx(it)
        q, lead = self.item_question(it, ctx)
        extra_fid = self.related_fact(it) if self.rng.random() < 0.4 else None
        lines = [f"Subject: {self.pick(['Question about ', 'How to dispose of ', 'Where do I put ', 'Help with '])}{ctx['item']}", ""]
        lines += [self.pick(Q.GREETINGS), " ".join(x for x in [self.pick(Q.RESIDENT_CONTEXTS) if self.rng.random() < 0.6 else "", q] if x),
                  self.pick(["Thanks,", "Thank you!", "Thanks in advance.", "Cheers,"]), "A Tidewell resident", ""]
        lines += [self.pick(["Reply:", "Response from Tidewell Waste Services:", "---", "Re: your question"])]
        body = [self.pick(Q.AGENT_GREETINGS), self.answer(it, ctx, lead)]
        if extra_fid:
            body.append(self.pick(["Also, ", "While we're at it: ", "Good to know: "]) + self.fact_text(extra_fid))
        body.append(self.pick(Q.AGENT_CLOSINGS))
        lines += [" ".join(body), self.pick(["Tidewell Waste Services", "Customer Service, Tidewell Waste & Recycling Services",
                                             "The Residential Waste Guidance team"])]
        return self.doc("cs_email", lines, [it], [extra_fid] if extra_fid else [])

    def chat(self):
        n = self.rng.randint(1, 3)
        items = self.item_cycle.take(n)
        fids = [self.related_fact(items[-1]) if self.rng.random() < 0.5 else self.fact_cycle.next()]             if self.rng.random() < 0.5 else []
        user, agent = self.pick([("Resident", "Agent"), ("Customer", "Support"), ("User", "Assistant"), ("Caller", "Rep")])
        lines = [self.pick(["Chat transcript", "Live chat log", "Support chat", "Help line call summary"]), ""]
        lines.append(f"{agent}: " + self.pick(["Hello! How can I help you today?", "Hi, welcome to Tidewell Waste Services chat.",
                                               "Thanks for calling. What can I help with?"]))
        for i, it in enumerate(items):
            ctx = self.ictx(it)
            q, lead = self.item_question(it, ctx)
            prefix = "" if i == 0 else self.pick(["Also, ", "One more: ", "And ", "Another one. "])
            if prefix == "And ":
                q = q[:1].lower() + q[1:]
            lines.append(f"{user}: {prefix}{q}")
            lines.append(f"{agent}: " + self.answer(it, ctx, lead))
        for fid in fids:
            lines.append(f"{user}: " + self.fact_question(fid))
            lines.append(f"{agent}: " + self.fact_text(fid))
        lines.append(f"{user}: " + self.pick(["Thanks!", "Great, thank you.", "That helps, thanks.", "Perfect."]))
        lines.append(f"{agent}: " + self.pick(Q.AGENT_CLOSINGS))
        return self.doc("chat", lines, items, fids)

    def social_post(self):
        n = self.rng.randint(1, 3)
        items = self.item_cycle.take(n)
        lines = []
        for it in items:
            ctx = self.ictx(it)
            head = self.pick(["Sorting tip: ", "Tip of the day: ", "Did you know? ", "Quick reminder: ", "", "Sorting Saturday: "])
            body = self.statement(it, ctx) + " " + " ".join(self.preps(it, ctx, limit=2))
            tag = self.pick(["#TidewellSorts", "#WasteWise", "#SortRight #Tidewell", "#RecycleRight", "", "#TidewellWaste"])
            lines += [f"{self.org_cap()} posted:", f"{head}{body} {tag}".strip(), ""]
        if self.rng.random() < 0.4:
            fid = self.fact_cycle.next()
            lines += [f"{self.org_cap()} posted:", self.fact_text(fid) + " " + self.pick(["#TidewellSorts", "#WasteWise", ""])]
            return self.doc("social_post", lines, items, [fid])
        return self.doc("social_post", lines, items)

    def topic_page(self):
        tk = self.topic_cycle.next()
        topic = self.kb["topics"][tk]
        fids = [f["id"] for f in topic["facts"]]
        k = self.rng.randint(max(2, len(fids) - 2), len(fids))
        chosen = self.rng.sample(fids, k)
        chosen.sort()
        title = self.pick([topic["title"], f"{topic['title']} - {self.pick(L.GUIDE_NAMES)}", f"Guide: {topic['title']}",
                           f"{topic['title']}: what residents need to know"])
        style = self.rng.randint(0, 2)
        lines = [cap(title), ""]
        if style == 0:
            lines += [f"- {self.fact_text(f)}" for f in chosen]
        elif style == 1:
            lines.append(" ".join(self.fact_text(f) for f in chosen))
        else:
            for f in chosen:
                lines += [self.fact_question(f), self.fact_text(f), ""]
        return self.doc("topic_page", lines, [], chosen)

    def notice(self):
        tk = self.pick(["collection_schedule", "cart_placement", "missed_collection", "bulky_items", "contamination_fines",
                        "drop_off", "seasonal", "organics_tips"])
        topic = self.kb["topics"][tk]
        chosen = self.rng.sample([f["id"] for f in topic["facts"]], min(3, len(topic["facts"])))
        lines = [self.pick(["NOTICE TO RESIDENTS", "PUBLIC NOTICE", "IMPORTANT REMINDER", "ATTENTION RESIDENTS",
                            "COMMUNITY NOTICE"]), self.pick(["", f"From {self.org()}", f"Issued by {self.org()}"]),
                 f"Re: {topic['title']}", ""]
        lines += [self.fact_text(f) for f in chosen]
        lines += ["", self.pick(["Thank you for your cooperation.", "Thank you for helping keep Tidewell clean.",
                                 "For more information, visit the waste service's website or app.", ""])]
        return self.doc("notice", lines, [], chosen)

    def forum_thread(self):
        it = self.item_cycle.next()
        ctx = self.ictx(it)
        q, lead = self.item_question(it, ctx)
        asker, r1, r2 = self.rng.sample(L.HANDLES, 3)
        lines = [f"Tidewell community forum > {self.pick(['Recycling & waste', 'Around town', 'Home and garden', 'Ask your neighbours'])}",
                 f"Thread: {q}", "", f"{asker} wrote:", " ".join(x for x in [self.pick(Q.RESIDENT_CONTEXTS), q] if x), ""]
        lines += [f"{r1} wrote:", (self.pick(["I checked the waste guide. ", "From the Tidewell guide: ", "I asked the waste service about this. ", ""])
                                   + self.answer(it, ctx, lead))]
        if self.rng.random() < 0.6:
            lines += ["", f"{r2} wrote:", self.pick(["Can confirm. ", "Same answer I got. ", "Yes, that's right. ", "+1. "]) + self.reason(it, ctx)]
        lines += ["", f"{asker} wrote:", self.pick(["Thanks everyone!", "Got it, thanks.", "Great, that clears it up.", "Thank you!"])]
        return self.doc("forum_thread", lines, [it], [], [it["rule"]] if it["rule"] else [])

    def school_handout(self):
        items = self.item_cycle.take(self.rng.randint(4, 7))
        lines = [self.pick(["Which Bin Does It Go In?", "Sorting Superstars", "Kids' Guide to Sorting Waste",
                            "Classroom Activity: Sort It!", "Be a Waste Hero"]), ""]
        lines.append(self.pick(["In Tidewell we have three carts at home: blue for recycling, green for organics and black for garbage.",
                                "Blue cart for recycling, green cart for food and yard waste, black cart for garbage!",
                                "Every home in Tidewell sorts waste into a blue cart, a green cart and a black cart."]))
        lines.append("")
        bullet = self.pick(["- ", "* ", ""])
        for it in items:
            ctx = self.ictx(it, it["name"])
            lines.append(bullet + self.statement(it, ctx) +
                         (" " + self.preps(it, ctx, limit=1)[0] if it["prep"] and self.rng.random() < 0.5 else ""))
        lines += ["", self.pick(["Ask a grown-up if you are not sure!", "Great job, waste heroes!",
                                 "Remember: batteries and electronics never go in any cart.", "Clean, empty and dry is the rule for recycling!"])]
        return self.doc("school_handout", lines, items)

    def building_notice(self):
        fids = self.rng.sample(["A01", "A02", "A03", "A04", "B03", "F05"], self.rng.randint(2, 4))
        items = self.item_cycle.take(self.rng.randint(1, 3))
        lines = [self.pick(["Notice from Property Management", "To All Residents of This Building", "Building Waste Room Rules",
                            "Message from Your Building Management", "Condo Board Reminder"]), ""]
        lines += [self.fact_text(f) for f in fids]
        lines += ["", self.pick(["Common questions in our building:", "Please also remember:", "Sorting reminders:"])]
        for it in items:
            lines.append("- " + self.guidance(it, p_note=0.5, p_reason=0.2))
        lines += ["", self.pick(["Thank you,", "Thanks for your cooperation,", "Regards,"]),
                  self.pick(["Property Management", "Building Management", "The Superintendent", "Condo Board"])]
        return self.doc("building_notice", lines, items, fids)

    def contamination_report(self):
        wrong_cart = self.pick(["recycling", "organics"])
        pool = [i for i in self.items if i["stream"] != wrong_cart]
        items, n = [], self.rng.randint(3, 5)
        while len(items) < n:
            it = self.item_cycle.next()
            if it in pool and it not in items:
                items.append(it)
        cart = {"recycling": "blue recycling carts", "organics": "green organics carts"}[wrong_cart]
        lines = [self.pick(["Route inspection summary", "Cart audit report", "Collection crew contamination report",
                            "Weekly contamination summary"]), self.pick(["Prepared for: Residential Waste Guidance", "Internal - collection operations", ""]), ""]
        lines.append(f"Items found in {cart} that do not belong there:")
        for it in items:
            ctx = self.ictx(it, it["name"])
            lines.append(f"- {cap(it['name'])}: wrong cart. " + self.statement(it, ctx))
        lines += ["", self.pick(["Action: ", "Follow-up: ", "Next steps: "]) + self.fact_text(self.pick(["F02", "F03", "F01"]))]
        lines.append(self.pick(["Outreach will include these items in the next newsletter.", "Crews tagged affected carts.", ""]))
        return self.doc("contamination_report", lines, items, [])

    def common_mistakes(self):
        items = self.item_cycle.take(self.rng.randint(3, 5))
        lines = [self.pick(["Common Sorting Mistakes", "Sorting Mix-Ups to Avoid", "Get It Right: Frequent Errors",
                            "Don't Get Tagged: Common Mistakes"]), ""]
        for it in items:
            wrong = self.pick([s for s in L.WRONG_PLACE if s != it["stream"]])
            ctx = self.ictx(it)
            lines.append(self.pick([f"Mistake: putting {ctx['item']} in {L.WRONG_PLACE[wrong]}. ",
                                    f"Not in {L.WRONG_PLACE[wrong]}: {ctx['item']}. ",
                                    f"{cap(ctx['item'])} do not belong in {L.WRONG_PLACE[wrong]}. " if ctx["plural"]
                                    else f"{cap(ctx['item'])} does not belong in {L.WRONG_PLACE[wrong]}. "])
                         + self.pick(["Instead: ", "Correct: ", "Do this: ", ""]) + self.guidance(it, ctx, p_note=0.6, p_reason=0.5))
        return self.doc("common_mistakes", lines, items)

    def rules_of_thumb(self):
        rids = self.rule_cycle.take(self.rng.randint(5, 9))
        lines = [self.pick(["Rules of Thumb for Sorting Anything", "How to Sort Items Not on the List",
                            "Sorting by Material", "Not Sure? Use These Rules", "The Material Rules"]), ""]
        lines.append(self.pick(["You don't need to memorize every item. Look at what it is made of:",
                                "If an item is not in the A-Z guide, these rules will usually tell you where it goes:",
                                "Most items can be sorted by asking what material they are made of.", ""]))
        for i, rid in enumerate(rids, 1):
            ex = [it for it in self.items if it["rule"] == rid]
            line = f"{i}. " + self.rule_text(rid)
            if ex and self.rng.random() < 0.6:
                exs = self.rng.sample(ex, min(len(ex), self.rng.randint(1, 3)))
                line += " " + self.pick(["Examples: ", "For example: ", "Think: ", "Like: "]) + ", ".join(e["name"] for e in exs) + "."
            lines.append(line)
        return self.doc("rules_of_thumb", lines, [], [], rids)


DOC_TYPES = {
    # name: weight
    "guide_page": 9, "az_entries": 10, "faq": 12, "newsletter": 5, "staff_memo": 5, "training_note": 6, "cs_email": 10,
    "chat": 10, "social_post": 5, "topic_page": 9, "notice": 4, "forum_thread": 6, "school_handout": 3,
    "building_notice": 3, "contamination_report": 2, "common_mistakes": 3, "rules_of_thumb": 5,
}
