"""Render data/kb/knowledge_base.json as readable markdown (data/kb/knowledge_base.md).

    python src/kb_to_md.py
"""
from kb import ROOT, load_kb


def main():
    kb = load_kb()
    org = kb["organization"]
    out = [f"# {org['guide_name']} - Knowledge Base", "",
           f"**Organization:** {org['name']} (fictional)  ", f"**Department:** {org['department']}  ",
           f"**Service area:** {org['service_area']}  ", f"**Purpose:** {org['purpose']}  ",
           f"**Contact:** {org['contact']}  ", f"**Out of scope:** {org['out_of_scope']}", "",
           "Facts are deliberately generic: no street addresses, phone numbers, staff names or clock times.", "",
           "## Disposal streams", ""]
    for key, s in kb["streams"].items():
        out += [f"### {s['container'].capitalize()} (`{key}`, {s['method']})", "", s["summary"], ""]
        out += [f"- {r}" for r in s["rules"]] + [""]
    out += ["## Material rules", "", "| id | stream | rule |", "|---|---|---|"]
    out += [f"| {r['id']} | {r['stream']} | {r['text']} |" for r in kb["material_rules"]]
    out += ["", "## Preparation codes", "", "| code | instruction |", "|---|---|"]
    out += [f"| `{k}` | {v} |" for k, v in kb["prep_codes"].items()]
    for split, title in [("core", "Items (used in training)"),
                         ("heldout", "Held-out items (never in corpus or fine-tuning; evaluation only)")]:
        out += ["", f"## {title}", "", "| item | also called | stream | rule | preparation | note |", "|---|---|---|---|---|---|"]
        for it in sorted((i for i in kb["items"] if i["split"] == split), key=lambda i: (i["stream"], i["name"])):
            prep = "; ".join(kb["prep_codes"][p] for p in it["prep"])
            out.append(f"| {it['name']} | {', '.join(it['aliases'])} | {it['stream']} | {it['rule'] or '-'} | {prep} | {it['note']} |")
    out += ["", "## Topics", ""]
    for t in kb["topics"].values():
        out += [f"### {t['title']}", ""] + [f"- **{f['id']}** {f['text']}" for f in t["facts"]] + [""]
    path = ROOT / "data" / "kb" / "knowledge_base.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
