"""Evaluate pretrained checkpoints (spec section 6).

    python -m src.evaluate_pretrain [--ckpt-dir checkpoints/pretrain] [--official ckpt_best.pt]

For every checkpoint found (snapshots/*.pt and ckpt_best.pt):
  - validation and test loss / perplexity over the FULL split (non-overlapping 256-token windows)
  - item probe: does greedy completion name the right disposal stream? 3 prompt frames x 147 core items,
    and the same for the 20 held-out items (never seen in training)
  - fact cloze probe: 25 familiar statements from the knowledge base, cut before the key word
For the official checkpoint, also generates example texts from familiar, novel, held-out and off-domain prompts.

Outputs
  results/tables/pretrain_eval.{md,csv}          one row per checkpoint
  results/tables/pretrain_item_probe.csv         every probe prompt, completion and verdict
  results/tables/pretrain_fact_cloze.csv
  results/samples/pretrain_generation_examples.md
  results/figures/pretrain_checkpoints.png       probe accuracy and loss across checkpoints
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.kb import ROOT, load_kb
from src.model import load_checkpoint
from src.probes import FACT_CLOZES, classify_destination, greedy, item_frames
from src.tokenizer import BPETokenizer

TABLES, SAMPLES, FIGURES = ROOT / "results" / "tables", ROOT / "results" / "samples", ROOT / "results" / "figures"

EXAMPLE_PROMPTS = {
    "Familiar statement (seen pattern)": ["Pizza boxes", "Leftover paint", "Glass bottles and jars go in the",
                                          "The green organics cart is collected every", "Bulky items"],
    "Novel framing (domain facts in a new context)": [
        "My neighbour asked what to do with her old couch, so I told her",
        "Here is a quick tip about batteries:",
        "Dear resident, thank you for your question about cooking grease.",
        "Question: Can I put plastic bags in the blue cart?\nAnswer:"],
    "Held-out items (never in training)": ["Plastic hummus tubs", "Q: Where do wireless routers go?\nA:",
                                           "Bubble wrap", "Nail polish remover"],
    "Off-domain": ["The capital of France is", "Once upon a time,"],
}


@torch.no_grad()
def full_split_loss(model, path, T, device, batch=16):
    data = np.fromfile(path, dtype=np.uint16).astype(np.int64)
    n = (len(data) - 1) // T
    total, count = 0.0, 0
    model.eval()
    for i in range(0, n, batch):
        idx = range(i, min(n, i + batch))
        x = torch.tensor(np.stack([data[j * T:(j + 1) * T] for j in idx]), device=device)
        y = torch.tensor(np.stack([data[j * T + 1:(j + 1) * T + 1] for j in idx]), device=device)
        _, loss = model(x, y)
        total += loss.item() * x.numel()
        count += x.numel()
    return total / count


def item_probe(model, tok, items, device):
    rows = []
    for it in items:
        for frame, prompt in item_frames(it["name"]).items():
            out = greedy(model, tok, prompt, max_new_tokens=30, device=device)
            pred = classify_destination(out)
            rows.append({"item": it["id"], "name": it["name"], "split": it["split"], "stream": it["stream"],
                         "frame": frame, "prompt": prompt, "completion": out, "predicted": pred or "none",
                         "correct": int(pred == it["stream"])})
    return rows


def fact_cloze(model, tok, device):
    rows = []
    for prompt, answers, fid in FACT_CLOZES:
        out = greedy(model, tok, prompt, max_new_tokens=12, device=device)
        ok = any(a.lower() in out.lower()[: len(a) + 25] for a in answers)
        rows.append({"fact": fid, "prompt": prompt, "expected": "/".join(answers), "completion": out, "correct": int(ok)})
    return rows


def acc(rows, **filt):
    sel = [r for r in rows if all(r[k] == v for k, v in filt.items())]
    return sum(r["correct"] for r in sel) / max(1, len(sel)), len(sel)


def find_checkpoints(ckpt_dir):
    found = {}
    for p in sorted((ckpt_dir / "snapshots").glob("*.pt")):
        found[p] = None
    if (ckpt_dir / "ckpt_best.pt").exists():
        found[ckpt_dir / "ckpt_best.pt"] = None
    return list(found)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "pretrain"))
    ap.add_argument("--official", default="ckpt_best.pt", help="checkpoint used for generation examples")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    for d in (TABLES, SAMPLES, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    kb = load_kb()
    tok = BPETokenizer.load(ROOT / "data" / "tokenizer" / "tokenizer.json")
    core = [i for i in kb["items"] if i["split"] == "core"]
    held = [i for i in kb["items"] if i["split"] == "heldout"]
    majority = Counter(i["stream"] for i in core).most_common(1)[0]

    summary, all_items, all_facts = [], [], []
    ckpts = find_checkpoints(Path(a.ckpt_dir))
    for path in ckpts:
        model, cfg, ck = load_checkpoint(path, a.device)
        label = f"{'best' if path.name == 'ckpt_best.pt' else 'snapshot'} step {ck['step']}"
        print(f"== {path.name} ({label})")
        val = full_split_loss(model, ROOT / "data" / "tokens" / "val.bin", cfg.context_length, a.device)
        test = full_split_loss(model, ROOT / "data" / "tokens" / "test.bin", cfg.context_length, a.device)
        items = item_probe(model, tok, core + held, a.device)
        facts = fact_cloze(model, tok, a.device)
        for r in items + facts:
            r["checkpoint"] = label
            r["step"] = ck["step"]
        all_items += items
        all_facts += facts
        row = {"checkpoint": label, "step": ck["step"], "val_loss": round(val, 4), "val_ppl": round(math.exp(val), 3),
               "test_loss": round(test, 4), "test_ppl": round(math.exp(test), 3),
               "core_item_acc": round(acc(items, split="core")[0], 3),
               "core_statement": round(acc(items, split="core", frame="statement")[0], 3),
               "core_directory": round(acc(items, split="core", frame="directory")[0], 3),
               "core_faq": round(acc(items, split="core", frame="faq")[0], 3),
               "heldout_item_acc": round(acc(items, split="heldout")[0], 3),
               "fact_cloze_acc": round(acc(facts)[0], 3)}
        summary.append(row)
        print("  " + json.dumps(row))
    summary.sort(key=lambda r: (r["step"], r["checkpoint"]))

    # ---- tables
    keys = list(summary[0])
    with open(TABLES / "pretrain_eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    md = ["# Pretrained model evaluation", "",
          f"Item probe: greedy completion of 3 prompt frames per item, classified by the destination named first. "
          f"Chance with 8 streams = 12.5%; always answering the most common stream ({majority[0]}) = "
          f"{majority[1] / len(core):.1%} on core items. Fact cloze: {len(FACT_CLOZES)} knowledge-base statements cut before the key word.", "",
          "| checkpoint | val loss | val ppl | test loss | test ppl | core items | statement | directory | FAQ | held-out items | fact cloze |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        md.append(f"| {r['checkpoint']} | {r['val_loss']} | {r['val_ppl']} | {r['test_loss']} | {r['test_ppl']} | "
                  f"{r['core_item_acc']:.1%} | {r['core_statement']:.1%} | {r['core_directory']:.1%} | {r['core_faq']:.1%} | "
                  f"{r['heldout_item_acc']:.1%} | {r['fact_cloze_acc']:.1%} |")
    (TABLES / "pretrain_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for name, rows in (("pretrain_item_probe.csv", all_items), ("pretrain_fact_cloze.csv", all_facts)):
        with open(TABLES / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # ---- figure across checkpoints
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    snaps = [r for r in summary if r["checkpoint"].startswith("snapshot")] or summary
    best = [r for r in summary if r["checkpoint"].startswith("best")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    for key, lab, c in (("core_item_acc", "core items", "#1f6feb"), ("heldout_item_acc", "held-out items", "#bf8700"),
                        ("fact_cloze_acc", "fact cloze", "#2da44e")):
        ax[0].plot([r["step"] for r in snaps], [r[key] for r in snaps], "o-", color=c, label=lab)
        for b in best:
            ax[0].scatter([b["step"]], [b[key]], marker="*", s=150, color=c, edgecolor="k", zorder=5)
    ax[0].axhline(majority[1] / len(core), ls="--", color="grey", lw=1, label=f"always '{majority[0]}'")
    ax[0].set_ylim(0, 1.02); ax[0].set_xlabel("training step"); ax[0].set_ylabel("accuracy")
    ax[0].set_title("Knowledge probes (star = best-val checkpoint)"); ax[0].legend(fontsize=7, frameon=False); ax[0].grid(alpha=.3)
    ax[1].plot([r["step"] for r in snaps], [r["val_loss"] for r in snaps], "o-", color="#d1242f", label="validation")
    ax[1].plot([r["step"] for r in snaps], [r["test_loss"] for r in snaps], "s--", color="#8250df", label="test")
    for b in best:
        ax[1].scatter([b["step"]], [b["test_loss"]], marker="*", s=150, color="#8250df", edgecolor="k", zorder=5)
    ax[1].set_xlabel("training step"); ax[1].set_ylabel("loss (full split)"); ax[1].set_title("Held-out loss")
    ax[1].legend(fontsize=7, frameon=False); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "pretrain_checkpoints.png")

    # ---- generation examples from the official checkpoint
    model, cfg, ck = load_checkpoint(Path(a.ckpt_dir) / a.official, a.device)
    out = [f"# Pretrained model: generation examples", "",
           f"Checkpoint `{a.official}` (step {ck['step']}). Greedy = temperature 0; sampled = temperature 0.8, top-k 50, seed 0. "
           "Prompt text is in bold; everything after it is generated.", ""]
    g = torch.Generator().manual_seed(0)
    for group, prompts in EXAMPLE_PROMPTS.items():
        out += [f"## {group}", ""]
        for p in prompts:
            ids = torch.tensor([tok.encode(p)])
            for temp in (0.0, 0.8):
                res = model.generate(ids, 70, temperature=temp, top_k=50 if temp else None, stop_ids=(tok.eot_id,), generator=g)
                cont = tok.decode(res[0, ids.shape[1]:].tolist(), skip_special=True)
                out += [f"**{'greedy' if temp == 0 else 'sampled T=0.8'}:**", "", "> **" + p.replace("\n", " ⏎ ") + "**"
                        + cont.replace("\n", " ⏎ "), ""]
    (SAMPLES / "pretrain_generation_examples.md").write_text("\n".join(out), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
