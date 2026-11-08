"""Supervised fine-tuning (SFT) for question answering (spec section 7).

    python -m src.finetune --init checkpoints/pretrain/ckpt_best.pt --out-dir /content/drive/MyDrive/mini-gpt/sft

Starts from the pretrained weights and continues training on question-answer pairs formatted as
    <|user|> question<|assistant|> answer<|endoftext|>
The loss is computed only on the answer tokens (and the final <|endoftext|>), so the model learns to
produce answers, not to predict questions. Resumes automatically from <out-dir>/ckpt_last.pt.

Writes to <out-dir>: ckpt_best.pt (lowest SFT validation loss), ckpt_last.pt, log.csv, train_log.txt,
samples.jsonl, summary.json. log.csv also tracks the loss on the PRETRAINING validation corpus, to
measure how much general text modelling is lost ("forgetting") during fine-tuning.
"""
import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.model import GPT, GPTConfig
from src.tokenizer import BPETokenizer
from src.train import TokenData, configure_optimizer, estimate_loss

ROOT = Path(__file__).resolve().parent.parent


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--init", default=str(ROOT / "checkpoints" / "pretrain" / "ckpt_best.pt"))
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--out-dir", default=str(ROOT / "checkpoints" / "sft"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=1000)          # ~2.9 epochs over 11k pairs
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--corpus-eval-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args(argv)


def lr_at(step, a):
    if step < a.warmup_steps:
        return a.lr * (step + 1) / a.warmup_steps
    progress = min(1.0, (step - a.warmup_steps) / max(1, a.max_steps - a.warmup_steps))
    return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * progress))


def encode_pairs(path, tok, T):
    """Returns list of (ids, n_prompt) per example; ids = prompt + answer tokens."""
    out = []
    for line in open(path, encoding="utf-8"):
        e = json.loads(line)
        prompt = tok.encode(f"<|user|> {e['question']}<|assistant|>", allow_special=True)
        answer = tok.encode(f" {e['answer']}<|endoftext|>", allow_special=True)
        ids = (prompt + answer)[: T + 1]
        out.append((ids, len(prompt)))
    return out


def make_batch(examples, pad_id, device):
    """x = ids[:-1], y = ids[1:]; mask is 1 where y is an answer token. Right-padded with <|pad|>."""
    L = max(len(ids) for ids, _ in examples) - 1
    x = torch.full((len(examples), L), pad_id, dtype=torch.long)
    y = torch.full((len(examples), L), pad_id, dtype=torch.long)
    m = torch.zeros((len(examples), L))
    for i, (ids, n_prompt) in enumerate(examples):
        n = len(ids) - 1
        x[i, :n] = torch.tensor(ids[:-1])
        y[i, :n] = torch.tensor(ids[1:])
        m[i, n_prompt - 1:n] = 1.0          # y[j] = ids[j+1]; answer starts at ids[n_prompt]
    return x.to(device), y.to(device), m.to(device)


@torch.no_grad()
def sft_val_loss(model, data, a, pad_id, device, amp_ctx):
    model.eval()
    total, count = 0.0, 0.0
    for i in range(0, len(data), a.batch_size):
        x, y, m = make_batch(data[i:i + a.batch_size], pad_id, device)
        with amp_ctx():
            _, loss = model(x, y, loss_mask=m)
        total += loss.item() * m.sum().item()
        count += m.sum().item()
    model.train()
    return total / count


@torch.no_grad()
def sample_answers(model, tok, questions, device, step):
    out = []
    for q in questions:
        ids = torch.tensor([tok.encode(f"<|user|> {q}<|assistant|>", allow_special=True)], device=device)
        res = model.generate(ids, 80, temperature=0, stop_ids=(tok.eot_id,))
        out.append({"step": step, "question": q, "answer": tok.decode(res[0, ids.shape[1]:].tolist(), skip_special=True).strip()})
    model.train()
    return out


def main(argv=None):
    a = get_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    use_amp = device == "cuda" and not a.no_amp
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_amp else (lambda: torch.autocast("cpu", enabled=False))
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)

    tok = BPETokenizer.load(Path(a.data_dir) / "tokenizer" / "tokenizer.json")
    init = torch.load(a.init, map_location=device, weights_only=False)
    cfg = GPTConfig(**init["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(init["model"])
    train = encode_pairs(Path(a.data_dir) / "sft" / "train.jsonl", tok, cfg.context_length)
    val = encode_pairs(Path(a.data_dir) / "sft" / "val.jsonl", tok, cfg.context_length)
    corpus_val = TokenData(Path(a.data_dir) / "tokens" / "val.bin", cfg.context_length)
    corpus_offsets = corpus_val.random_offsets(a.corpus_eval_batches * a.batch_size, np.random.default_rng(0))
    sample_qs = [json.loads(l)["question"] for l in open(Path(a.data_dir) / "sft" / "val.jsonl", encoding="utf-8")][:8]

    opt = configure_optimizer(model, a)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step, best, elapsed_before, order, pos = 0, float("inf"), 0.0, list(range(len(train))), 0
    rng.shuffle(order)
    last = out / "ckpt_last.pt"
    if last.exists() and not a.no_resume:
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); scaler.load_state_dict(ck["scaler"])
        step, best, elapsed_before, order, pos = ck["step"], ck["best"], ck["elapsed"], ck["order"], ck["pos"]
        rng.setstate(ck["rng"])
        print(f"resumed from step {step}")

    header = (f"device={device} amp={use_amp} init={a.init} (pretrain step {init.get('step')}) sft_train={len(train):,} "
              f"sft_val={len(val):,} steps={a.max_steps} (~{a.max_steps * a.batch_size / len(train):.1f} epochs)")
    print(header)
    log_txt = open(out / "train_log.txt", "a", encoding="utf-8")
    log_txt.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {header}\n{json.dumps(vars(a))}\n")
    new_csv = not (out / "log.csv").exists()
    csv_f = open(out / "log.csv", "a", newline="")
    writer = csv.writer(csv_f)
    if new_csv:
        writer.writerow(["step", "epoch", "lr", "sft_train_loss_running", "sft_val_loss", "corpus_val_loss", "elapsed_s"])

    if step == 0:   # baseline row: the pretrained model before any fine-tuning
        sv = sft_val_loss(model, val, a, tok.pad_id, device, amp_ctx)
        cv = estimate_loss(model, corpus_val, corpus_offsets, a, device, amp_ctx)
        writer.writerow([0, 0, 0, "", f"{sv:.4f}", f"{cv:.4f}", 0]); csv_f.flush()
        msg = f"step     0 | (pretrained) sft val {sv:.4f} | corpus val {cv:.4f}"
        print(msg); log_txt.write(msg + "\n")
        with open(out / "samples.jsonl", "a", encoding="utf-8") as f:
            for s in sample_answers(model, tok, sample_qs, device, 0):
                f.write(json.dumps(s) + "\n")

    def save(path, full):
        d = {"model": model.state_dict(), "config": cfg.to_dict(), "step": step, "sft_val_loss": best,
             "init": a.init, "train_args": vars(a), "tokenizer": "data/tokenizer/tokenizer.json", "format": "qa"}
        if full:
            d.update({"optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "best": best, "order": order,
                      "pos": pos, "rng": rng.getstate(), "elapsed": elapsed_before + time.time() - t_start})
        torch.save(d, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", path)

    model.train()
    t_start, running = time.time(), []
    while step < a.max_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        if pos + a.batch_size > len(order):
            rng.shuffle(order)
            pos = 0
        x, y, m = make_batch([train[i] for i in order[pos:pos + a.batch_size]], tok.pad_id, device)
        pos += a.batch_size
        with amp_ctx():
            _, loss = model(x, y, loss_mask=m)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        scaler.step(opt)
        scaler.update()
        running.append(loss.item())
        step += 1
        if step % a.eval_every == 0 or step == a.max_steps:
            sv = sft_val_loss(model, val, a, tok.pad_id, device, amp_ctx)
            cv = estimate_loss(model, corpus_val, corpus_offsets, a, device, amp_ctx)
            el = elapsed_before + time.time() - t_start
            run = float(np.mean(running)); running = []
            writer.writerow([step, f"{step * a.batch_size / len(train):.2f}", f"{lr_at(step, a):.2e}", f"{run:.4f}",
                             f"{sv:.4f}", f"{cv:.4f}", f"{el:.0f}"]); csv_f.flush()
            msg = f"step {step:5d} | sft train(run) {run:.4f} | sft val {sv:.4f} | corpus val {cv:.4f} | {el / 60:.1f} min"
            if sv < best:
                best = sv
                save(out / "ckpt_best.pt", full=False)
                msg += "  *best*"
            print(msg); log_txt.write(msg + "\n"); log_txt.flush()
            with open(out / "samples.jsonl", "a", encoding="utf-8") as f:
                for s in sample_answers(model, tok, sample_qs, device, step):
                    f.write(json.dumps(s) + "\n")
            save(last, full=True)

    summary = {"steps": step, "epochs": step * a.batch_size / len(train), "best_sft_val_loss": best,
               "init": a.init, "sft_train_pairs": len(train), "train_minutes": (elapsed_before + time.time() - t_start) / 60,
               "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
