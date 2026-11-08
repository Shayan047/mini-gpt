"""Pretraining: next-token prediction on the synthetic corpus (spec section 5).

    python -m src.train --out-dir /content/drive/MyDrive/mini-gpt/pretrain

Resumes automatically from <out-dir>/ckpt_last.pt if it exists (Colab sessions can disconnect).

Writes to <out-dir>:
    ckpt_last.pt            model + optimizer + scaler + step + RNG states (for resuming)
    ckpt_best.pt            model weights with the lowest validation loss so far (+ config)
    snapshots/step_XXXXX.pt model weights at regular intervals (to study how text changes over training)
    log.csv                 step, tokens seen, lr, train loss, validation loss, perplexity, elapsed time
    train_log.txt           human-readable log
    samples.jsonl           generations from fixed prompts at every evaluation
    summary.json            final numbers
"""
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.model import GPT, GPTConfig
from src.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent.parent

SAMPLE_PROMPTS = [
    "Plastic bags",
    "Leftover paint",
    "If you live in an apartment,",
    "Question: How do I dispose of old batteries?\nAnswer:",
    "The green organics cart",
    "Bulky items",
]


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--out-dir", default=str(ROOT / "checkpoints" / "pretrain"))
    # model
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    # optimization
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr", type=float, default=6e-5)
    p.add_argument("--warmup-steps", type=int, default=150)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # evaluation / saving
    p.add_argument("--eval-every", type=int, default=150)
    p.add_argument("--eval-batches", type=int, default=40)
    p.add_argument("--snapshot-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true", help="disable fp16 mixed precision on GPU")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args(argv)


def lr_at(step, a):
    """Linear warmup, then cosine decay from lr to min_lr."""
    if step < a.warmup_steps:
        return a.lr * (step + 1) / a.warmup_steps
    progress = min(1.0, (step - a.warmup_steps) / max(1, a.max_steps - a.warmup_steps))
    return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * progress))


class TokenData:
    def __init__(self, path, context_length):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.T = context_length

    def batch(self, idx, device):
        x = torch.stack([torch.from_numpy(self.data[i:i + self.T].astype(np.int64)) for i in idx])
        y = torch.stack([torch.from_numpy(self.data[i + 1:i + 1 + self.T].astype(np.int64)) for i in idx])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def random_offsets(self, n, rng):
        return rng.integers(0, len(self.data) - self.T - 1, size=n)


def configure_optimizer(model, a):
    """AdamW; weight decay only on matrices (not on biases, LayerNorm or position/token embeddings)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.dim() >= 2 and "emb" not in name else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), eps=1e-8)


@torch.no_grad()
def estimate_loss(model, data, offsets, a, device, amp_ctx):
    model.eval()
    losses = []
    for i in range(0, len(offsets), a.batch_size):
        x, y = data.batch(offsets[i:i + a.batch_size], device)
        with amp_ctx():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def sample_texts(model, tok, device, step):
    out = []
    g = torch.Generator(device=device).manual_seed(0)
    for prompt in SAMPLE_PROMPTS:
        ids = torch.tensor([tok.encode(prompt)], device=device)
        for temp in (0.0, 0.8):
            res = model.generate(ids, 60, temperature=temp, top_k=50 if temp else None,
                                 stop_ids=(tok.eot_id,), generator=g)
            out.append({"step": step, "prompt": prompt, "temperature": temp,
                        "text": tok.decode(res[0].tolist(), skip_special=True)})
    model.train()
    return out


def main(argv=None):
    a = get_args(argv)
    out = Path(a.out_dir)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    use_amp = device == "cuda" and not a.no_amp
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_amp else (lambda: torch.autocast("cpu", enabled=False))
    torch.manual_seed(a.seed)
    np_rng = np.random.default_rng(a.seed)

    meta = json.load(open(Path(a.data_dir) / "tokens" / "meta.json"))
    tok = BPETokenizer.load(Path(a.data_dir) / "tokenizer" / "tokenizer.json")
    train = TokenData(Path(a.data_dir) / "tokens" / "train.bin", a.context_length)
    val = TokenData(Path(a.data_dir) / "tokens" / "val.bin", a.context_length)
    # fixed evaluation windows so every evaluation is measured on exactly the same text
    eval_rng = np.random.default_rng(0)
    val_offsets = val.random_offsets(a.eval_batches * a.batch_size, eval_rng)
    train_eval_offsets = train.random_offsets(a.eval_batches * a.batch_size, eval_rng)

    cfg = GPTConfig(vocab_size=meta["vocab_size"], context_length=a.context_length, n_layer=a.n_layer,
                    n_head=a.n_head, d_model=a.d_model, d_ff=4 * a.d_model, dropout=a.dropout)
    model = GPT(cfg).to(device)
    opt = configure_optimizer(model, a)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step, best_val, elapsed_before = 0, float("inf"), 0.0
    last = out / "ckpt_last.pt"
    if last.exists() and not a.no_resume:
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        step, best_val, elapsed_before = ck["step"], ck["best_val"], ck.get("elapsed", 0.0)
        np_rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"])
        if device == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"])
        print(f"resumed from step {step} (best val {best_val:.4f})")

    tokens_per_step = a.batch_size * a.context_length
    n_params = model.num_params()
    header = (f"device={device} amp={use_amp} params={n_params:,} train_tokens={len(train.data):,} "
              f"tokens/step={tokens_per_step:,} steps={a.max_steps} (~{a.max_steps * tokens_per_step / len(train.data):.1f} epochs)")
    print(header)
    log_txt = open(out / "train_log.txt", "a", encoding="utf-8")
    log_txt.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {header}\n{json.dumps(vars(a))}\n")
    csv_path = out / "log.csv"
    new_csv = not csv_path.exists()
    csv_f = open(csv_path, "a", newline="")
    writer = csv.writer(csv_f)
    if new_csv:
        writer.writerow(["step", "tokens_seen", "lr", "train_loss_running", "train_loss", "val_loss", "val_ppl", "elapsed_s"])

    def save_last(step):
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "best_val": best_val, "elapsed": elapsed_before + time.time() - t_start,
                    "config": cfg.to_dict(), "train_args": vars(a), "np_rng": np_rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None}, str(last) + ".tmp")
        os.replace(str(last) + ".tmp", last)        # atomic: a disconnect mid-save cannot corrupt it

    def save_weights(path, step, val_loss):
        torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": step, "val_loss": val_loss,
                    "train_args": vars(a), "tokenizer": "data/tokenizer/tokenizer.json"}, path)

    model.train()
    t_start, running = time.time(), []
    while step < a.max_steps:
        lr = lr_at(step, a)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = train.batch(train.random_offsets(a.batch_size, np_rng), device)
        with amp_ctx():
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        scaler.step(opt)
        scaler.update()
        running.append(loss.item())
        step += 1

        if step % a.eval_every == 0 or step == a.max_steps:
            tr = estimate_loss(model, train, train_eval_offsets, a, device, amp_ctx)
            vl = estimate_loss(model, val, val_offsets, a, device, amp_ctx)
            el = elapsed_before + time.time() - t_start
            run = float(np.mean(running))
            running = []
            writer.writerow([step, step * tokens_per_step, f"{lr:.2e}", f"{run:.4f}", f"{tr:.4f}", f"{vl:.4f}",
                             f"{math.exp(vl):.3f}", f"{el:.0f}"])
            csv_f.flush()
            msg = (f"step {step:5d} | lr {lr:.2e} | train(run) {run:.4f} | train {tr:.4f} | val {vl:.4f} "
                   f"| val ppl {math.exp(vl):.2f} | {el / 60:.1f} min")
            if vl < best_val:
                best_val = vl
                save_weights(out / "ckpt_best.pt", step, vl)
                msg += "  *best*"
            print(msg)
            log_txt.write(msg + "\n")
            log_txt.flush()
            with open(out / "samples.jsonl", "a", encoding="utf-8") as f:
                for s in sample_texts(model, tok, device, step):
                    f.write(json.dumps(s) + "\n")
            if step % a.snapshot_every == 0 or step == a.max_steps:
                save_weights(out / "snapshots" / f"step_{step:05d}.pt", step, vl)
            save_last(step)

    summary = {"steps": step, "tokens_seen": step * tokens_per_step, "best_val_loss": best_val,
               "best_val_ppl": math.exp(best_val), "params": n_params, "config": cfg.to_dict(),
               "train_minutes": (elapsed_before + time.time() - t_start) / 60, "device": device,
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
