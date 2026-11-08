"""Plot training / validation loss curves from a training log.csv.

    python -m src.plot_losses --log <run-dir>/log.csv --out results/figures/pretrain_loss.png --title "Pretraining"
"""
import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="Training and validation loss")
    a = p.parse_args(argv)
    rows = list(csv.DictReader(open(a.log)))
    steps = [int(r["step"]) for r in rows]
    run = [float(r["train_loss_running"]) for r in rows]
    tr = [float(r["train_loss"]) for r in rows]
    vl = [float(r["val_loss"]) for r in rows]
    best = min(range(len(vl)), key=vl.__getitem__)

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.plot(steps, run, color="#9aa5b1", lw=1, label="train (running avg, dropout on)")
    ax.plot(steps, tr, color="#1f6feb", lw=2, label="train (fixed eval batches)")
    ax.plot(steps, vl, color="#d1242f", lw=2, label="validation")
    ax.scatter([steps[best]], [vl[best]], color="#d1242f", zorder=5)
    ax.annotate(f"best val {vl[best]:.3f} (ppl {math.exp(vl[best]):.2f})\nstep {steps[best]}",
                (steps[best], vl[best]), textcoords="offset points", xytext=(-10, 18), ha="right", fontsize=8)
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss (nats/token)")
    ax.set_title(a.title)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
