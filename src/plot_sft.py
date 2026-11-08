"""Plot fine-tuning curves from the SFT log.csv.

    python -m src.plot_sft --log <sft-run>/log.csv --out results/figures/sft_loss.png
"""
import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rows = list(csv.DictReader(open(a.log)))
    steps = [int(r["step"]) for r in rows]
    sv = [float(r["sft_val_loss"]) for r in rows]
    cv = [float(r["corpus_val_loss"]) for r in rows]
    tr = [(int(r["step"]), float(r["sft_train_loss_running"])) for r in rows if r["sft_train_loss_running"]]
    best = min(range(len(sv)), key=sv.__getitem__)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    ax[0].plot(*zip(*tr), color="#9aa5b1", lw=1.2, label="SFT train (running avg)")
    ax[0].plot(steps, sv, "o-", color="#d1242f", ms=3, lw=2, label="SFT validation (answer tokens)")
    ax[0].scatter([steps[best]], [sv[best]], s=60, color="#d1242f", zorder=5)
    ax[0].annotate(f"best {sv[best]:.3f} @ step {steps[best]}", (steps[best], sv[best]), textcoords="offset points",
                   xytext=(0, 12), ha="center", fontsize=8)
    ax[0].set_title("Fine-tuning loss (step 0 = pretrained model)"); ax[0].set_xlabel("fine-tuning step")
    ax[0].set_ylabel("cross-entropy (nats/token)"); ax[0].legend(frameon=False, fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(steps, cv, "s-", color="#8250df", ms=3, lw=2)
    ax[1].set_title("Pretraining-corpus validation loss (forgetting)"); ax[1].set_xlabel("fine-tuning step")
    ax[1].set_ylabel("cross-entropy (nats/token)"); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
