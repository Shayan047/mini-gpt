"""Generate text from a trained checkpoint (used for the demo and by graders).

Examples
    # one prompt, pretrained model (plain text continuation)
    python -m src.generate --ckpt checkpoints/pretrain/ckpt_best.pt --prompt "Pizza boxes"

    # fine-tuned model, question-answer mode
    python -m src.generate --ckpt checkpoints/sft/ckpt_best.pt --qa --prompt "How do I get rid of old paint?"

    # interactive session (type prompts, empty line or 'quit' to exit)
    python -m src.generate --ckpt checkpoints/sft/ckpt_best.pt --qa

Options: --temperature (0 = greedy), --top-k, --top-p, --max-new-tokens, --seed, --device
"""
import argparse
import sys
import time
from pathlib import Path

import torch

from src.model import load_checkpoint
from src.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent.parent


def qa_prompt(question):
    """Fine-tuning format: <|user|> question<|assistant|> answer<|endoftext|>"""
    return f"<|user|> {question.strip()}<|assistant|>"


def run(model, tok, prompt, a, device):
    text = qa_prompt(prompt) if a.qa else prompt
    ids = torch.tensor([tok.encode(text, allow_special=a.qa)], device=device)
    g = torch.Generator(device=device).manual_seed(a.seed) if a.seed is not None else None
    t0 = time.time()
    out = model.generate(ids, a.max_new_tokens, temperature=a.temperature, top_k=a.top_k, top_p=a.top_p,
                         stop_ids=(tok.eot_id,), generator=g)
    new = out[0, ids.shape[1]:].tolist()
    reply = tok.decode(new, skip_special=True)
    return reply, len(new), time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate text with the Tidewell Mini GPT")
    ap.add_argument("--ckpt", required=True, help="path to a checkpoint .pt file")
    ap.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer" / "tokenizer.json"))
    ap.add_argument("--prompt", help="prompt text; omit for an interactive session")
    ap.add_argument("--qa", action="store_true", help="wrap the prompt as a question (use with the fine-tuned model)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    tok = BPETokenizer.load(a.tokenizer)
    model, cfg, ck = load_checkpoint(a.ckpt, a.device)
    print(f"Loaded {a.ckpt} (step {ck.get('step')}, {model.num_params():,} parameters, "
          f"{cfg.n_layer} layers, d_model {cfg.d_model}, vocab {cfg.vocab_size}) on {a.device}. "
          f"Mode: {'question answering' if a.qa else 'text continuation'}, temperature {a.temperature}.", file=sys.stderr)

    if a.prompt is not None:
        reply, n, dt = run(model, tok, a.prompt, a, a.device)
        print(reply if a.qa else a.prompt + reply)
        return
    label = "Question" if a.qa else "Prompt"
    while True:
        try:
            p = input(f"\n{label}> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not p or p.lower() in ("quit", "exit"):
            break
        reply, n, dt = run(model, tok, p, a, a.device)
        print((reply if a.qa else p + reply).strip())
        print(f"[{n} tokens in {dt:.1f}s]", file=sys.stderr)


if __name__ == "__main__":
    main()
