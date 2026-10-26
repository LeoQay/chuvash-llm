import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from common import Batches, evaluate

# 32 факта, у каждого 3 перефразировки и 4 варианта ответа
# Правильный вариант всегда первый
ITEMS = json.loads((Path(__file__).parent / "data" / "benchmark_chv.json").read_text())
PROMPTS = [p for it in ITEMS for p in it["prompts"]]
OPTIONS = [it["options"] for it in ITEMS for _ in it["prompts"]]


@torch.no_grad()
def option_logprobs(model, tokenizer, device: str, pairs: list[tuple[str, str]]) -> list[float]:
    # EOS в начале = начало нового документа
    prompts = [[tokenizer.eos_token_id] + tokenizer(p).input_ids for p, _ in pairs]
    options = [tokenizer(" " + o).input_ids for _, o in pairs]

    batch = tokenizer.pad({"input_ids": [p + o for p, o in zip(prompts, options)]}, return_tensors="pt").to(device)
    logp = model(**batch).logits.float().log_softmax(-1)

    scores = []

    for i, (p, o) in enumerate(zip(prompts, options)):
        a, b = len(p), len(p) + len(o)
        scores.append(logp[i, a - 1:b - 1].gather(-1, batch.input_ids[i, a:b, None]).sum().item())

    return scores


def run_benchmark(model, tokenizer, device: str) -> dict:
    was_training = model.training
    model.eval()

    pairs = [(p, o) for p, opts in zip(PROMPTS, OPTIONS) for o in opts]
    scores = np.array(option_logprobs(model, tokenizer, device, pairs)).reshape(len(ITEMS), 3, 4)

    unique = sorted({o for _, o in pairs})
    prior = dict(zip(unique, option_logprobs(model, tokenizer, device, [("", o) for o in unique])))

    model.train(was_training)

    # [факт, перефразировка, вариант]
    priors = np.array([[prior[o] for o in it["options"]] for it in ITEMS])[:, None, :]
    lengths = np.array([[len(o) for o in it["options"]] for it in ITEMS])[:, None, :]

    picks = (scores - priors).argmax(-1)

    summary = {
        "acc": (scores.argmax(-1) == 0).mean(),
        "acc_norm": ((scores / lengths).argmax(-1) == 0).mean(),
        "acc_calib": (picks == 0).mean(),
        "consistency": (picks == picks[:, :1]).all(1).mean(),
        "consistent_correct": (picks == 0).all(1).mean(),
    }

    items = [
        {"fact": it["fact"], "answers": [it["options"][j] for j in row], "all_correct": bool((row == 0).all())}
        for it, row in zip(ITEMS, picks)
    ]

    return {"summary": {k: round(float(v), 3) for k, v in summary.items()}, "items": items}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--out", default="results/benchmark.json")

    return p.parse_args()


def main():
    args = parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    val_chars = json.loads(Path("data/stats.json").read_text())["val_chars"]
    autocast = torch.autocast(device, dtype=torch.bfloat16)

    results = {}

    for ckpt in args.checkpoints:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(ckpt)
        model = AutoModelForCausalLM.from_pretrained(ckpt).to(device)

        name = Path(ckpt).name

        val = Batches(f"data/tok{len(tokenizer) // 1000 * 1000}/val.bin", 512, 32, device)
        val_loss, bpc = evaluate(model, val, val_chars, autocast)

        bench = run_benchmark(model, tokenizer, device)

        results[name] = {
            "val_loss": round(val_loss, 4),
            "ppl": round(math.exp(val_loss), 1),
            "bpc": round(bpc, 4),
            **bench,
        }

        print(f"{name:28s} ppl {math.exp(val_loss):9.1f}  bpc {bpc:.3f}  {bench['summary']}")

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
