import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    MixtralConfig,
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)

from common import Batches, evaluate, lm_loss
from eval_benchmark import run_benchmark
from moe_experts import swap_experts

PRESETS = {
    "tiny": dict(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=704,
    ),
    "small": dict(
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=1408,
    ),
    "medium": dict(
        hidden_size=768,
        num_hidden_layers=8,
        num_attention_heads=12,
        intermediate_size=2048,
    ),
    "large": dict(
        hidden_size=1024,
        num_hidden_layers=8,
        num_attention_heads=16,
        intermediate_size=2752,
    ),

    # Dense, подобранная так, чтобы на MPS училась примерно со скоростью large_moe (~1.25x вычислений large)
    "xl": dict(hidden_size=1152, num_hidden_layers=8, num_attention_heads=18, intermediate_size=3072),

    # То же число активных параметров, что у large
    "large_moe": dict(
        hidden_size=1024,
        num_hidden_layers=8,
        num_attention_heads=16,
        intermediate_size=1376,
        num_local_experts=8,
        num_experts_per_tok=2,
    ),

    # То же число активных параметров, что у tiny
    "tiny_moe": dict(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=352,
        num_local_experts=8,
        num_experts_per_tok=2,
    ),
}

PROMPTS = ["Чӑваш Республикин тӗп хули", "Пирӗн ялта", "Ҫулла ачасем", "Константин Иванов"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--preset", default="small", choices=PRESETS)
    p.add_argument("--vocab", type=int, default=16000)
    p.add_argument("--data", default=None, help="token dir, default data/tok{vocab}")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--tokens", type=float, default=70e6, help="training budget in tokens")
    p.add_argument("--max_steps", type=int, default=0, help="override the step count (smoke tests)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--no_compile", action="store_true", help="torch.compile gives ~1.5x on MPS")
    p.add_argument(
        "--experts_impl",
        default="eager",
        choices=["eager", "batched_mm", "grouped_mm", "packed", "grouped"],
        help="HF implementations, or our own swapped-in modules (packed / grouped, see moe_experts.py)",
    )
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--name", default=None)

    return p.parse_args()


@torch.no_grad()
def samples(model, tokenizer, device: str) -> list[str]:
    model.eval()

    out = []

    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        gen = model.generate(
            ids,
            max_new_tokens=60,
            do_sample=True,
            top_k=50,
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id,
        )

        out.append(tokenizer.decode(gen[0]))

    model.train()

    return out


def build_model(args: argparse.Namespace, tokenizer: PreTrainedTokenizerFast):
    preset = PRESETS[args.preset]

    common = dict(
        vocab_size=len(tokenizer),
        max_position_embeddings=args.seq_len,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attn_implementation="sdpa",
        **preset,
    )

    own_experts = args.experts_impl in ("packed", "grouped")

    if "num_local_experts" in preset:
        config = MixtralConfig(
            num_key_value_heads=preset["num_attention_heads"],
            output_router_logits=True,
            router_aux_loss_coef=0.01,
            experts_implementation="eager" if own_experts else args.experts_impl,
            **common,
        )
    else:
        config = LlamaConfig(**common)

    model = AutoModelForCausalLM.from_config(config)

    if own_experts:
        swap_experts(model, args.experts_impl)

    return model, config


def main():
    args = parse_args()

    name = args.name or f"{args.preset}_v{args.vocab // 1000}k"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    torch.manual_seed(0)
    np.random.seed(0)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(f"tokenizers/bpe{args.vocab}")

    data = Path(args.data or f"data/tok{args.vocab}")

    train = Batches(data / "train.bin", args.seq_len, args.batch_size, device)
    val = Batches(data / "val.bin", args.seq_len, args.batch_size, device)

    val_chars = json.loads(Path("data/stats.json").read_text())["val_chars"]

    model, config = build_model(args, tokenizer)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())

    # как в GPT-3 / nanoGPT
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]

    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    tokens_per_step = args.batch_size * args.seq_len
    steps = args.max_steps or int(args.tokens // tokens_per_step)

    scheduler = get_cosine_schedule_with_warmup(opt, args.warmup, steps)

    autocast = torch.autocast(device, dtype=torch.bfloat16, enabled=args.dtype == "bf16")

    if args.no_compile:
        step_model = model
    else:
        step_model = torch.compile(model)

    writer = SummaryWriter(f"runs/{name}")
    writer.add_text("config", json.dumps({**vars(args), "params": n_params, "steps": steps}))

    ckpt = Path(f"checkpoints/{name}")
    best = float("inf")
    t0 = time.time()
    tokens_seen = 0

    for step in range(1, steps + 1):
        with autocast:
            loss, aux = lm_loss(step_model, train.random())

        if aux is None:
            loss.backward()
        else:
            (loss + config.router_aux_loss_coef * aux).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()
        scheduler.step()
        opt.zero_grad()

        tokens_seen += tokens_per_step

        if step % 10 == 0:
            # .item() синхронизирует GPU, поэтому замер скорости ниже честный
            loss_v = loss.item()
            tok_s = 10 * tokens_per_step / (time.time() - t0)
            t0 = time.time()

            writer.add_scalar("train/loss", loss_v, tokens_seen)
            writer.add_scalar("train/grad_norm", grad_norm.item(), tokens_seen)

            if aux is not None:
                writer.add_scalar("train/router_aux_loss", aux.item(), tokens_seen)

            drops = [
                layer.mlp.experts.drop_rate
                for layer in model.model.layers
                if getattr(getattr(layer.mlp, "experts", None), "drop_rate", None) is not None
            ]

            if drops:
                writer.add_scalar("train/expert_drop_rate", torch.stack(drops).mean().item(), tokens_seen)

            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], tokens_seen)
            writer.add_scalar("perf/tokens_per_sec", tok_s, tokens_seen)

            if step % 50 == 0:
                print(f"step {step}/{steps} loss {loss_v:.3f} lr {scheduler.get_last_lr()[0]:.2e} {tok_s:,.0f} tok/s")

        if step % args.eval_every == 0 or step == steps:
            val_loss, bpc = evaluate(model, val, val_chars, autocast)
            bench = run_benchmark(model, tokenizer, device)

            writer.add_scalar("val/loss", val_loss, tokens_seen)
            writer.add_scalar("val/bpc", bpc, tokens_seen)

            for k, v in bench["summary"].items():
                writer.add_scalar(f"bench/{k}", v, tokens_seen)

            gens = samples(model, tokenizer, device)
            writer.add_text("samples", "\n\n".join(gens), tokens_seen)

            print(
                f"[eval] step {step} val_loss {val_loss:.3f} ppl {math.exp(val_loss):.1f} bpc {bpc:.3f} "
                f"bench {bench['summary']}\n  " + "\n  ".join(g.replace("\n", " ") for g in gens)
            )

            if val_loss < best:
                best = val_loss

                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)

                state = {"step": step, "tokens_seen": tokens_seen, "val_loss": val_loss, "bpc": bpc, "args": vars(args)}
                (ckpt / "train_state.json").write_text(json.dumps(state))

            t0 = time.time()

    writer.close()


if __name__ == "__main__":
    main()
