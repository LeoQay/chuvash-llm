import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

DATA = Path("data")
TOKENIZERS = Path("tokenizers")

VOCABS = [8000, 16000, 32000]
EOS, PAD = "<|endoftext|>", "<pad>"
SAMPLE = "Чӑваш Республикин тӗп хули — Шупашкар. Вӑл Атӑл юханшывӗн сылтӑм ҫыранӗнче вырнаҫнӑ."


def train_bpe(texts: pd.Series, vocab_size: int) -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOS, PAD],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer=trainer)

    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token=EOS, bos_token=EOS, pad_token=PAD)


def encode_to_bin(tok: PreTrainedTokenizerFast, texts: pd.Series, path: Path) -> tuple[list[list[int]], int]:
    ids = tok(list(texts))["input_ids"]

    flat = np.fromiter((t for doc in ids for t in (*doc, tok.eos_token_id)), dtype=np.uint16)
    flat.tofile(path)

    return ids, len(flat)


def main():
    df = pd.read_parquet(DATA / "clean.parquet")

    train, val = df.text[df.split == "train"], df.text[df.split == "val"]
    val_chars = int(val.str.len().sum())
    val_words = int(val.str.split().str.len().sum())

    report = {}

    for v in VOCABS:
        tok = train_bpe(train, v)
        tok.save_pretrained(TOKENIZERS / f"bpe{v}")

        out = DATA / f"tok{v}"
        out.mkdir(parents=True, exist_ok=True)

        train_ids, n_train = encode_to_bin(tok, train, out / "train.bin")
        val_ids, n_val = encode_to_bin(tok, val, out / "val.bin")

        assert tok.decode(val_ids[0]) == val.iloc[0]

        freq = collections.Counter(t for doc in train_ids for t in doc)
        counts = np.array([freq.get(i, 0) for i in range(len(tok))])
        n_val_tokens = sum(map(len, val_ids))

        report[v] = {
            "train_tokens": n_train,
            "val_tokens": n_val,
            "chars_per_token": round(val_chars / n_val_tokens, 3),
            "tokens_per_word": round(n_val_tokens / val_words, 3),
            "unused_tokens_pct": round(100 * (counts == 0).mean(), 2),
            "rare_tokens_lt10_pct": round(100 * (counts < 10).mean(), 2),
            "rare_tokens_lt100_pct": round(100 * (counts < 100).mean(), 2),
            "embedding_params_d512": v * 512,
            "sample": tok.convert_ids_to_tokens(tok(SAMPLE)["input_ids"]),
        }

        np.save(out / "token_counts.npy", counts)

    (TOKENIZERS / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
