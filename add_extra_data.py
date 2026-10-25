import collections
import gzip
import json
import re
import shutil
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

from prepare_data import chuvash_rate, clean_doc, normalize, text_hash


DATA = Path("data")
EXTRA = DATA / "raw" / "extra"
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")

EXTRA_FILES = [
    ("alexantonov/chuvash_mono", "data/train-00000-of-00001.parquet"),
    ("alexantonov/chuvash_mono", "data/train-kusaru-20261018.parquet"),
    ("alexantonov/chuvash_parallel", "chv-ru.chv"),
    ("alexantonov/chuvash_parallel", "chv-en.chv"),
    ("allenai/MADLAD-400", "data/cv/cv_clean_0000.jsonl.gz"),
    ("MaLA-LM/mala-monolingual-split", "chv_Cyrl/train.parquet"),
    ("HuggingFaceFW/fineweb-2", "data/chv_Cyrl_removed/train/000_00000.parquet"),
]


def download():
    for repo, filename in EXTRA_FILES:
        hf_hub_download(
            repo,
            filename,
            repo_type="dataset",
            local_dir=EXTRA / repo.split("/")[1],
        )


def unescape(text) -> str:
    # В MaLA и MADLAD часть переносов строк записана буквально как "\n"
    return str(text).replace("\\n", "\n")


def sentence_sources() -> Iterator[tuple[str, pd.Series]]:
    # Корпуса из отдельных предложений, без структуры документов
    mono = EXTRA / "chuvash_mono" / "data"

    yield "chuvash_mono", pd.read_parquet(mono / "train-00000-of-00001.parquet").chv
    yield "chuvash_mono_kusaru", pd.read_parquet(mono / "train-kusaru-20261018.parquet").chv

    for filename in ["chv-ru.chv", "chv-en.chv"]:
        lines = (EXTRA / "chuvash_parallel" / filename).read_text().splitlines()
        yield "parallel_" + filename.split(".")[0], pd.Series(lines)

    # MaLA (сборник Glot500: Библия/NLLB, OSCAR, Leipzig, …) тоже хранится построчно
    mala = pd.read_parquet(EXTRA / "mala-monolingual-split" / "chv_Cyrl" / "train.parquet", columns=["text"])
    yield "mala", mala.text


def document_sources() -> Iterator[tuple[str, pd.Series]]:
    with gzip.open(EXTRA / "MADLAD-400" / "data" / "cv" / "cv_clean_0000.jsonl.gz", "rt") as f:
        yield "madlad400_clean", pd.Series([json.loads(line)["text"] for line in f])

    # Страницы, которые отбросили фильтры качества FineWeb-2 (в основном короткие строки и списки).
    # Наши фильтры проверяют их заново
    fw = pd.read_parquet(
        EXTRA / "fineweb-2" / "data" / "chv_Cyrl_removed" / "train" / "000_00000.parquet",
        columns=["text", "url"],
    )
    is_wiki_copy = fw.url.str.contains("cv.wikipedia.org", regex=False)

    yield "fineweb2_removed", fw.text[~is_wiki_copy]


def hash_v1(v1: pd.DataFrame) -> tuple[set[bytes], set[bytes]]:
    # Хэширую весь v1 с валидацией, чтобы текст из валидации не попал в обучение
    seen_lines, seen_sents = set(), set()

    for text in v1.text:
        for line in text.split("\n"):
            seen_lines.add(text_hash(line))
            seen_sents.update(text_hash(s) for s in SENT_SPLIT.split(line) if len(s) >= 20)

    return seen_lines, seen_sents


def main():
    download()

    v1 = pd.read_parquet(DATA / "clean.parquet")
    seen_lines, seen_sents = hash_v1(v1)
    print(f"v1: {len(seen_lines):,} lines, {len(seen_sents):,} sentences hashed")

    stats = {}
    rows = []
    examples = collections.defaultdict(list)

    for source, sents in sentence_sources():
        source_stats = collections.Counter(raw_rows=len(sents), raw_chars=int(sents.str.len().sum()))
        kept = []

        for sent in sents:
            sent = normalize(unescape(sent))
            h = text_hash(sent)

            if len(sent) < 20 or h in seen_sents:
                source_stats["dup_or_short_chars"] += len(sent)
                continue

            seen_sents.add(h)

            # Те же фильтры строк, что и в v1 (русский, латиница, ссылки, …)
            line = clean_doc(sent, seen_lines, source_stats, examples)

            if line:
                kept.append(line)

        # Предложения перемешаны: собираю их в псевдодокументы по 30, как Leipzig в v1
        docs = ["\n".join(kept[i:i + 30]) for i in range(0, len(kept), 30)]
        docs = [d for d in docs if chuvash_rate(d) >= 0.01]

        rows += [(d, source) for d in docs]

        source_stats.update(clean_docs=len(docs), clean_chars=sum(map(len, docs)))
        stats[source] = dict(source_stats)
        print(source, stats[source])

    for source, texts in document_sources():
        source_stats = collections.Counter(raw_rows=len(texts), raw_chars=int(texts.str.len().sum()))

        for text in texts:
            text = clean_doc(normalize(unescape(text)), seen_lines, source_stats, examples)

            if len(text) < 200 or chuvash_rate(text) < 0.01:
                source_stats["short_or_non_chuvash_doc_chars"] += len(text)
                continue

            # Выкидываю документы, большая часть предложений которых уже есть (почти дубли страниц из v1)
            sents = [s for s in SENT_SPLIT.split(text.replace("\n", " ")) if len(s) >= 20]
            known = sum(text_hash(s) in seen_sents for s in sents)
            seen_sents.update(text_hash(s) for s in sents)

            if sents and known / len(sents) > 0.5:
                source_stats["near_dup_doc_chars"] += len(text)
                continue

            rows.append((text, source))

            source_stats["clean_docs"] += 1
            source_stats["clean_chars"] += len(text)

        stats[source] = dict(source_stats)
        print(source, stats[source])

    extra = pd.DataFrame(rows, columns=["text", "source"])
    extra["split"] = "train"

    # Валидация остается от v1, чтобы bits per char сравнивались с моделями на v1
    v2 = pd.concat([v1, extra], ignore_index=True)
    v2.to_parquet(DATA / "clean_v2.parquet")

    stats["total_v2"] = {
        "docs": len(v2),
        "chars": int(v2.text.str.len().sum()),
        "train_chars": int(v2.text[v2.split == "train"].str.len().sum()),
        "by_source_chars": v2.groupby("source").text.apply(lambda t: int(t.str.len().sum())).to_dict(),
    }

    (DATA / "stats_v2.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))

    print(json.dumps(stats["total_v2"], indent=1))

    tok = PreTrainedTokenizerFast.from_pretrained("tokenizers/bpe16000")

    out = DATA / "tok16000_v2"
    out.mkdir(exist_ok=True)

    ids = tok(list(v2.text[v2.split == "train"]))["input_ids"]
    flat = np.fromiter((t for doc in ids for t in (*doc, tok.eos_token_id)), dtype=np.uint16)
    flat.tofile(out / "train.bin")

    shutil.copy(DATA / "tok16000" / "val.bin", out / "val.bin")

    print(f"train tokens v2: {len(flat):,}")


if __name__ == "__main__":
    main()
