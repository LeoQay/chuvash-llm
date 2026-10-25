import collections
import hashlib
import json
import random
import re
import tarfile
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

DATA = Path("data")
RAW = DATA / "raw"
LEIPZIG_URL = "https://downloads.wortschatz-leipzig.de/corpora/chv_community_2017.tar.gz"

# Выглядят похоже, но одно кириллица, другое utf-8. Долгое время пользовались (и продолжают) кириллицей
LOOKALIKES = str.maketrans("ăĕçÿĂĔÇŸ", "ӑӗҫӳӐӖҪӲ")

CHV_LETTERS = set("ӑӗҫӳӐӖҪӲ")
STOPWORDS = Path(__file__).parent / "stopwords"
CHV_STOP = set((STOPWORDS / "chv.txt").read_text().split())
RU_STOP = set((STOPWORDS / "ru.txt").read_text().split())
CUT_SECTIONS = {"асӑрхавсем", "вуламалли", "каҫӑсем", "ҫавӑн пекех пӑхӑр", "литература", "ссылки"}
WORD_RE = re.compile(r"[^\W\d_]+")
LATIN_RE = re.compile(r"[A-Za-z]")
URL_RE = re.compile(r"https?://|www\.|\.ru\b|\.com\b|@")


def download():
    RAW.mkdir(parents=True, exist_ok=True)

    fw = hf_hub_download(
        "HuggingFaceFW/fineweb-2",
        "data/chv_Cyrl/train/000_00000.parquet",
        repo_type="dataset",
        local_dir=RAW,
    )

    wiki = hf_hub_download(
        "wikimedia/wikipedia",
        "20231101.cv/train-00000-of-00001.parquet",
        repo_type="dataset",
        local_dir=RAW,
    )

    chv_community_2017_tar = RAW / "chv_community_2017.tar.gz"
    if not chv_community_2017_tar.exists():
        urllib.request.urlretrieve(LEIPZIG_URL, chv_community_2017_tar)

    with tarfile.open(chv_community_2017_tar) as t:
        f = t.extractfile("chv_community_2017/chv_community_2017-sentences.txt")
        if f is not None:
            lines = f.read().decode().splitlines()
        else:
            lines = []

        sentences = [line.split("\t", 1)[1].strip() for line in lines]

    return fw, wiki, sentences


def load_docs():
    fw_path, wiki_path, sentences = download()

    fw = pd.read_parquet(fw_path, columns=["text", "url"])

    # Удаляю вики, так как добавляю ее из другого источника
    is_wiki_copy = fw.url.str.contains("cv.wikipedia.org", regex=False)
    docs = [("fineweb2", t) for t in fw.text[~is_wiki_copy]]

    docs += [("wikipedia", t) for t in pd.read_parquet(wiki_path, columns=["text"]).text]

    # Последовательности короткие сами по себе, тут я объединяю их в документы побольше
    docs += [("leipzig", "\n".join(sentences[i:i + 30])) for i in range(0, len(sentences), 30)]

    return docs, int(is_wiki_copy.sum())


def normalize(text: str) -> str:
    text = (
        unicodedata.normalize("NFC", text)
            .translate(LOOKALIKES)
            .replace(" ", " ")
            .replace("­", "")
    )

    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))


def keep_line_or_reason_to_drop(line: str) -> str | None:
    words = WORD_RE.findall(line.lower())

    if not words:
        return "no_words"

    letters = sum(len(w) for w in words)

    if len(LATIN_RE.findall(line)) > 0.3 * letters:
        return "latin"

    if URL_RE.search(line) and len(words) < 15:
        return "url"

    if len(line) < 40 and line[-1] not in ".!?…»\"":
        return "short"  # headers, menus, captions

    ru = sum(w in RU_STOP for w in words) / len(words)
    chv = sum(w in CHV_STOP or bool(CHV_LETTERS & set(w)) for w in words) / len(words)

    if ru >= 0.08 and chv < ru:
        return "russian"

    return None


def clean_doc(text: str, seen: set[bytes], stats: dict[str, int], examples: dict[str, list[str]]) -> str:
    kept = []

    for line in text.split("\n"):
        if not line:
            continue

        if line.lower().rstrip(":") in CUT_SECTIONS:
            # Дальше скорее всего будут не интересные нам разделы википедии
            break

        reason = keep_line_or_reason_to_drop(line)

        if reason is None:
            h = hashlib.md5(line.lower().encode()).digest()

            if h in seen:
                reason = "dup_line"
            else:
                seen.add(h)

        if reason is not None:
            stats[reason] += len(line)

            if len(examples[reason]) < 15 and len(line) > 30:
                examples[reason].append(line[:200])

            continue

        kept.append(line)

    return "\n".join(kept)


def chuvash_rate(text: str) -> float:
    cyr = sum("а" <= c <= "я" or c in "ӑӗҫӳё" for c in text.lower())

    return sum(c in CHV_LETTERS for c in text) / max(cyr, 1)


def main():
    docs, n_wiki_copies = load_docs()

    stats = {
        "fineweb_wiki_copies_dropped": n_wiki_copies,
        "raw": collections.Counter(),
        "lookalike_chars": 0,
        "removed_chars": collections.Counter(),
        "clean": collections.Counter(),
    }

    examples = collections.defaultdict(list)
    seen = set()
    rows = []

    for source, text in docs:
        stats["raw"][f"{source}_docs"] += 1
        stats["raw"][f"{source}_chars"] += len(text)

        stats["lookalike_chars"] += sum(c in "ăĕçÿĂĔÇŸ" for c in text)

        text = clean_doc(normalize(text), seen, stats["removed_chars"], examples)

        # Не берем слишком короткие последовательности
        if len(text) < 200:
            stats["removed_chars"]["short_doc"] += len(text)
            continue

        if chuvash_rate(text) < 0.01:
            stats["removed_chars"]["non_chuvash_doc"] += len(text)

            if len(examples["non_chuvash_doc"]) < 15:
                examples["non_chuvash_doc"].append(text[:200])

            continue

        stats["clean"][f"{source}_docs"] += 1
        stats["clean"][f"{source}_chars"] += len(text)

        rows.append((text, source))

    random.seed(0)

    df = pd.DataFrame(rows, columns=["text", "source"])
    df["split"] = ["val" if random.random() < 0.005 else "train" for _ in range(len(df))]
    df.to_parquet(DATA / "clean.parquet")

    words = collections.Counter(w for t in df.text[df.split == "train"] for w in WORD_RE.findall(t.lower()))

    stats["words_total"] = sum(words.values())
    stats["words_unique"] = len(words)
    stats["hapax_share_of_vocab"] = sum(c == 1 for c in words.values()) / len(words)
    stats["top_words"] = words.most_common(50)
    stats["clean_chars_total"] = int(df.text.str.len().sum())
    stats["clean_mb_utf8"] = round(sum(len(t.encode()) for t in df.text) / 1e6, 1)
    stats["split_docs"] = df.split.value_counts().to_dict()
    stats["val_chars"] = int(df.text[df.split == "val"].str.len().sum())

    pd.DataFrame(words.most_common(), columns=["word", "count"]).to_csv(DATA / "word_counts.tsv", sep="\t", index=False)

    (DATA / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))

    (DATA / "removed_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=1))

    print(json.dumps({k: v for k, v in stats.items() if k != "top_words"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
