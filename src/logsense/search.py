"""Phase 5: find similar past events with sentence embeddings + FAISS.

Hugging Face concepts used here
-------------------------------
* Sentence embeddings — an encoder gives one vector per *token*; a sentence-embedding model
  pools them (``all-MiniLM-L6-v2`` averages them: "mean pooling") into one fixed-size vector per
  *text* — 384 numbers here.
* Why a sentence-transformers model, not plain DistilBERT — MiniLM was further trained
  *contrastively* on ~1B sentence pairs: pairs that mean the same thing are pulled together,
  unrelated ones pushed apart. So "close vectors" reliably means "similar meaning". Raw
  DistilBERT vectors were never trained for that.
* Cosine similarity — the cosine of the angle between two vectors (1 = same direction).
  If every vector is first scaled to length 1 (``normalize_embeddings=True``), cosine
  similarity is just the dot product, which is what FAISS ``IndexFlatIP`` computes.
* FAISS — a vector-search library. ``IndexFlatIP`` is exact brute force (compare the query with
  every stored vector): microseconds for our ~1.5k vectors. Approximate indexes (IVF, HNSW)
  trade a little accuracy for speed and only pay off at millions of vectors.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from datasets import load_from_disk

from logsense.baseline import make_tfidf_vectorizer
from logsense.config import load_config

# texts -> float32 array of shape (n, dim), each row L2-normalized
Encoder = Callable[[list[str]], np.ndarray]
EVAL_SPLITS = ("test", "test_unseen")


def index_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["models_dir"]) / "search"


@lru_cache(maxsize=2)
def _sentence_model(name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)  # downloads from the Hub on first use, then cached


def load_encoder(cfg: dict[str, Any]) -> Encoder:
    model = _sentence_model(cfg["models"]["embeddings"])

    def encode(texts: list[str]) -> np.ndarray:
        vecs = model.encode(
            list(texts),
            batch_size=cfg["search"]["batch_size"],
            normalize_embeddings=True,  # length-1 vectors -> dot product == cosine similarity
            convert_to_numpy=True,
        )
        return vecs.astype("float32")  # FAISS works in float32

    return encode


def build(cfg: dict[str, Any], encode: Encoder | None = None) -> faiss.Index:
    """Embed the train split, build a FAISS index, save it with per-row metadata."""
    encode = encode or load_encoder(cfg)
    train = load_from_disk(cfg["data"]["processed_dir"])["train"]
    names = train.features["label"].names

    vecs = encode(list(train["text"]))
    index = faiss.IndexFlatIP(vecs.shape[1])  # exact inner-product (= cosine) search
    index.add(vecs)

    out = index_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "index.faiss"))
    with open(out / "meta.jsonl", "w", encoding="utf-8") as f:
        for ex in train:  # row i of meta.jsonl describes vector i of the index
            meta = {"text": ex["text"], "category": names[ex["label"]], "vendor": ex["vendor"]}
            f.write(json.dumps(meta) + "\n")
    _load_index.cache_clear()
    return index


@lru_cache(maxsize=2)
def _load_index(path: str) -> tuple[faiss.Index, list[dict[str, Any]]]:
    index = faiss.read_index(str(Path(path) / "index.faiss"))
    with open(Path(path) / "meta.jsonl", encoding="utf-8") as f:
        meta = [json.loads(line) for line in f]
    return index, meta


def find_similar(
    text: str,
    k: int | None = None,
    cfg: dict[str, Any] | None = None,
    encode: Encoder | None = None,
) -> list[dict[str, Any]]:
    """Top-k most similar past logs: [{"text", "category", "vendor", "score"}, ...]."""
    return find_similar_batch([text], k, cfg, encode)[0]


def find_similar_batch(
    texts: list[str],
    k: int | None = None,
    cfg: dict[str, Any] | None = None,
    encode: Encoder | None = None,
) -> list[list[dict[str, Any]]]:
    """``find_similar`` for many texts at once: one embedding pass and one FAISS search."""
    cfg = cfg or load_config()
    encode = encode or load_encoder(cfg)
    index, meta = _load_index(str(index_dir(cfg)))
    scores, ids = index.search(encode(list(texts)), k or cfg["search"]["top_k"])
    return [
        [
            {**meta[i], "score": round(float(s), 4)}
            for s, i in zip(row_scores, row_ids, strict=True)
            if i != -1  # FAISS pads with -1 when k > number of stored vectors
        ]
        for row_scores, row_ids in zip(scores, ids, strict=True)
    ]


def top1_agreement(gold: list[str], neighbor: list[str], labels: list[str]) -> dict[str, Any]:
    """How often the nearest neighbor shares the query's category — overall and per category."""
    per_class = {}
    for label in labels:
        pairs = [(g, n) for g, n in zip(gold, neighbor, strict=True) if g == label]
        per_class[label] = round(sum(g == n for g, n in pairs) / len(pairs), 4) if pairs else None
    hits = sum(g == n for g, n in zip(gold, neighbor, strict=True))
    overall = round(hits / len(gold), 4) if gold else None  # None = no rows to score
    return {"n": len(gold), "top1_agreement": overall, "per_class": per_class}


def evaluate(cfg: dict[str, Any], encode: Encoder | None = None) -> dict[str, Any]:
    """Top-1 category agreement for the embedding index vs a TF-IDF nearest-neighbor baseline."""
    encode = encode or load_encoder(cfg)
    dsd = load_from_disk(cfg["data"]["processed_dir"])
    names = dsd["train"].features["label"].names
    index, meta = _load_index(str(index_dir(cfg)))
    train_labels = [m["category"] for m in meta]

    # Baseline: same nearest-neighbor idea, but with TF-IDF char n-gram vectors (Phase 2 settings).
    vectorizer = make_tfidf_vectorizer(cfg)  # same settings as the Phase 2 baseline
    train_tfidf = vectorizer.fit_transform([m["text"] for m in meta])

    report: dict[str, Any] = {
        "embedding_model": cfg["models"]["embeddings"],
        "index_size": len(meta),
    }
    for split in [s for s in EVAL_SPLITS if s in dsd]:
        texts = list(dsd[split]["text"])
        gold = [names[i] for i in dsd[split]["label"]]
        _, ids = index.search(encode(texts), 1)
        emb_nn = [train_labels[i] for i in ids[:, 0]]
        sims = (vectorizer.transform(texts) @ train_tfidf.T).toarray()
        tfidf_nn = [train_labels[i] for i in sims.argmax(axis=1)]
        report[split] = {
            "embeddings": top1_agreement(gold, emb_nn, names),
            "tfidf": top1_agreement(gold, tfidf_nn, names),
        }

    out = Path(cfg["paths"]["reports_dir"]) / "search.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"  # "-" = category absent from this split


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the similar-event index and evaluate it.")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--query", default=None, help="Only search this log line (no rebuild)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.query:
        for hit in find_similar(args.query, cfg=cfg):
            print(f"{hit['score']:.3f} [{hit['category']}] {hit['text']}")
        return

    build(cfg)
    report = evaluate(cfg)
    for split in EVAL_SPLITS:
        if split in report:
            emb, tfidf = report[split]["embeddings"], report[split]["tfidf"]
            print(f"\n[{split}] n={emb['n']}{'embeddings':>14}{'tfidf':>8}")
            print(
                f"  {'top-1 agree':<20}{emb['top1_agreement']:>10.3f}{tfidf['top1_agreement']:>8.3f}"
            )
            for label in emb["per_class"]:
                e, t = emb["per_class"][label], tfidf["per_class"][label]
                print(f"  {label:<20}{_fmt(e):>10}{_fmt(t):>8}")
    print(f"\nIndex saved to {index_dir(cfg)}; report in reports/search.json")


if __name__ == "__main__":
    main()
