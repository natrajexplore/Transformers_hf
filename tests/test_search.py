"""Phase 5 tests. A deterministic fake encoder (hashed character trigrams) stands in for
MiniLM, so these run offline in a second."""

import json
import zlib

import numpy as np
import pytest

from logsense.config import load_config
from logsense.generate import build_dataset_dict, generate_rows
from logsense.search import build, evaluate, find_similar, top1_agreement


def fake_encode(texts, dim=256):
    vecs = np.zeros((len(texts), dim), dtype="float32")
    for row, text in enumerate(texts):
        for i in range(len(text) - 2):
            vecs[row, zlib.crc32(text[i : i + 3].encode()) % dim] += 1
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("search")
    c = json.loads(json.dumps(load_config()))
    c["data"]["num_samples"] = 210
    c["data"]["processed_dir"] = str(tmp / "processed")
    c["paths"] = {"models_dir": str(tmp / "models"), "reports_dir": str(tmp / "reports")}
    build_dataset_dict(generate_rows(c), c).save_to_disk(c["data"]["processed_dir"])
    build(c, encode=fake_encode)
    return c


def test_indexed_line_finds_itself(cfg):
    from datasets import load_from_disk

    ex = load_from_disk(cfg["data"]["processed_dir"])["train"][0]
    hits = find_similar(ex["text"], k=3, cfg=cfg, encode=fake_encode)
    assert len(hits) == 3
    assert hits[0]["text"] == ex["text"]
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert hits[0]["category"] in cfg["labels"]
    assert hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]


def test_k_larger_than_index(cfg):
    hits = find_similar("anything", k=10_000, cfg=cfg, encode=fake_encode)
    assert 0 < len(hits) < 10_000  # FAISS -1 padding is dropped


def test_top1_agreement():
    m = top1_agreement(["a", "a", "b"], ["a", "b", "b"], ["a", "b", "c"])
    assert m["top1_agreement"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["per_class"] == {"a": 0.5, "b": 1.0, "c": None}


def test_evaluate_report(cfg):
    report = evaluate(cfg, encode=fake_encode)
    for split in ("test", "test_unseen"):
        for method in ("embeddings", "tfidf"):
            assert 0.0 <= report[split][method]["top1_agreement"] <= 1.0
