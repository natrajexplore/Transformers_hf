"""Phase 4 tests. The end-to-end test trains a tiny random DistilBERT for one epoch."""

import json
from pathlib import Path

import pytest
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from logsense.config import load_config
from logsense.generate import build_dataset_dict, generate_rows, render
from logsense.train_ner import (
    align_labels,
    bio_labels,
    entity_report,
    extract_entities,
    tokenize_and_align,
    train,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def tokenizer(cfg):
    return AutoTokenizer.from_pretrained(cfg["models"]["ner"])


@pytest.fixture(scope="module")
def label2id(cfg):
    return {t: i for i, t in enumerate(bio_labels(cfg["entity_types"]))}


def test_bio_labels(cfg):
    tags = bio_labels(cfg["entity_types"])
    assert tags[0] == "O"
    assert len(tags) == 1 + 2 * len(cfg["entity_types"])
    assert {"B-IP", "I-IP", "B-VLAN", "I-VLAN"} <= set(tags)


def test_align_hand_checked(tokenizer, label2id):
    text, ents = render(
        "by {user} on vty4 ({ip})", {"user": ("user32", "USER"), "ip": ("10.36.120.185", "IP")}
    )
    enc = tokenizer(text, return_offsets_mapping=True)
    labels = align_labels(ents, enc.word_ids(), enc["offset_mapping"], label2id)
    id2label = {i: t for t, i in label2id.items()}
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    got = [(tok, id2label.get(lab, "-100")) for tok, lab in zip(tokens, labels, strict=True)]
    assert got[0] == ("[CLS]", "-100") and got[-1] == ("[SEP]", "-100")
    assert ("user", "B-USER") in got
    assert ("##32", "-100") in got  # later subword of "user32" is ignored
    ip_tags = [tag for tok, tag in got if tok in {"10", ".", "36", "120", "185"}]
    assert ip_tags == ["B-IP"] + ["I-IP"] * 6  # 10 . 36 . 120 . 185 = 7 words, one entity
    assert got[got.index(("(", "O"))][1] == "O"


def test_every_entity_gets_exactly_one_b_tag(cfg, tokenizer, label2id):
    """Across real generated data: no span lost or split by the char -> token alignment."""
    c = {**cfg, "data": {**cfg["data"], "num_samples": 350}}
    dsd = build_dataset_dict(generate_rows(c), c)
    tok = tokenize_and_align(dsd, tokenizer, max_length=256, label2id=label2id)
    b_ids = {i for t, i in label2id.items() if t.startswith("B-")}
    for split in dsd:
        for ents, labels in zip(dsd[split]["entities"], tok[split]["labels"], strict=True):
            assert sum(lab in b_ids for lab in labels) == len(ents)


def test_entity_report_per_type():
    y_true = [["B-IP", "I-IP", "O", "B-USER"]]
    y_pred = [["B-IP", "I-IP", "O", "O"]]
    rep = entity_report(y_true, y_pred)
    assert rep["IP"]["f1-score"] == 1.0
    assert rep["USER"]["recall"] == 0.0
    assert rep["micro avg"]["support"] == 2


@pytest.fixture(scope="module")
def tiny_setup(tmp_path_factory, tokenizer, cfg, label2id):
    tmp = tmp_path_factory.mktemp("ner")
    c = json.loads(json.dumps(cfg))  # deep copy
    c["data"]["num_samples"] = 210
    c["data"]["processed_dir"] = str(tmp / "processed")
    c["paths"] = {"models_dir": str(tmp / "models"), "reports_dir": str(tmp / "reports")}
    c["ner"] = {**c["ner"], "epochs": 1, "use_cpu": True}
    build_dataset_dict(generate_rows(c), c).save_to_disk(c["data"]["processed_dir"])

    id2label = {i: t for t, i in label2id.items()}
    tiny_cfg = AutoConfig.for_model(
        "distilbert", vocab_size=tokenizer.vocab_size, dim=32, hidden_dim=64, n_layers=1,
        n_heads=2, num_labels=len(label2id), label2id=label2id, id2label=id2label,
    )  # fmt: skip
    AutoModelForTokenClassification.from_config(tiny_cfg).save_pretrained(tmp / "tiny")
    tokenizer.save_pretrained(tmp / "tiny")
    c["models"]["ner"] = str(tmp / "tiny")
    return c


def test_train_and_extract_end_to_end(tiny_setup):
    c = tiny_setup
    report = train(c)
    assert "micro avg" in report["test"] and "micro avg" in report["test_unseen"]
    assert json.loads((Path(c["paths"]["reports_dir"]) / "ner.json").read_text())["base_model"]

    text = "Login failed [user: user07] [Source: 10.1.2.3]"
    ents = extract_entities(text, cfg=c)
    for e in ents:  # an untrained tiny model may find nothing; shape must still be right
        assert e["type"] in c["entity_types"]
        assert text[e["start"] : e["end"]] == e["text"]
        assert 0.0 <= e["score"] <= 1.0
