"""Phase 3 tests. The end-to-end test trains a *tiny* randomly initialized DistilBERT (1 layer,
32-dim) for one epoch, so it checks the whole pipeline in seconds without the real model."""

import json
from pathlib import Path

import numpy as np
import pytest
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from logsense.config import label_maps, load_config
from logsense.generate import build_dataset_dict, generate_rows
from logsense.train_classifier import compute_metrics, predict, tokenize_dataset, train


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(load_config()["models"]["classifier"])


@pytest.fixture(scope="module")
def tiny_setup(tmp_path_factory, tokenizer):
    tmp = tmp_path_factory.mktemp("clf")
    cfg = load_config()
    cfg["data"]["num_samples"] = 210
    cfg["data"]["processed_dir"] = str(tmp / "processed")
    cfg["paths"] = {"models_dir": str(tmp / "models"), "reports_dir": str(tmp / "reports")}
    cfg["training"] = {**cfg["training"], "epochs": 1, "use_cpu": True}
    build_dataset_dict(generate_rows(cfg), cfg).save_to_disk(cfg["data"]["processed_dir"])

    label2id, id2label = label_maps(cfg)
    tiny_cfg = AutoConfig.for_model(
        "distilbert", vocab_size=tokenizer.vocab_size, dim=32, hidden_dim=64, n_layers=1,
        n_heads=2, num_labels=len(label2id), label2id=label2id, id2label=id2label,
    )  # fmt: skip
    AutoModelForSequenceClassification.from_config(tiny_cfg).save_pretrained(tmp / "tiny")
    tokenizer.save_pretrained(tmp / "tiny")
    cfg["models"]["classifier"] = str(tmp / "tiny")
    return cfg


def test_tokenize_keeps_only_model_inputs(tokenizer):
    cfg = load_config()
    cfg["data"]["num_samples"] = 350  # enough rows per class to stratify after hold-outs
    dsd = build_dataset_dict(generate_rows(cfg), cfg)
    tok = tokenize_dataset(dsd, tokenizer, max_length=32)
    cols = set(tok["train"].column_names)
    assert {"label", "input_ids", "attention_mask"} <= cols
    assert not cols & {"text", "vendor", "template", "entities"}
    assert max(len(ids) for ids in tok["train"]["input_ids"]) <= 32


def test_compute_metrics():
    logits = np.array([[2.0, 0.1], [0.1, 2.0], [2.0, 0.1], [0.1, 2.0]])
    labels = np.array([0, 1, 1, 1])
    m = compute_metrics((logits, labels))
    assert m["accuracy"] == 0.75
    assert 0 < m["macro_f1"] < 1


def test_train_and_predict_end_to_end(tiny_setup):
    cfg = tiny_setup
    report = train(cfg)
    for split in ("test", "test_unseen"):
        assert set(report[split]["per_class_f1"]) == set(cfg["labels"])
    reports = Path(cfg["paths"]["reports_dir"])
    assert json.loads((reports / "classifier.json").read_text())["base_model"]
    assert (reports / "classifier_confusion_test.png").exists()

    out = predict(["%LINK-3-UPDOWN: Interface Gi1/0/1, changed state to down"], cfg=cfg)
    assert len(out) == 1
    assert out[0]["label"] in cfg["labels"]
    assert 0.0 <= out[0]["score"] <= 1.0
