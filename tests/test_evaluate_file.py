import json

import pytest

from logsense import evaluate_file as ef
from logsense.config import load_config


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")


def test_evaluate_file(tmp_path, monkeypatch):
    cfg = load_config()
    cfg["paths"] = {**cfg["paths"], "reports_dir": str(tmp_path)}
    data = tmp_path / "lab.jsonl"
    write_jsonl(data, [{"text": "a", "label": "vpn"}, {"text": "b", "label": "noise"}])
    monkeypatch.setattr(ef, "predict", lambda texts, cfg=None: [{"label": "vpn"} for _ in texts])

    report = ef.evaluate_file(data, cfg)
    assert report["n"] == 2 and report["accuracy"] == 0.5
    assert set(report["per_class_f1"]) == {"vpn", "noise"}  # absent labels aren't scored
    assert json.loads((tmp_path / "real_eval.json").read_text())["n"] == 2


def test_unknown_label_rejected(tmp_path):
    data = tmp_path / "bad.jsonl"
    write_jsonl(data, [{"text": "a", "label": "dns"}])
    with pytest.raises(ValueError, match="unknown label 'dns'"):
        ef.load_labeled(data, load_config()["labels"])


def test_bom_and_bad_rows(tmp_path):
    labels = load_config()["labels"]
    bom = tmp_path / "bom.jsonl"
    bom.write_text(json.dumps({"text": "a", "label": "vpn"}) + "\n", encoding="utf-8-sig")
    assert ef.load_labeled(bom, labels) == (["a"], ["vpn"])
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"text": "a", "label": "vpn"}) + '\n{"text": "b"}\n')
    with pytest.raises(ValueError, match="line 2"):
        ef.load_labeled(bad, labels)
