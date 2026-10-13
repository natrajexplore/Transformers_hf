import json

import pytest
from fastapi.testclient import TestClient

from logsense import api
from logsense import evaluate_file as ef
from logsense.config import load_config

LINES = ["%LINK-3-UPDOWN: Interface Gi1/0/1, changed state to down", "%SYS-5-CONFIG_I: by user01"]


def fake_predict(texts, path=None, cfg=None):
    return [{"label": "link_event", "score": 0.9} for _ in texts]


def fake_entities(text, path=None, cfg=None):
    return [{"type": "INTERFACE", "text": "Gi1/0/1", "start": 0, "end": 7, "score": 0.99}]


def fake_similar(texts, k=None, cfg=None, encode=None):
    return [
        [{"text": "past", "category": "link_event", "vendor": "cisco_ios", "score": 0.8}]
        for _ in texts
    ]


class FakeSummarizer:
    def __init__(self, cfg):
        pass

    def summarize(self, lines, use_lora, max_new_tokens):
        return f"Incident: {len(lines)} interface up/down events.\nScope: 1 device.\nAction: check."


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c["paths"] = {**c["paths"], "reports_dir": str(tmp_path)}
    return c


@pytest.fixture
def client(cfg, monkeypatch):
    monkeypatch.setattr(api, "predict", fake_predict)
    monkeypatch.setattr(ef, "predict", fake_predict)
    monkeypatch.setattr(api, "extract_entities", fake_entities)
    monkeypatch.setattr(api, "find_similar_batch", fake_similar)
    monkeypatch.setattr(api, "Summarizer", FakeSummarizer)
    monkeypatch.setattr(
        api,
        "sample_rows",
        lambda cfg, split, n, seed=0: (LINES[:n], ["link_event", "config_change"][:n]),
    )
    return TestClient(api.create_app(cfg))


def test_analyze(client):
    r = client.post("/api/analyze", json={"lines": LINES + ["  "], "k": 1})
    assert r.status_code == 200
    body = r.json()
    assert [x["category"] for x in body["results"]] == [
        "link_event",
        "link_event",
    ]  # blank line dropped
    assert body["results"][0]["entities"][0]["type"] == "INTERFACE"
    assert set(body["timing_ms"]) == {"classifier", "ner", "search"}
    assert set(client.get("/api/health").json()["loaded"]) == {"classifier", "ner", "search"}


def test_analyze_rejects_empty_and_too_many(client, cfg):
    assert client.post("/api/analyze", json={"lines": [" "]}).status_code == 422
    many = ["x"] * (cfg["api"]["max_lines"] + 1)
    assert client.post("/api/analyze", json={"lines": many}).status_code == 422


def test_benchmark_reports_speed_and_accuracy(client):
    r = client.post(
        "/api/benchmark", json={"n_lines": 2, "batch_size": 1, "models": ["classifier", "ner"]}
    )
    assert r.status_code == 200
    rows = {x["model"]: x for x in r.json()["results"]}
    assert rows["classifier"]["batches"] == 2 and rows["classifier"]["accuracy"] == 0.5
    assert "accuracy" not in rows["ner"]
    assert rows["ner"]["batch_ms"]["p95"] >= rows["ner"]["batch_ms"]["p50"]


def test_evaluate_upload(client):
    rows = [{"text": "a", "label": "link_event"}, {"text": "b", "label": "vpn"}]
    data = "﻿" + "\n".join(json.dumps(x) for x in rows)  # with a BOM, like Notepad writes
    r = client.post("/api/evaluate", files={"file": ("lab.jsonl", data.encode("utf-8"))})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 2 and body["accuracy"] == 0.5
    assert body["mistakes"] == [{"text": "b", "label": "vpn", "predicted": "link_event"}]


def test_evaluate_bad_file(client):
    r = client.post("/api/evaluate", files={"file": ("x.jsonl", b'{"text": "a", "label": "dns"}')})
    assert r.status_code == 422 and "unknown label" in r.json()["detail"]


def test_summarize_with_facts(client):
    burst = client.get("/api/sample-burst", params={"seed": 3}).json()
    assert burst["lines"] and burst["reference"].startswith("Incident:")
    r = client.post(
        "/api/summarize", json={"lines": burst["lines"], "mode": "prompt", "facts": burst["facts"]}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["checks"]["format_ok"] is True
    assert body["classifier_counts"] == {"link_event": len(burst["lines"])}


def test_metrics_and_models(client, cfg, tmp_path):
    (tmp_path / "classifier.json").write_text(json.dumps({"test": {"macro_f1": 1.0}}))
    m = client.get("/api/metrics").json()
    assert m["classifier"]["test"]["macro_f1"] == 1.0 and m["ner"] is None
    models = client.get("/api/models").json()
    assert [x["name"] for x in models["models"]] == ["classifier", "ner", "search", "summarizer"]
    assert models["labels"] == cfg["labels"]


def test_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "LogSense" in r.text
    assert client.get("/app.js").status_code == 200


def test_config_feed_and_anonymize(client, cfg):
    c = client.get("/api/config").json()
    assert c["labels"] == cfg["labels"] and set(c["severity"]) == set(cfg["labels"])
    f = client.post("/api/feed", json={"n": 5, "t": 1_760_000_000, "seed": 1}).json()
    assert len(f["events"]) == 5 and f["t"] > 1_760_000_000
    a = client.post(
        "/api/anonymize", json={"lines": ["[user: jsmith] from 172.20.4.9", " "]}
    ).json()
    assert a["lines"] == ["[user: user01] from 10.0.0.1"]


def test_analyze_k0_skips_search(client, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("search must not run")

    monkeypatch.setattr(api, "find_similar_batch", boom)
    r = client.post("/api/analyze", json={"lines": LINES, "k": 0}).json()
    assert r["results"][0]["similar"] == [] and r["timing_ms"]["search"] == 0.0
