import ipaddress
import re
from collections import Counter

import pytest

from logsense.config import load_config
from logsense.generate import build_dataset_dict, generate_rows, render

ALLOWED_NETS = [
    ipaddress.ip_network(n)
    for n in [
        "192.0.2.0/24",
        "198.51.100.0/24",
        "203.0.113.0/24",  # RFC 5737
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",  # RFC 1918
    ]
]
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["data"]["num_samples"] = 700  # small and fast; 100 per class
    return c


@pytest.fixture(scope="module")
def rows(cfg):
    return generate_rows(cfg)


def test_row_count_and_balance(cfg, rows):
    assert len(rows) == cfg["data"]["num_samples"]
    counts = Counter(r["label"] for r in rows)
    assert set(counts) == set(cfg["labels"])
    assert max(counts.values()) - min(counts.values()) <= 1


def test_texts_unique(rows):
    assert len({r["text"] for r in rows}) == len(rows)


def test_entity_spans_valid(cfg, rows):
    for r in rows:
        for e in r["entities"]:
            assert e["type"] in cfg["entity_types"]
            assert 0 <= e["start"] < e["end"] <= len(r["text"])
            span = r["text"][e["start"] : e["end"]]
            assert span == span.strip() and span


def test_all_ips_are_safe_and_labeled(rows):
    """Every IPv4 in the text is in an allowed range and covered by an IP entity span."""
    for r in rows:
        ip_spans = {(e["start"], e["end"]) for e in r["entities"] if e["type"] == "IP"}
        for m in IPV4.finditer(r["text"]):
            assert any(ipaddress.ip_address(m.group()) in n for n in ALLOWED_NETS), r["text"]
            assert (m.start(), m.end()) in ip_spans, r["text"]


def test_render_records_spans():
    text, ents = render("a {x} b {y}", {"x": ("10.0.0.1", "IP"), "y": "plain"})
    assert text == "a 10.0.0.1 b plain"
    assert ents == [{"start": 2, "end": 10, "type": "IP"}]


def test_deterministic(cfg, rows):
    assert generate_rows(cfg) == rows


def test_hard_examples_present(rows):
    noise = [r["text"] for r in rows if r["label"] == "noise"]
    assert any("LOGIN_SUCCESS" in t or " 5200 " in t for t in noise)


def test_splits_disjoint_and_stratified(cfg, rows):
    dsd = build_dataset_dict(rows, cfg)
    assert set(dsd) == {"train", "validation", "test"}
    assert sum(len(s) for s in dsd.values()) == len(rows)
    texts = [set(s["text"]) for s in dsd.values()]
    assert not (texts[0] & texts[1] or texts[0] & texts[2] or texts[1] & texts[2])
    names = dsd["train"].features["label"].names
    assert names == cfg["labels"]
    for split in dsd.values():
        counts = Counter(split["label"])
        assert set(counts) == set(range(len(names)))  # every class in every split
        assert max(counts.values()) - min(counts.values()) <= 2
