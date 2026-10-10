"""Phase 7 tests. Uses the real Qwen tokenizer (small download) but a tiny random Qwen2 model,
so LoRA training and generation are exercised end to end in seconds."""

import json
import random

import pytest
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from logsense.config import load_config
from logsense.summarize import (
    adapter_dir,
    generate_summary,
    host,
    make_burst,
    make_bursts,
    reference_summary,
    score,
    signature,
    tokenize_pair,
    train_lora,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def tokenizer(cfg):
    return AutoTokenizer.from_pretrained(cfg["summarizer"]["model"])


def test_signature_and_host():
    ios = "Oct  1 10:00:00 corp-sw-01 12: *Oct  1 10:00:00.1 UTC: %LINK-3-UPDOWN: Interface x"
    assert signature(ios) == "LINK-3-UPDOWN" and host(ios) == "corp-sw-01"
    assert (
        signature("Oct  1 10:00:00 corp-ise-02 CISE_X 1 1 0 t 2 5400 NOTICE Failed") == "ISE 5400"
    )
    assert (
        signature("Oct  1 10:00:00 corp-fw-03 1,2026/10/01,0128,THREAT,virus,") == "PAN-OS THREAT"
    )
    assert host("Oct  1 10:00:00 corp-wlc-04: *Oct  1 ...") == "corp-wlc-04"


def test_burst_facts_consistent(cfg):
    s = cfg["summarizer"]
    for b in make_bursts(cfg, 30, seed=1):
        f = b["facts"]
        assert s["burst_min"] <= f["total"] == len(b["lines"]) <= s["burst_max"]
        assert 1 <= f["count"] <= f["total"]
        assert f["label"] in s["incidents"]
        assert f["top_message"] != "unknown"
        assert sum(signature(line) == f["top_message"] for line in b["lines"]) >= 1


def test_reference_passes_its_own_checks(cfg):
    b = make_burst(random.Random(3), cfg)
    ref = reference_summary(b["facts"], cfg)
    assert len(ref.splitlines()) == 3
    assert all(score(ref, b["facts"], cfg).values())
    assert not score("Everything is fine.", b["facts"], cfg)["format_ok"]


def test_score_is_strict(cfg):
    facts = {"label": "wireless", "count": 3, "total": 9, "minutes": 3, "devices": 3,
             "top_message": "x"}  # fmt: skip
    wrong = "Incident: 5 changes approved within 3 min.\nScope: 3 devices.\nAction: none."
    checks = score(wrong, facts, cfg)
    assert not checks["count_ok"]  # 3 appears, but not as the Incident count
    assert not checks["category_ok"]  # "ap" inside "approved" must not count as wireless
    right = "Incident: 3 wireless AP events.\nScope: 3 devices.\nAction: check APs."
    assert score(right, facts, cfg) == {"format_ok": True, "count_ok": True, "category_ok": True}


def test_tokenize_pair_masks_prompt(cfg, tokenizer):
    b = make_burst(random.Random(4), cfg)
    ex = tokenize_pair(tokenizer, b["lines"], "Incident: x\nScope: y\nAction: z")
    n_answer = sum(lab != -100 for lab in ex["labels"])
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])
    assert 0 < n_answer < len(ex["labels"]) // 4  # only the short summary is learned
    first = ex["labels"].index(next(lab for lab in ex["labels"] if lab != -100))
    assert tokenizer.decode(ex["input_ids"][first:]).startswith("Incident:")
    assert ex["input_ids"][-1] == tokenizer.eos_token_id  # never truncated: EOS is kept


def test_lora_train_and_generate_tiny(tmp_path, cfg, tokenizer):
    c = json.loads(json.dumps(cfg))
    c["paths"]["models_dir"] = str(tmp_path / "models")
    c["summarizer"].update({"max_length": 4096, "use_cpu": True, "burst_min": 2, "burst_max": 3})
    tiny = AutoConfig.for_model(
        "qwen2", vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
    )  # fmt: skip
    model = AutoModelForCausalLM.from_config(tiny)
    model = train_lora(model, tokenizer, make_bursts(c, 4, seed=5), c)
    assert (adapter_dir(c) / "adapter_config.json").exists()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 0 < trainable < sum(p.numel() for p in model.parameters()) / 10
    out = generate_summary(model.eval(), tokenizer, ["%LINK-3-UPDOWN: Interface x down"], 5)
    assert isinstance(out, str)
