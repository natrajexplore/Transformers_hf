import pytest

from logsense.baseline import run_tfidf, run_zero_shot
from logsense.config import load_config
from logsense.generate import generate_rows


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["data"]["num_samples"] = 350
    return c


def test_tfidf_learns_synthetic_data(cfg):
    rows = generate_rows(cfg)
    train, test = rows[:280], rows[280:]
    preds = run_tfidf(
        [r["text"] for r in train], [r["label"] for r in train], [r["text"] for r in test], cfg
    )
    assert len(preds) == len(test)
    assert set(preds) <= set(cfg["labels"])
    acc = sum(p == r["label"] for p, r in zip(preds, test)) / len(test)
    assert acc > 0.8


class FakeZeroShot:
    """Mimics the pipeline output: picks the phrase containing a keyword, else the last one."""

    def __call__(self, texts, candidate_labels, hypothesis_template, multi_label):
        assert "{}" in hypothesis_template and multi_label is False
        out = []
        for t in texts:
            hit = next(
                (p for p in candidate_labels if "configuration" in p and "CONFIG" in t),
                candidate_labels[-1],
            )
            out.append(
                {
                    "labels": [hit] + [p for p in candidate_labels if p != hit],
                    "scores": [1.0] + [0.0] * (len(candidate_labels) - 1),
                }
            )
        return out[0] if len(out) == 1 else out


def test_zero_shot_maps_phrases_back_to_labels(cfg):
    texts = ["%SYS-5-CONFIG_I: Configured from console", "%NTP-5-PEERSYNC: synced"] * 10
    preds = run_zero_shot(texts, cfg, clf=FakeZeroShot())
    assert len(preds) == len(texts)
    assert preds[0] == "config_change"
    assert preds[1] == cfg["labels"][-1]
