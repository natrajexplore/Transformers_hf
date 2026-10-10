"""Phase 6 tests. The three model helpers are stubbed, so this checks the app's own logic
(line splitting, entity offsets across lines, table shapes) without loading any model."""

import gradio as gr

from logsense import app
from logsense.config import load_config


def fake_predict(lines, path=None, cfg=None):
    return [{"label": "noise", "score": 0.91234} for _ in lines]


def fake_extract(line, path=None, cfg=None):
    i = line.index("user01")
    return [{"type": "USER", "text": "user01", "start": i, "end": i + 6, "score": 0.99}]


def fake_similar_batch(lines, k=5, cfg=None):
    hit = {"text": "past event", "category": "noise", "vendor": "cisco_ios", "score": 0.8}
    return [[hit] * k for _ in lines]


def test_analyze_offsets_across_lines(monkeypatch):
    monkeypatch.setattr(app, "predict", fake_predict)
    monkeypatch.setattr(app, "extract_entities", fake_extract)
    monkeypatch.setattr(app, "find_similar_batch", fake_similar_batch)

    text = "  login by user01  \n\nsecond line user01\n"
    cats, highlighted, similar = app.analyze(text, 2, load_config(), None, None)

    assert cats == [[1, "noise", 0.912], [2, "noise", 0.912]]
    joined = highlighted["text"]
    assert joined == "login by user01\nsecond line user01"
    for ent in highlighted["entities"]:
        assert joined[ent["start"] : ent["end"]] == "user01"
    assert len(highlighted["entities"]) == 2
    assert [row[0] for row in similar] == [1, 1, 2, 2]


def test_analyze_empty_input():
    cats, highlighted, similar = app.analyze("   \n ", 3, load_config(), None, None)
    assert cats == [] and similar == [] and highlighted["entities"] == []


def test_build_demo():
    assert isinstance(app.build_demo(load_config()), gr.Blocks)
