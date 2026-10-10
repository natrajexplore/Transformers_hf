"""Phase 8 tests: the MCP server lists the expected tools, describes them from config, and
the tools return the right shapes through a real MCP call. Model helpers are stubbed."""

import asyncio
import json

from logsense import serve
from logsense.config import load_config


def test_tools_registered_from_config():
    cfg = load_config()
    tools = {t.name: t for t in asyncio.run(serve.build_server(cfg).list_tools())}
    assert set(tools) == {"classify_logs", "extract_log_entities", "find_similar_events", "triage"}
    assert all(label in tools["classify_logs"].description for label in cfg["labels"])
    assert f"k={cfg['search']['top_k']}" in tools["find_similar_events"].description


def test_triage_combines_tools(monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(serve, "predict", lambda lines, cfg=None: [{"label": "vpn", "score": 0.9}])
    monkeypatch.setattr(
        serve, "extract_entities", lambda line, cfg=None: [{"type": "IP", "text": "10.0.0.1"}]
    )
    monkeypatch.setattr(
        serve, "find_similar", lambda line, k=5, cfg=None: [{"category": "vpn", "score": 1}] * k
    )
    server = serve.build_server(cfg)
    result = asyncio.run(server.call_tool("triage", {"line": "Crypto tunnel is UP. Peer 10.0.0.1"}))
    out = json.loads(result.content[0].text)
    assert out["category"] == "vpn" and out["confidence"] == 0.9
    assert out["entities"][0]["type"] == "IP"
    assert len(out["similar"]) == cfg["search"]["top_k"]
