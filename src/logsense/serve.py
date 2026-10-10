"""Phase 8: expose LogSense to AI agents as MCP tools.

MCP (Model Context Protocol) is a standard way for an AI agent — Claude Code, a CrewAI crew,
an n8n AI-agent node — to discover and call tools. Each ``@server.tool()`` function below
becomes a tool whose name, description, and type hints tell the agent what it does and how to
call it. The models load lazily on the first call and then stay cached.

Run over stdio (how most MCP clients launch a local server):
    python -m logsense.serve
or over HTTP for remote agents:
    python -m logsense.serve --transport streamable-http
Example Claude Code registration:
    claude mcp add logsense -- .venv/Scripts/python -m logsense.serve
"""

from __future__ import annotations

import argparse
from typing import Any

from mcp.server.mcpserver import MCPServer

from logsense.config import load_config
from logsense.search import find_similar
from logsense.train_classifier import predict
from logsense.train_ner import extract_entities


def build_server(cfg: dict[str, Any]) -> MCPServer:
    """Create the MCP server; labels and the default k come from ``cfg``, read once."""
    labels = ", ".join(cfg["labels"])
    top_k = cfg["search"]["top_k"]
    server = MCPServer(
        "logsense",
        instructions=(
            "Triage tools for Cisco IOS/ISE/C9800 and PAN-OS syslog lines. Trained on synthetic "
            "logs: treat low-confidence results on unfamiliar formats with caution."
        ),
    )

    @server.tool(
        description=f"Classify syslog lines into one of: {labels}. "
        'Returns one {"line", "category", "confidence"} per input line.'
    )
    def classify_logs(lines: list[str]) -> list[dict[str, Any]]:
        preds = predict(lines, cfg=cfg)
        return [
            {"line": line, "category": p["label"], "confidence": round(p["score"], 4)}
            for line, p in zip(lines, preds, strict=True)
        ]

    @server.tool(
        description=f"Extract {', '.join(cfg['entity_types'])} entities from one syslog line. "
        'Returns [{"type", "text", "start", "end", "score"}].'
    )
    def extract_log_entities(line: str) -> list[dict[str, Any]]:
        return extract_entities(line, cfg=cfg)

    @server.tool(
        description="Find the k most similar past log events (cosine similarity of sentence "
        f'embeddings), default k={top_k}. Returns [{{"text", "category", "vendor", "score"}}].'
    )
    def find_similar_events(line: str, k: int = top_k) -> list[dict[str, Any]]:
        return find_similar(line, k=k, cfg=cfg)

    @server.tool(
        description="One-call triage of a single syslog line: category, entities, and the "
        f"top-{top_k} similar past events."
    )
    def triage(line: str) -> dict[str, Any]:
        return {
            **classify_logs([line])[0],
            "entities": extract_log_entities(line),
            "similar": find_similar_events(line),
        }

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run LogSense as an MCP server.")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    args = parser.parse_args(argv)
    build_server(load_config(args.config)).run(transport=args.transport)


if __name__ == "__main__":
    main()
