"""Phase 6: Gradio demo — paste log lines, see category, entities, and similar past events.

Hugging Face concepts used here
-------------------------------
* Gradio — builds a web UI from Python functions. ``gr.Blocks`` lays out components;
  ``button.click(fn, inputs, outputs)`` wires a function to them. Hugging Face Spaces runs
  exactly this kind of app.
* ``gr.HighlightedText`` — shows text with labeled character spans, which is precisely what
  our NER helper returns (start/end/type).
* Model paths — the helpers take a local folder (``models/classifier``) or a Hub repo id
  (``natrajexplore/logsense-classifier``); ``pipeline()`` loads either, which is how the same
  code runs locally and in a Space.

Run locally:  python -m logsense.app
"""

from __future__ import annotations

import argparse
from typing import Any

import gradio as gr

from logsense.config import load_config
from logsense.search import find_similar_batch
from logsense.train_classifier import predict
from logsense.train_ner import extract_entities

EXAMPLES = [
    (
        "Oct 10 08:02:11 corp-ise-02 CISE_Failed_Attempts 0000412233 1 0 2026-10-10 08:02:11.120 "
        "+00:00 0001122334 5400 NOTICE Failed-Attempt: Authentication failed, ConfigVersionId=88, "
        "Device IP Address=10.9.8.7, Device Port=1812, UserName=user33, "
        "Calling-Station-ID=AA-BB-CC-11-22-33, NAS-Port-Id=GigabitEthernet1/0/14, "
        "FailureReason=24408 User authentication against Active Directory failed since user has "
        "entered the wrong password"
    ),
    (
        "Oct 10 09:30:02 corp-sw-03 4411: *Oct 10 09:30:02.101 UTC: %LINK-3-UPDOWN: Interface "
        "GigabitEthernet1/0/7, changed state to down"
    ),
    (
        "Oct 10 09:14:02 corp-fw-03 1,2026/10/10 09:14:02,012801055501,THREAT,vulnerability,2561,"
        "2026/10/10 09:14:02,203.0.113.7,10.20.30.44,,,inbound-web-dmz,,,web-browsing,vsys1,"
        "untrust,trust,ethernet1/1,ethernet1/2,Log-Fwd,88123,1,51544,443,0,0,0x2000,tcp,reset-both,"
        '"",Apache Log4j Remote Code Execution Vulnerability(91991),any,critical,client-to-server'
    ),
]


def analyze(
    text: str, k: int, cfg: dict[str, Any], classifier: str | None, ner: str | None
) -> tuple[list[list[Any]], dict[str, Any], list[list[Any]]]:
    """Run all three models over every non-empty line of ``text``."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [], {"text": "", "entities": []}, []

    preds = predict(lines, path=classifier, cfg=cfg)
    categories = [[i + 1, p["label"], round(p["score"], 3)] for i, p in enumerate(preds)]

    # One HighlightedText for all lines: shift each line's entity offsets by its position.
    joined, entities, offset = "\n".join(lines), [], 0
    for line in lines:
        for e in extract_entities(line, path=ner, cfg=cfg):
            entities.append(
                {"entity": e["type"], "start": e["start"] + offset, "end": e["end"] + offset}
            )
        offset += len(line) + 1  # +1 for the "\n"

    similar = [
        [i + 1, hit["score"], hit["category"], hit["text"]]
        for i, hits in enumerate(find_similar_batch(lines, k=int(k), cfg=cfg))
        for hit in hits
    ]
    return categories, {"text": joined, "entities": entities}, similar


def build_demo(
    cfg: dict[str, Any] | None = None, classifier: str | None = None, ner: str | None = None
) -> gr.Blocks:
    cfg = cfg or load_config()
    # analytics_enabled=False: Gradio otherwise sends usage telemetry to its servers
    with gr.Blocks(title="LogSense", analytics_enabled=False) as demo:
        gr.Markdown(
            "# LogSense\nSyslog triage for Cisco IOS / ISE / C9800 and PAN-OS. Paste one or "
            "more log lines (one per line).\n\n*Trained on synthetic logs — expect weaker "
            "results on formats it has not seen.*"
        )
        logs = gr.Textbox(label="Log lines", lines=6, placeholder="Paste syslog lines here")
        k = gr.Slider(1, 10, value=cfg["search"]["top_k"], step=1, label="Similar events per line")
        run = gr.Button("Analyze", variant="primary")
        categories = gr.Dataframe(
            headers=["line", "category", "confidence"], label="Category", interactive=False
        )
        entities = gr.HighlightedText(label="Entities", combine_adjacent=False)
        similar = gr.Dataframe(
            headers=["line", "score", "category", "similar past event"],
            label="Similar past events",
            interactive=False,
            wrap=True,
        )
        gr.Examples(examples=[[ex] for ex in EXAMPLES], inputs=logs)
        run.click(
            lambda text, kk: analyze(text, kk, cfg, classifier, ner),
            inputs=[logs, k],
            outputs=[categories, entities, similar],
        )
    return demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the LogSense Gradio demo.")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--share", action="store_true", help="Create a temporary public link")
    args = parser.parse_args(argv)
    build_demo(load_config(args.config)).launch(share=args.share)


if __name__ == "__main__":
    main()
