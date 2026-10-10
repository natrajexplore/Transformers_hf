"""Phase 8: score the classifier on your own labeled logs (e.g. anonymized lab syslog).

Input is JSONL, one object per line: {"text": "<syslog line>", "label": "<category>"}.
Labels must be from configs/default.yaml. Anonymize first (RFC 5737 / RFC 1918 IPs, fake
users and hostnames) — never commit real logs.

    python -m logsense.evaluate_file --input data/raw/lab_logs.jsonl

Writes reports/real_eval.json with the same metrics as the synthetic test splits, so the two
can be compared side by side.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from logsense.config import load_config
from logsense.metrics import classification_metrics
from logsense.train_classifier import predict


def load_labeled(path: str | Path, labels: list[str]) -> tuple[list[str], list[str]]:
    texts, gold = [], []
    # utf-8-sig: also accepts files with a byte-order mark (PowerShell / Notepad add one)
    with open(path, encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                text, label = row["text"], row["label"]
            except (json.JSONDecodeError, KeyError, TypeError) as err:
                raise ValueError(f"line {n}: expected {{'text': ..., 'label': ...}}") from err
            if label not in labels:
                raise ValueError(f"line {n}: unknown label {label!r}")
            texts.append(text)
            gold.append(label)
    return texts, gold


def evaluate_file(path: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    texts, gold = load_labeled(path, cfg["labels"])
    preds = [p["label"] for p in predict(texts, cfg=cfg)]
    # Score only categories that occur in the file (as truth or prediction): a lab sample with
    # no VPN logs shouldn't drag macro-F1 down with an F1 of 0 for vpn.
    present = [lbl for lbl in cfg["labels"] if lbl in set(gold) | set(preds)]
    report = {"input": str(path), **classification_metrics(gold, preds, present)}
    out = Path(cfg["paths"]["reports_dir"]) / "real_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the classifier on labeled JSONL logs.")
    parser.add_argument("--input", required=True, help="JSONL with text + label per line")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    args = parser.parse_args(argv)
    r = evaluate_file(args.input, load_config(args.config))
    print(f"n={r['n']} accuracy={r['accuracy']:.3f} macro_f1={r['macro_f1']:.3f}")
    for label, f1 in r["per_class_f1"].items():
        print(f"  {label:<14}{f1:.3f}")


if __name__ == "__main__":
    main()
