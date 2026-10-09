"""Phase 2: two baselines the fine-tuned model has to beat.

(a) Zero-shot classification with an NLI model (no training on our data at all)
(b) TF-IDF character n-grams + logistic regression (classic ML, trained on our train split)

How zero-shot NLI classification works
--------------------------------------
An NLI (natural language inference) model like ``facebook/bart-large-mnli`` was trained on
pairs of sentences — a *premise* and a *hypothesis* — to predict whether the premise
entails, contradicts, or is neutral toward the hypothesis.

``pipeline("zero-shot-classification")`` turns classification into NLI:
  premise    = the log line
  hypothesis = hypothesis_template.format(label_phrase), e.g.
               "This network log message is about a device configuration change."
It runs one forward pass per (log, label) pair — 7 passes per log here — and turns the
entailment scores across labels into a softmax distribution. The top label wins.

Because the model only "reads" the label phrase as English, the wording matters a lot
(that's why configs/default.yaml maps ``link_event`` to a descriptive phrase). And it has
never seen Cisco/PAN-OS message codes, so expect it to struggle on terse syslog.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from logsense.config import load_config
from logsense.metrics import classification_metrics


def run_zero_shot(
    texts: list[str], cfg: dict[str, Any], clf: Callable[..., Any] | None = None
) -> list[str]:
    """Predict a label for each text with zero-shot NLI. Returns label names.

    ``clf`` defaults to a Hugging Face pipeline; tests pass a fake to avoid the download.
    """
    if clf is None:
        from transformers import pipeline

        # pipeline() bundles three steps: tokenizer (text -> token ids), model (ids -> logits),
        # and task-specific post-processing (logits -> {"labels": [...], "scores": [...]}).
        clf = pipeline("zero-shot-classification", model=cfg["models"]["zero_shot"])

    phrases: dict[str, str] = cfg["baseline"]["zero_shot_labels"]
    phrase_to_label = {phrase: label for label, phrase in phrases.items()}
    batch = cfg["training"]["batch_size"]

    preds: list[str] = []
    for i in range(0, len(texts), batch):
        outputs = clf(
            texts[i : i + batch],
            candidate_labels=list(phrases.values()),
            hypothesis_template=cfg["baseline"]["hypothesis_template"],
            multi_label=False,  # exactly one category per log -> softmax across labels
        )
        if isinstance(outputs, dict):  # a single input returns a dict, not a list
            outputs = [outputs]
        # Each output's "labels" are sorted by score, highest first.
        preds.extend(phrase_to_label[out["labels"][0]] for out in outputs)
        print(f"  zero-shot {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    return preds


def run_tfidf(
    train_texts: list[str], train_labels: list[str], test_texts: list[str], cfg: dict[str, Any]
) -> list[str]:
    """Fit TF-IDF + logistic regression on train, predict test. Returns label names."""
    tf = cfg["baseline"]["tfidf"]
    model = make_pipeline(
        TfidfVectorizer(
            analyzer=tf["analyzer"],
            ngram_range=(tf["ngram_min"], tf["ngram_max"]),
            max_features=tf["max_features"],
            lowercase=True,
        ),
        LogisticRegression(C=cfg["baseline"]["logreg_C"], max_iter=2000, random_state=cfg["seed"]),
    )
    model.fit(train_texts, train_labels)
    return list(model.predict(test_texts))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot and TF-IDF baselines.")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument(
        "--zero-shot-limit", type=int, default=None, help="Only score the first N test logs"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    labels: list[str] = cfg["labels"]
    dsd = load_from_disk(cfg["data"]["processed_dir"])
    names = dsd["train"].features["label"].names

    train_texts, train_labels = dsd["train"]["text"], [names[i] for i in dsd["train"]["label"]]
    test_texts, test_labels = dsd["test"]["text"], [names[i] for i in dsd["test"]["label"]]

    report: dict[str, Any] = {"split": "test"}

    t0 = time.perf_counter()
    tfidf_preds = run_tfidf(train_texts, train_labels, test_texts, cfg)
    report["tfidf_logreg"] = {
        "seconds": round(time.perf_counter() - t0, 1),
        **classification_metrics(test_labels, tfidf_preds, labels),
    }

    n = args.zero_shot_limit or len(test_texts)
    t0 = time.perf_counter()
    zs_preds = run_zero_shot(test_texts[:n], cfg)
    report["zero_shot"] = {
        "model": cfg["models"]["zero_shot"],
        "seconds": round(time.perf_counter() - t0, 1),
        **classification_metrics(test_labels[:n], zs_preds, labels),
    }

    out = Path(cfg["paths"]["reports_dir"]) / "baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'':<14}{'zero_shot':>10}{'tfidf':>10}")
    zs, tf = report["zero_shot"], report["tfidf_logreg"]
    print(f"{'n':<14}{zs['n']:>10}{tf['n']:>10}")
    for key in ("accuracy", "macro_f1"):
        print(f"{key:<14}{zs[key]:>10.3f}{tf[key]:>10.3f}")
    for label in labels:
        zf, tff = zs["per_class_f1"][label], tf["per_class_f1"][label]
        print(f"  {label:<12}{zf:>10.3f}{tff:>10.3f}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
