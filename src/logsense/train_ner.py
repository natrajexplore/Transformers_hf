"""Phase 4: entity extraction (NER) as token classification.

Hugging Face concepts used here
-------------------------------
* Token classification — instead of one label per log line, the model predicts one label per
  token. ``AutoModelForTokenClassification`` = DistilBERT + a linear layer applied to every
  token's hidden state.
* BIO tags — ``B-IP`` marks the first word of an IP entity, ``I-IP`` the following words,
  ``O`` anything that isn't an entity. 6 entity types -> 13 labels. The B/I split lets the
  model separate two adjacent entities of the same type.
* Words vs subwords — a fast tokenizer first splits text into "words" (on whitespace and
  punctuation), then each word into subwords from its vocabulary:
      "user32 (10.36.120.185)" -> words: user32 ( 10 . 36 . 120 . 185 )
                               -> tokens: user ##32 ( 10 . 36 . 120 . 185 )
  ``encoding.word_ids()`` tells which word each token came from (None for [CLS]/[SEP]);
  ``offset_mapping`` gives each token's (start, end) character offsets in the text. That's
  how our character spans from Phase 1 become token labels.
* The -100 trick — PyTorch's cross-entropy loss ignores positions whose label is -100.
  We give a real label only to the *first* subword of each word, and -100 to later subwords
  (``##32``) and special tokens, so every word counts exactly once in the loss and metrics.
* seqeval — scores whole entities, not tokens: an IP is correct only if all its words get the
  right B/I tags. Reports precision, recall, F1 per entity type.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from datasets import DatasetDict, load_from_disk
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    pipeline,
    set_seed,
)

from logsense.config import load_config
from logsense.train_classifier import build_training_args

EVAL_SPLITS = ("test", "test_unseen")
IGNORE = -100  # label value the loss function skips


def bio_labels(entity_types: list[str]) -> list[str]:
    """["O", "B-IP", "I-IP", "B-USER", ...] — index in this list = label id."""
    return ["O"] + [f"{p}-{t}" for t in entity_types for p in ("B", "I")]


def ner_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Config with `training` overridden by the `ner` section, for build_training_args."""
    return {**cfg, "training": {**cfg["training"], **cfg.get("ner", {})}}


def model_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["models_dir"]) / "ner"


def align_labels(
    entities: list[dict[str, Any]],
    word_ids: list[int | None],
    offsets: list[tuple[int, int]],
    label2id: dict[str, int],
) -> list[int]:
    """Turn character spans into one label id per token.

    The first token of each word gets B-/I-/O from the entity covering that word's start
    character; every other token (special tokens, later subwords) gets -100.
    """
    labels: list[int] = []
    prev_word = None
    for word, (start, _end) in zip(word_ids, offsets, strict=True):
        if word is None or word == prev_word:
            labels.append(IGNORE)
        else:
            tag = "O"
            for ent in entities:
                if ent["start"] <= start < ent["end"]:
                    prefix = "B" if start == ent["start"] else "I"
                    tag = f"{prefix}-{ent['type']}"
                    break
            labels.append(label2id[tag])
        prev_word = word
    return labels


def tokenize_and_align(
    dsd: DatasetDict, tokenizer: Any, max_length: int, label2id: dict[str, int]
) -> DatasetDict:
    """Tokenize every split and attach aligned BIO ``labels``; drop all other columns."""

    def _fn(batch: dict[str, list]) -> dict[str, list]:
        enc = tokenizer(
            batch["text"], truncation=True, max_length=max_length, return_offsets_mapping=True
        )
        enc["labels"] = [
            align_labels(ents, enc.word_ids(i), enc["offset_mapping"][i], label2id)
            for i, ents in enumerate(batch["entities"])
        ]
        enc.pop("offset_mapping")  # the model doesn't take offsets as input
        return enc

    return dsd.map(_fn, batched=True, remove_columns=dsd["train"].column_names)


def to_tag_sequences(
    predictions: np.ndarray, label_ids: np.ndarray, id2label: dict[int, str]
) -> tuple[list[list[str]], list[list[str]]]:
    """Drop -100 positions and convert ids to tag strings, the format seqeval expects."""
    y_true, y_pred = [], []
    for pred_row, gold_row in zip(predictions, label_ids, strict=True):
        keep = gold_row != IGNORE
        y_true.append([id2label[int(i)] for i in gold_row[keep]])
        y_pred.append([id2label[int(i)] for i in pred_row[keep]])
    return y_true, y_pred


def make_compute_metrics(id2label: dict[int, str]):
    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, label_ids = eval_pred
        y_true, y_pred = to_tag_sequences(np.argmax(logits, axis=-1), label_ids, id2label)
        return {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),  # micro-averaged over all entities
        }

    return compute_metrics


def entity_report(y_true: list[list[str]], y_pred: list[list[str]]) -> dict[str, Any]:
    """Per-entity-type precision/recall/F1/support plus micro and macro averages."""
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        name: {k: (int(v) if k == "support" else round(float(v), 4)) for k, v in m.items()}
        for name, m in rep.items()
        if name != "weighted avg"
    }


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fine-tune the NER model, save the best to models/ner/, evaluate on the test splits."""
    set_seed(cfg["seed"])
    tags = bio_labels(cfg["entity_types"])
    label2id = {t: i for i, t in enumerate(tags)}
    id2label = dict(enumerate(tags))
    base = cfg["models"]["ner"]
    ncfg = ner_cfg(cfg)

    dsd = load_from_disk(cfg["data"]["processed_dir"])
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenized = tokenize_and_align(dsd, tokenizer, ncfg["training"]["max_length"], label2id)

    model = AutoModelForTokenClassification.from_pretrained(
        base, num_labels=len(tags), id2label=id2label, label2id=label2id
    )

    eval_sets = {s: tokenized[s] for s in ("validation", "validation_unseen") if s in tokenized}
    best_metric = "validation_unseen_f1" if "validation_unseen" in eval_sets else "validation_f1"

    out_dir = model_dir(cfg)
    trainer = Trainer(
        model=model,
        args=build_training_args(ncfg, out_dir.parent / "ner_checkpoints", best_metric),
        train_dataset=tokenized["train"],
        eval_dataset=eval_sets,
        processing_class=tokenizer,
        # Pads input_ids with the pad token AND the labels with -100, so padding is ignored.
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=make_compute_metrics(id2label),
        callbacks=[EarlyStoppingCallback(cfg["training"]["early_stopping_patience"])],
    )

    t0 = time.perf_counter()
    train_out = trainer.train()
    train_seconds = round(time.perf_counter() - t0, 1)
    trainer.save_model(str(out_dir))

    report: dict[str, Any] = {
        "base_model": base,
        "max_length": ncfg["training"]["max_length"],
        "train_rows": len(tokenized["train"]),
        "epochs_run": round(trainer.state.epoch or 0, 2),
        "best_metric": best_metric,
        "best_metric_value": trainer.state.best_metric,
        "train_seconds": train_seconds,
        "train_loss": round(train_out.training_loss, 4),
    }
    for split in [s for s in EVAL_SPLITS if s in tokenized]:
        pred = trainer.predict(tokenized[split])
        y_true, y_pred = to_tag_sequences(pred.predictions.argmax(-1), pred.label_ids, id2label)
        report[split] = entity_report(y_true, y_pred)

    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "ner.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


@lru_cache(maxsize=4)
def _load_pipeline(path: str) -> Any:
    # aggregation_strategy="first": each word takes the label of its first subword (matching
    # how we trained), then consecutive B-X / I-X words are merged into one entity group.
    return pipeline("token-classification", model=path, aggregation_strategy="first")


def extract_entities(
    text: str, path: str | Path | None = None, cfg: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Return [{"type", "text", "start", "end", "score"}, ...] for one log line."""
    cfg = cfg or load_config()
    ner = _load_pipeline(str(path or model_dir(cfg)))
    return [
        {
            "type": g["entity_group"],
            "text": text[g["start"] : g["end"]],  # original casing (the tokenizer lowercases)
            "start": int(g["start"]),
            "end": int(g["end"]),
            "score": round(float(g["score"]), 4),
        }
        for g in ner(text)
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the LogSense NER model.")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    args = parser.parse_args(argv)

    report = train(load_config(args.config))
    print(json.dumps({k: v for k, v in report.items() if k not in EVAL_SPLITS}, indent=2))
    for split in EVAL_SPLITS:
        if split in report:
            print(f"\n[{split}] {'precision':>10}{'recall':>8}{'f1':>8}{'support':>9}")
            for name, m in report[split].items():
                print(
                    f"  {name:<12}{m['precision']:>8.3f}{m['recall']:>8.3f}"
                    f"{m['f1-score']:>8.3f}{m['support']:>9}"
                )


if __name__ == "__main__":
    main()
