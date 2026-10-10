"""Phase 3: fine-tune a pretrained encoder (DistilBERT) to classify log lines.

Hugging Face concepts used here
-------------------------------
* ``AutoTokenizer`` — turns text into subword token ids from the model's vocabulary
  ("GigabitEthernet1/0/3" -> ["gig", "##abi", "##tet", ...]). It also returns an
  ``attention_mask``: 1 for real tokens, 0 for padding, so the model ignores the padding.
* ``AutoModelForSequenceClassification`` — the pretrained encoder plus a fresh, randomly
  initialized classification head (a linear layer -> one logit per label). Fine-tuning trains
  the head and nudges the pretrained weights toward our task.
* ``label2id`` / ``id2label`` — stored in ``model.config`` so the saved model outputs
  "threat" rather than "LABEL_3".
* ``TrainingArguments`` + ``Trainer`` — the training loop (batching, AdamW optimizer, LR schedule,
  evaluation, checkpointing, restoring the best checkpoint) so we don't write it by hand.
* ``DataCollatorWithPadding`` — pads each batch only to its own longest sequence (dynamic
  padding), much cheaper on CPU than padding everything to max_length.
* ``evaluate`` — standard metric implementations loaded by name from the Hub.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    pipeline,
    set_seed,
)

from logsense.config import label_maps, load_config
from logsense.metrics import classification_metrics

EVAL_SPLITS = ("test", "test_unseen")


def model_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["models_dir"]) / "classifier"


def tokenize_dataset(dsd: DatasetDict, tokenizer: Any, max_length: int) -> DatasetDict:
    """Add ``input_ids`` / ``attention_mask`` and drop every column the model can't consume.

    No padding here — the data collator pads per batch. ``truncation=True`` cuts anything
    longer than ``max_length`` tokens (the category signal sits early in each line).
    """
    keep = {"label"}  # Trainer renames "label" -> "labels", the name the model's loss expects
    drop = [c for c in dsd["train"].column_names if c not in keep]
    return dsd.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=drop,
    )


@lru_cache(maxsize=1)
def _metrics() -> tuple[Any, Any]:
    # Loaded on first use, not at import: importing this module for predict() needs no Hub I/O.
    return evaluate.load("accuracy"), evaluate.load("f1")


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Called by Trainer after each evaluation. ``eval_pred`` = (logits, label ids)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)  # highest logit = predicted class
    accuracy, f1 = _metrics()
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        # average over *every* class (absent ones score 0), matching metrics.classification_metrics
        "macro_f1": f1.compute(
            predictions=preds,
            references=labels,
            average="macro",
            labels=list(range(logits.shape[-1])),
        )["f1"],
    }


def build_training_args(
    cfg: dict[str, Any], output_dir: Path, best_metric: str = "macro_f1"
) -> TrainingArguments:
    t = cfg["training"]
    return TrainingArguments(
        output_dir=str(output_dir),  # where checkpoints are written during training
        num_train_epochs=t["epochs"],  # full passes over the training set (upper bound here)
        per_device_train_batch_size=t["batch_size"],  # examples per optimizer step
        per_device_eval_batch_size=t["batch_size"] * 2,  # no gradients at eval -> can be larger
        learning_rate=t[
            "learning_rate"
        ],  # small (2e-5): we adjust, not overwrite, pretrained weights
        weight_decay=t["weight_decay"],  # L2-style penalty on weights; mild regularization
        warmup_steps=t["warmup_ratio"],  # float < 1 = fraction of steps spent ramping LR up from 0
        lr_scheduler_type=t["lr_scheduler"],  # "linear": after warmup, LR decays to 0
        eval_strategy="epoch",  # evaluate on the validation split after every epoch...
        save_strategy="epoch",  # ...and checkpoint at the same points (must match for best-model)
        load_best_model_at_end=True,  # restore the checkpoint with the best validation metric
        # "best" = highest value of this compute_metrics key. With several eval sets, keys are
        # prefixed by set name, e.g. "validation_unseen_macro_f1". Early stopping watches it too.
        metric_for_best_model=best_metric,
        greater_is_better=True,
        save_total_limit=t["save_total_limit"],  # keep disk use bounded (best + latest)
        logging_strategy="steps",
        logging_steps=t["logging_steps"],  # print training loss every N steps
        seed=cfg["seed"],  # model head init + data shuffling are reproducible
        report_to="none",  # no wandb/tensorboard
        use_cpu=t.get("use_cpu", False),  # Trainer picks a GPU automatically if one exists
        dataloader_num_workers=0,  # Windows-safe; data is tiny anyway
    )


def save_confusion_png(metrics: dict[str, Any], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # render to file, no display needed
    import matplotlib.pyplot as plt

    cm = np.array(metrics["confusion_matrix"]["matrix"])
    labels = metrics["confusion_matrix"]["labels"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fine-tune, save the best model to models/classifier/, and evaluate on the test splits."""
    set_seed(cfg["seed"])
    labels: list[str] = cfg["labels"]
    label2id, id2label = label_maps(cfg)
    base = cfg["models"]["classifier"]

    dsd = load_from_disk(cfg["data"]["processed_dir"])
    if dsd["train"].features["label"].names != labels:
        # label ids in the data come from generation time; a changed config would mislabel them
        raise ValueError("configs labels differ from the saved dataset; re-run logsense.generate")
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenized = tokenize_dataset(dsd, tokenizer, cfg["training"]["max_length"])

    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    # Select checkpoints on templates the model never trained on, when that split exists;
    # in-distribution validation scores 1.0 from epoch 1 and can't tell checkpoints apart.
    eval_sets = {s: tokenized[s] for s in ("validation", "validation_unseen") if s in tokenized}
    best_metric = (
        "validation_unseen_macro_f1" if "validation_unseen" in eval_sets else "validation_macro_f1"
    )

    out_dir = model_dir(cfg)
    trainer = Trainer(
        model=model,
        args=build_training_args(cfg, out_dir.parent / "classifier_checkpoints", best_metric),
        train_dataset=tokenized["train"],
        eval_dataset=eval_sets,  # a dict -> Trainer evaluates each and prefixes metric names
        processing_class=tokenizer,  # saved alongside the model so predict() can reload it
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        # Stop if validation macro-F1 hasn't improved for `patience` evaluations (epochs).
        callbacks=[EarlyStoppingCallback(cfg["training"]["early_stopping_patience"])],
    )

    t0 = time.perf_counter()
    train_out = trainer.train()
    train_seconds = round(time.perf_counter() - t0, 1)
    trainer.save_model(str(out_dir))  # best model (load_best_model_at_end) + tokenizer

    report: dict[str, Any] = {
        "base_model": base,
        "max_length": cfg["training"]["max_length"],
        "train_rows": len(tokenized["train"]),
        "epochs_run": round(trainer.state.epoch or 0, 2),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": best_metric,
        "best_metric_value": trainer.state.best_metric,
        "train_seconds": train_seconds,
        "train_loss": round(train_out.training_loss, 4),
    }
    reports_dir = Path(cfg["paths"]["reports_dir"])
    for split in [s for s in EVAL_SPLITS if s in tokenized]:
        pred = trainer.predict(tokenized[split])
        y_pred = [id2label[i] for i in pred.predictions.argmax(-1)]
        y_true = [id2label[i] for i in pred.label_ids]
        metrics = classification_metrics(y_true, y_pred, labels)
        report[split] = metrics
        save_confusion_png(
            metrics, f"{base} — {split}", reports_dir / f"classifier_confusion_{split}.png"
        )

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "classifier.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


@lru_cache(maxsize=4)
def _load_pipeline(path: str) -> Any:
    # "text-classification" pipeline = tokenizer + model + softmax over the logits.
    return pipeline("text-classification", model=path, tokenizer=path)


def predict(
    texts: list[str], path: str | Path | None = None, cfg: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Classify log lines with the saved model. Returns [{"label": ..., "score": ...}, ...].

    ``score`` is the softmax probability of the predicted label (the model's confidence).
    """
    cfg = cfg or load_config()
    clf = _load_pipeline(str(path or model_dir(cfg)))
    outputs = clf(list(texts), truncation=True, max_length=cfg["training"]["max_length"])
    return [{"label": o["label"], "score": float(o["score"])} for o in outputs]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the LogSense classifier.")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    args = parser.parse_args(argv)

    report = train(load_config(args.config))
    print(json.dumps({k: v for k, v in report.items() if k not in EVAL_SPLITS}, indent=2))
    for split in EVAL_SPLITS:
        if split in report:
            m = report[split]
            print(
                f"\n[{split}] n={m['n']} accuracy={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f}"
            )
            for label, f in m["per_class_f1"].items():
                print(f"  {label:<14}{f:.3f}")


if __name__ == "__main__":
    main()
