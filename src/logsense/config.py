"""Config loading. Every module reads labels, model names, and paths from here."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config file (defaults to configs/default.yaml)."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def label_maps(cfg: dict[str, Any]) -> tuple[dict[str, int], dict[int, str]]:
    """Build label2id / id2label.

    Hugging Face models predict integer class ids. Storing these two maps in the
    model config (model.config.label2id / id2label) lets a saved model return
    human-readable labels like "auth_failure" instead of "LABEL_0".
    """
    labels = cfg["labels"]
    label2id = {name: i for i, name in enumerate(labels)}
    id2label = {i: name for name, i in label2id.items()}
    return label2id, id2label
