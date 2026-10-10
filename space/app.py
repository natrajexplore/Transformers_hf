"""Hugging Face Space entry point for LogSense.

Differences from running locally (python -m logsense.app):
* models load from the Hub repos named in the config's `hub` section, not from models/;
* the FAISS index is not on the Hub, so it is rebuilt at startup: the generator is seeded,
  so it reproduces the exact training split, and embedding ~1.5k lines takes about a minute.
"""

from pathlib import Path

from logsense.app import build_demo
from logsense.config import load_config
from logsense.generate import build_dataset_dict, generate_rows
from logsense.search import build, index_dir

cfg = load_config(Path(__file__).with_name("default.yaml"))  # copied in at upload time
hub = cfg["hub"]

if not (index_dir(cfg) / "index.faiss").exists():
    build_dataset_dict(generate_rows(cfg), cfg).save_to_disk(cfg["data"]["processed_dir"])
    build(cfg)

build_demo(
    cfg,
    classifier=f"{hub['username']}/{hub['classifier_repo']}",
    ner=f"{hub['username']}/{hub['ner_repo']}",
).launch()
