---
title: LogSense
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# LogSense — syslog triage demo

Paste Cisco IOS / ISE / Catalyst 9800 or PAN-OS syslog lines and get a category, extracted
entities (IP, USER, INTERFACE, MAC, RULE, VLAN), and similar past events.

Trained only on **synthetic, anonymized** logs — results on real logs, especially in formats
the generator never produced, will be weaker. See the project repo for honest metrics.

## Publishing checklist (not done automatically)

1. Push the GitHub repo (the Space installs `logsense` from it).
2. Upload the models with their cards, e.g.
   `cp docs/model_cards/classifier.md models/classifier/README.md` then
   `huggingface-cli upload natrajexplore/logsense-classifier models/classifier`
   (same for `ner` → `natrajexplore/logsense-ner`).
3. Copy the config next to this file: `cp configs/default.yaml space/default.yaml`.
4. Create a Gradio Space and upload the `space/` folder:
   `huggingface-cli upload natrajexplore/logsense space --repo-type space`
