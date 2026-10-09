# LogSense — Project Memory for Claude Code

Transformer-based syslog triage for Cisco IOS/IOS-XE, Cisco ISE, Cisco Catalyst 9800 WLC,
and Palo Alto PAN-OS logs. Built with the Hugging Face stack as a hands-on learning project.

## What LogSense does
1. **Classify** a log line into a category (fine-tuned encoder model)
2. **Extract entities** — IP, USER, INTERFACE, MAC, RULE, VLAN (token classification / NER)
3. **Find similar past events** (sentence embeddings + FAISS)
4. **Summarize** a burst of logs in plain English (stretch goal — small instruct LLM + LoRA)

## Categories (single source of truth: configs/default.yaml)
auth_failure, config_change, link_event, threat, vpn, wireless, noise

## Stack
- Python 3.11+
- transformers, datasets, evaluate, accelerate
- sentence-transformers, faiss-cpu
- scikit-learn (metrics, baselines)
- gradio (demo UI), huggingface_hub (push to Hub / Spaces)
- pytest, ruff

## Project layout
```
configs/            YAML configs (labels, model names, hyperparameters, paths)
data/raw/           generated or real raw logs (gitignored)
data/processed/     HF DatasetDict saved to disk (gitignored)
docs/PHASES.md      the build plan — one phase per Claude Code session
notebooks/          exploration only; never import from notebooks
src/logsense/       package code
  config.py         load YAML config
  generate.py       Phase 1: synthetic labeled log generator
  baseline.py       Phase 2: zero-shot + TF-IDF baselines
  train_classifier.py  Phase 3
  train_ner.py      Phase 4
  search.py         Phase 5
  app.py            Phase 6: Gradio demo
tests/              pytest, mirrors src/ structure
models/             saved checkpoints (gitignored)
```

## Commands
- Install: `pip install -e ".[dev]"`
- Tests: `pytest -q`
- Lint: `ruff check . && ruff format --check .`
- Generate data: `python -m logsense.generate --config configs/default.yaml`

## Working rules
- Work **one phase at a time** from docs/PHASES.md. Start each phase in plan mode and wait for my approval.
- Small, testable modules. Every new module gets a test in tests/.
- All paths, label lists, model names, and hyperparameters come from configs/ — never hardcode.
- Set random seeds (config `seed`) so results are reproducible.
- **Never use real IPs, usernames, or hostnames.** Use RFC 5737 IPs (192.0.2.0/24, 198.51.100.0/24,
  203.0.113.0/24), RFC 1918 ranges, and fake names like `user01`, `corp-sw-01`.
- Must run on CPU. Keep default training small (DistilBERT, ≤3 epochs, ≤5k samples). GPU is optional.
- **I am learning Hugging Face.** Explain each HF concept in docstrings and comments
  (tokenizer, attention mask, label2id, TrainingArguments, BIO tags, embeddings).
  After each phase, give me a 5-line summary of what I should understand.
- Report metrics honestly: accuracy, macro-F1, per-class F1, confusion matrix. Call out
  when synthetic data makes results look better than real-world performance would be.
- Don't push anything to the Hugging Face Hub without asking me first.

## Domain notes (for realistic log generation)
- Cisco IOS: `%LINK-3-UPDOWN`, `%LINEPROTO-5-UPDOWN`, `%SYS-5-CONFIG_I`, `%SEC_LOGIN-4-LOGIN_FAILED`
- Cisco ISE: `5400 Authentication failed`, `5200 Authentication succeeded`, `5405 RADIUS Request dropped`
- C9800 WLC: client association/deauth, AP join/disjoin, `%CLIENT_ORCH_LOG`, `%APMGR_TRACE_MESSAGE`
- PAN-OS: CSV syslog — TRAFFIC, THREAT, SYSTEM, CONFIG, GLOBALPROTECT log types
- `noise` = routine informational messages (NTP sync, CDP, successful logins, cron)
