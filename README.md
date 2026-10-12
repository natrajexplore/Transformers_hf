# LogSense

Transformer-based syslog triage for **Cisco IOS/IOS-XE, Cisco ISE, Cisco Catalyst 9800 WLC, and
Palo Alto PAN-OS** logs, built end to end with the Hugging Face stack as a hands-on learning project.

Paste a log line and LogSense tells you:

| Question | Answer from | Example |
|---|---|---|
| What kind of event is this? | Fine-tuned DistilBERT classifier (+ TF-IDF baseline) | `%LINK-3-UPDOWN ... down` → `link_event` (0.97) |
| Who and what is involved? | DistilBERT token classifier (NER, BIO tags) | `GigabitEthernet1/0/7` → INTERFACE |
| Have we seen this before? | MiniLM sentence embeddings + FAISS index | top-5 most similar past events |
| What happened in this burst? | Qwen2.5-0.5B-Instruct, prompted (LoRA run pending) | 3-line incident note |

All of it runs on a CPU laptop, through a Gradio web demo, the command line, or an MCP tool server
that Claude Code and other agents can call.

---

## Contents
- [Features](#features)
- [How it works](#how-it-works)
- [What we built, phase by phase](#what-we-built-phase-by-phase)
- [Results (honest version)](#results-honest-version)
- [Quick start](#quick-start)
- [Commands](#commands)
- [Project layout](#project-layout)
- [Hugging Face concepts covered](#hugging-face-concepts-covered)
- [Known limitations](#known-limitations)
- [What's left](#whats-left)
- [Data privacy](#data-privacy) · [License](#license)

---

## Features

**Seven triage categories** (single source of truth: `configs/default.yaml`)

| Category | Typical events |
|---|---|
| `auth_failure` | IOS `%SEC_LOGIN-4-LOGIN_FAILED`, ISE `5400`/`5405`, 802.1X/MAB failures, WLC wrong PSK, PAN-OS auth fail |
| `config_change` | IOS `%SYS-5-CONFIG_I`, parser command accounting, ISE TACACS+ `3300` and admin config, WLC/PAN-OS config logs |
| `link_event` | `%LINK-3-UPDOWN`, `%LINEPROTO-5-UPDOWN`, admin-down, PAN-OS link changes, WLC uplinks |
| `threat` | PAN-OS THREAT, IOS IPS signatures and ARP inspection, WLC rogue AP, ISE threat-centric NAC quarantine |
| `vpn` | IOS crypto/IKE (incl. invalid SPI), PAN-OS IKE and GlobalProtect (even GlobalProtect auth failures) |
| `wireless` | C9800 client association/deauth, `%CLIENT_ORCH_LOG`, `%APMGR_TRACE_MESSAGE`, AP join/disjoin |
| `noise` | NTP sync, CDP, successful logins, `5200` success, RADIUS accounting, routine CoA, TRAFFIC deny, cron |

**Six entity types**: `IP`, `USER`, `INTERFACE`, `MAC`, `RULE`, `VLAN`, each returned with its
character offsets and a confidence score.

**Realistic synthetic data**: 53 message templates across 4 vendors. Formats follow Cisco and
PAN-OS documentation and Cisco Live 2026 sessions: verbatim C9800 syslog shapes from BRKEWN-3628,
plus ISE / IBNS 2.0 session-manager and RADIUS concepts from BRKSEC-2347 and BRKSEC-2660. Only
events that fit the seven categories went in; posture, RADIUS dead/alive, ISE alarms and debug-file
formats were left out on purpose.

| Vendor | Templates |
|---|---|
| Cisco IOS / IOS-XE | 18 |
| Cisco ISE | 12 |
| Cisco C9800 WLC | 13 |
| Palo Alto PAN-OS | 10 (CSV TRAFFIC, THREAT, SYSTEM, CONFIG, GLOBALPROTECT) |

**Deliberately hard labeling rules**, so the models must read meaning rather than vendor:
a successful login is `noise` rather than `auth_failure`; a TRAFFIC `deny` is `noise`, and only THREAT
logs count as `threat`; a GlobalProtect failure is `vpn`; a WLC uplink flap is `link_event` rather
than `wireless`. The full list is in the docstring of `src/logsense/generate.py`.

**Three ways to use it**
- **Gradio demo**: paste many lines and get a category table, highlighted entities and similar past
  events, with a "neighbors per line" slider.
- **MCP server**: four tools for Claude Code, n8n or any MCP client: `classify_logs`,
  `extract_log_entities`, `find_similar_events`, and a one-call `triage`.
- **CLI**: every phase is a `python -m logsense.<module>` command, and `evaluate_file` scores the
  models on your own labeled lab logs.

---

## How it works

```
                 ┌──────────────────────────────┐
 log line(s) ──► │ classifier (DistilBERT)      │──► category + confidence
                 ├──────────────────────────────┤
                 │ NER (DistilBERT, BIO tags)   │──► IP / USER / INTERFACE / MAC / RULE / VLAN
                 ├──────────────────────────────┤
                 │ MiniLM embedding → FAISS     │──► top-k similar past events (cosine)
                 └──────────────────────────────┘
 burst of 8–15 ──► Qwen2.5-0.5B-Instruct (chat template, optional LoRA) ──► 3-line incident note

 Front ends:  Gradio app (app.py)  ·  MCP server (serve.py)  ·  CLI / evaluate_file.py
```

The models, label lists, hyperparameters and paths all come from `configs/default.yaml`, and the
random seed is fixed (`seed: 42`) so runs are reproducible.

---

## What we built, phase by phase

| Phase | What | Status | Key files |
|---|---|---|---|
| 1 | Synthetic labeled dataset: 3,000 lines, entity spans, stratified splits, held-out-template splits | ✅ | `generate.py` |
| 2 | Baselines: zero-shot NLI (`bart-large-mnli`) and TF-IDF char n-grams + logistic regression | ✅ | `baseline.py`, `metrics.py` |
| 3 | Fine-tuned classifier: DistilBERT, checkpoint chosen on unseen templates, early stopping | ✅ | `train_classifier.py` |
| 4 | NER: BIO tags aligned via `word_ids()` and offsets, `-100` masking, seqeval metrics | ✅ | `train_ner.py` |
| 5 | Similar-event search: MiniLM embeddings, FAISS `IndexFlatIP`, compared against TF-IDF | ✅ | `search.py` |
| 6 | Gradio demo, model cards, Hugging Face Space folder | ✅ code (Hub upload awaiting approval) | `app.py`, `docs/model_cards/`, `space/` |
| 7 | Burst summarizer: prompting vs LoRA fine-tuning (stretch goal) | ⚠️ prompting done, LoRA run pending | `summarize.py` |
| 8 | MCP tool server and a real-log evaluation script | ✅ code (needs anonymized lab logs) | `serve.py`, `evaluate_file.py` |

**Engineering along the way**
- **Held-out-template splits.** One template per category is held out as `test_unseen` and a
  second as `validation_unseen`. That makes the evaluation measure generalization instead of
  memorization, and the classifier checkpoint is chosen on `validation_unseen`.
- **Data v2.** The dataset was rebuilt with Cisco Live 2026 formats and every model was retrained.
- **Two code-review passes**, which led to 18 fixes. Among them: macro-F1 over all labels, a check
  on label order, validation of the held-out lists, strict word-boundary scoring for summaries, and
  report saving that survives a crashed run.
- **48 pytest tests**, one test module per source module, with ruff for lint and format.
- **CPU-friendly.** Batched embedding and dynamic padding keep inference light, gradient
  checkpointing and one long burst per step keep LoRA training within 16 GB of RAM, and the CPU
  training times are recorded in `reports/`.

---

## Results (honest version)

Everything is trained on **synthetic** logs. The in-distribution `test` split comes from the same
templates as training, so every model scores about 1.0 there, which only measures memorization.
**`test_unseen` holds out whole message templates and is the number that matters.**

### Classification (macro-F1 on `test_unseen`)

| | Zero-shot (bart-large-mnli) | TF-IDF + LogReg | DistilBERT fine-tuned |
|---|---|---|---|
| **Macro-F1** | 0.369 | **0.632** | 0.475 |
| auth_failure | 1.00 | 1.00 | 0.42 |
| config_change | 0.54 | 1.00 | 1.00 |
| link_event | 0.45 | 0.52 | 0.91 |
| threat | 0.59 | 0.00 | 0.00 |
| vpn | 0.00 | 0.87 | 0.00 |
| wireless | 0.00 | 1.00 | 1.00 |
| noise | 0.00 | 0.03 | 0.00 |

Confusion matrices: `reports/classifier_confusion_test_unseen.png` and `..._test.png`.

### Entity extraction (`test_unseen`)

| Micro-F1 | USER | INTERFACE | IP | VLAN | MAC |
|---|---|---|---|---|---|
| **0.863** | 1.00 | 1.00 | 0.93 | 0.58 | 0.43 |

The held-out templates contain no `RULE` entities, so `RULE` isn't scored on this split.

### Similar-event search
How often the top-1 neighbor has the same category: **MiniLM 0.521** vs TF-IDF 0.490.

### Burst summarizer (20 held-out bursts, prompting only)
Qwen2.5-0.5B-Instruct mentioned the correct category 0.65 of the time, but got the exact event count
0.00 and the exact 3-line format 0.00 (ROUGE-L 0.117). It writes long free-form markdown instead of
the requested note, which is the gap the pending LoRA run is meant to close.

### Takeaways
- **A cheap baseline can win.** TF-IDF character n-grams beat the fine-tuned transformer on unseen
  formats, because they catch shared tokens like `IKE`, `AUTH` and `%SYS`.
- **Fine-tuning on narrow data teaches shortcuts.** DistilBERT learned vendor and format cues and
  missed `threat`, `vpn` and `noise` when the wording was new.
- **Each method fails differently.** Zero-shot is the only model that finds some unseen `threat` lines.
- **NER transfers better than classification.** Entities such as IPs and interfaces look alike in
  every format.
- **Real-world performance will differ** from both splits. `evaluate_file.py` exists to measure that
  on real, anonymized logs.

Full numbers are in `reports/*.json`. The model cards are in `docs/model_cards/`.

---

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # add ,stretch for Phase 7 and ,serve for the MCP server
pytest -q                          # 48 tests
```
Tested with Python 3.11+, transformers 5.x, datasets 5.x, torch 2.x (CPU), sentence-transformers,
faiss-cpu, peft, gradio 6 and mcp.

## Commands
```bash
python -m logsense.generate                 # Phase 1: data/processed + data/raw/logs.jsonl
python -m logsense.baseline --zero-shot-limit 100   # Phase 2 (zero-shot is slow on CPU)
python -m logsense.train_classifier         # Phase 3 (~25 min CPU)
python -m logsense.train_ner                # Phase 4 (~35 min CPU)
python -m logsense.search                   # Phase 5: build index + evaluate
python -m logsense.search --query "<log line>"
python -m logsense.app                      # Phase 6: Gradio demo at http://127.0.0.1:7860
python -m logsense.summarize                # Phase 7: prompting vs LoRA (needs .[stretch], ~1.5 h)
python -m logsense.serve                    # Phase 8: MCP server over stdio (needs .[serve])
python -m logsense.evaluate_file --input lab.jsonl   # Phase 8: score on your labeled logs
```
Every command accepts `--config <path>`. On a 16 GB laptop, start long training runs from your own
terminal and run them one at a time.

**Use LogSense from Claude Code** (or any MCP client):
```bash
claude mcp add logsense -- .venv/Scripts/python -m logsense.serve
```
Then ask, for example, *"triage this log line: …"*. Claude calls the `triage` tool and gets back the
category, entities and similar events.

**Evaluate on your own logs**: create a `.jsonl` file with one `{"text": ..., "label": ...}` per line,
using anonymized lines only. `evaluate_file` reports accuracy, macro-F1 and per-class F1 for the
labels present in the file.

---

## Project layout
```
configs/default.yaml   labels, entity types, model names, hyperparameters, paths
src/logsense/
  config.py            load YAML config, label2id / id2label
  generate.py          Phase 1: 53-template synthetic log generator + DatasetDict
  baseline.py          Phase 2: zero-shot NLI and TF-IDF + LogReg
  metrics.py           accuracy, macro-F1, per-class F1, confusion matrix
  train_classifier.py  Phase 3: DistilBERT sequence classification
  train_ner.py         Phase 4: DistilBERT token classification (BIO)
  search.py            Phase 5: MiniLM embeddings + FAISS
  app.py               Phase 6: Gradio Blocks demo
  summarize.py         Phase 7: Qwen2.5 prompting vs LoRA
  serve.py             Phase 8: MCP tool server
  evaluate_file.py     Phase 8: score models on a labeled .jsonl
tests/                 48 pytest tests, one module per source module
docs/PHASES.md         the build plan
docs/model_cards/      classifier.md, ner.md
space/                 ready-to-upload Hugging Face Space (app, requirements, checklist)
reports/               metrics JSON + confusion-matrix PNGs (logs gitignored)
models/                trained models (gitignored)
```

---

## Hugging Face concepts covered
Each concept is explained in the docstrings and comments of the module that uses it.

| Concept | Where |
|---|---|
| `Dataset`, `DatasetDict`, `Features`, `ClassLabel`, Arrow, `save_to_disk` | `generate.py` |
| Zero-shot classification with NLI pipelines and hypothesis templates | `baseline.py` |
| `AutoTokenizer`, subwords, attention mask, `label2id`/`id2label` | `train_classifier.py` |
| `AutoModelForSequenceClassification`, `Trainer`, `TrainingArguments`, early stopping, `evaluate` | `train_classifier.py` |
| Dynamic padding with `DataCollatorWithPadding`, `pipeline("text-classification")` | `train_classifier.py` |
| Token classification, BIO tags, `word_ids()`, `-100` labels, seqeval, aggregation strategies | `train_ner.py` |
| Sentence embeddings, mean pooling, cosine similarity, FAISS inner-product index | `search.py` |
| Chat templates, causal LM generation, LoRA with `peft`, label masking, gradient checkpointing | `summarize.py` |
| Gradio Blocks, Hugging Face Spaces, model cards | `app.py`, `space/`, `docs/model_cards/` |

---

## Known limitations
- **Synthetic data only.** Real devices produce formats, typos and volumes the generator does not.
- **Classifier gaps.** `threat`, `vpn` and `noise` score 0 for DistilBERT on unseen templates;
  more template variety per category would help most.
- **NER.** Bare numbers get confused (VLAN IDs versus other numeric fields), and MAC F1 is low
  on unseen formats.
- **Fixed taxonomy.** Events outside the seven categories, such as posture, RADIUS server
  dead/alive and ISE alarms, would need a new category.
- **Summarizer.** Prompt-only output doesn't follow the requested format yet, and the LoRA results
  are still pending.

## What's left
- [ ] Run the LoRA training (`python -m logsense.summarize`, about 1.5 h on CPU) and add its results here
- [ ] Evaluate on anonymized real lab logs with `evaluate_file`
- [ ] Publish the models and the Space to the Hugging Face Hub (steps in `space/README.md`, only after approval)

---

## Building with Claude Code
This project was built one phase at a time with Claude Code, following `docs/PHASES.md`.
```bash
claude
```
- `/next-phase`: picks up the next unfinished phase, explains the concepts and proposes a plan
- `/explain <concept>`: for example `/explain attention mask` or `/explain BIO tagging`
- Press **Shift+Tab** to enter plan mode before approving a phase
- Run `/clear` between phases; `CLAUDE.md` carries the project context forward

## Data privacy
All sample data is synthetic. It uses only RFC 5737 documentation IPs (`192.0.2.0/24`,
`198.51.100.0/24`, `203.0.113.0/24`), RFC 1918 ranges, and placeholder names such as `user07` and
`corp-sw-03`. Real lab logs must be anonymized the same way before they go into `data/raw/`.

## License
Apache-2.0 (see [LICENSE](LICENSE))
