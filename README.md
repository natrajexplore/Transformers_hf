# LogSense

Transformer-based syslog triage for **Cisco IOS/IOS-XE, Cisco ISE, Cisco Catalyst 9800 WLC, and
Palo Alto PAN-OS** logs, built end to end with the Hugging Face stack as a hands-on learning project.

Paste a log line and LogSense tells you:

| Question | Answer from | Example |
|---|---|---|
| What kind of event is this? | Fine-tuned DistilBERT classifier (+ TF-IDF baseline) | `%LINK-3-UPDOWN ... down` → `link_event` (0.97) |
| Who and what is involved? | DistilBERT token classifier (NER, BIO tags) | `GigabitEthernet1/0/7` → INTERFACE |
| Have we seen this before? | MiniLM sentence embeddings + FAISS index | top-5 most similar past events |
| What happened in this burst? | Qwen2.5-0.5B-Instruct + LoRA adapter | 3-line incident note |

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
- [Test & performance console](#test--performance-console)
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
| `threat` | PAN-OS THREAT (incl. scans/floods), IOS IPS signatures, ARP inspection, port-security violations, rogue DHCP, WLC rogue AP, ISE threat-centric NAC quarantine |
| `vpn` | IOS crypto/IKE (incl. invalid SPI), IKEv2 SA, ISAKMP failures, DMVPN NHS, PAN-OS IKE and GlobalProtect (even GlobalProtect auth failures) |
| `wireless` | C9800 client association/deauth, `%CLIENT_ORCH_LOG`, `%APMGR_TRACE_MESSAGE`, AP join/disjoin |
| `noise` | NTP sync, CDP, successful logins (incl. SSH success), WLC config-datastore sync, `5200` success, RADIUS accounting, routine CoA, TRAFFIC deny, cron |

**Six entity types**: `IP`, `USER`, `INTERFACE`, `MAC`, `RULE`, `VLAN`, each returned with its
character offsets and a confidence score.

**Realistic synthetic data**: 62 message templates across 4 vendors. Formats follow Cisco and
PAN-OS documentation and Cisco Live 2026 sessions: verbatim C9800 syslog shapes from BRKEWN-3628,
plus ISE / IBNS 2.0 session-manager and RADIUS concepts from BRKSEC-2347 and BRKSEC-2660. Only
events that fit the seven categories went in; posture, RADIUS dead/alive, ISE alarms and debug-file
formats were left out on purpose.

| Vendor | Templates |
|---|---|
| Cisco IOS / IOS-XE | 24 |
| Cisco ISE | 12 |
| Cisco C9800 WLC | 14 |
| Palo Alto PAN-OS | 12 (CSV TRAFFIC, THREAT, SYSTEM, CONFIG, GLOBALPROTECT) |

**Deliberately hard labeling rules**, so the models must read meaning rather than vendor:
a successful login is `noise` rather than `auth_failure`; a TRAFFIC `deny` is `noise`, and only THREAT
logs count as `threat`; a GlobalProtect failure is `vpn`; a WLC uplink flap is `link_event` rather
than `wireless`. The full list is in the docstring of `src/logsense/generate.py`.

**Four ways to use it**
- **Test & performance console**: a FastAPI backend and a web page that feels like a small SOC
  tool. It has a live triage stream with a dashboard and alerts, incident drill-down with an LLM
  note, a labeling workbench that anonymizes your logs, a benchmark and an evaluation view. See
  [Test & performance console](#test--performance-console).
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

 Front ends:  Console (api.py + web/)  ·  Gradio app (app.py)  ·  MCP server (serve.py)  ·  CLI
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
| 7 | Burst summarizer: prompting vs LoRA fine-tuning (stretch goal) | ✅ | `summarize.py` |
| 8 | MCP tool server and a real-log evaluation script | ✅ code (needs anonymized lab logs) | `serve.py`, `evaluate_file.py` |

**Engineering along the way**
- **Held-out-template splits.** One template per category is held out as `test_unseen` and a
  second as `validation_unseen`. That makes the evaluation measure generalization instead of
  memorization, and the classifier checkpoint is chosen on `validation_unseen`.
- **Data v2.** The dataset was rebuilt with Cisco Live 2026 formats and every model was retrained.
- **Data v3.** Error analysis showed `vpn` trained on only 2 formats once two were held out, so the
  model keyed on surface tokens. 9 templates were added to `vpn`, `threat` and `noise`, and the
  classifier went from 0.475 to 0.628 unseen macro-F1. A test now guards the per-label format count.
- **Two code-review passes**, which led to 18 fixes. Among them: macro-F1 over all labels, a check
  on label order, validation of the held-out lists, strict word-boundary scoring for summaries, and
  report saving that survives a crashed run.
- **66 pytest tests**, one test module per source module, with ruff for lint and format.
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
| **Macro-F1** | 0.320 | **0.677** | 0.628 |
| auth_failure | 1.00 | 1.00 | 0.93 |
| config_change | 0.55 | 1.00 | 1.00 |
| link_event | 0.49 | 0.97 | 1.00 |
| threat | 0.20 | 0.00 | 0.00 |
| vpn | 0.00 | 0.68 | 0.48 |
| wireless | 0.00 | 1.00 | 0.98 |
| noise | 0.00 | 0.08 | 0.00 |

Data v3 (62 templates), 383 unseen-template lines; zero-shot scored on a 100-line sample.
Before data v3, DistilBERT scored 0.475 and TF-IDF 0.632.

What still fails: the unseen IOS `%IPS-4-SIGNATURE` lines are all predicted `vpn` (the "IPS"
token looks like IPsec), and the unseen `%CDP-4-NATIVE_VLAN_MISMATCH` lines are predicted `threat`.
Each per-class unseen score comes from a single held-out template, so it swings a lot.

Confusion matrices: `reports/classifier_confusion_test_unseen.png` and `..._test.png`.

### Entity extraction (`test_unseen`)

| Micro-F1 | USER | INTERFACE | IP | VLAN | MAC |
|---|---|---|---|---|---|
| **0.904** | 0.99 | 0.98 | 1.00 | 0.11 | 0.61 |

Before data v3 the score was 0.863 (VLAN 0.58, MAC 0.43). IP and MAC improved; VLAN got worse.
All unseen VLANs come from one template (`%CDP-4-NATIVE_VLAN_MISMATCH`, where the VLAN is a
bare number in parentheses), and all unseen MACs from one AP-event template. Each score is a
single-format result, so it swings a lot between retrains.

The held-out templates contain no `RULE` entities, so `RULE` isn't scored on this split.

### Similar-event search
How often the top-1 neighbor has the same category: **MiniLM 0.637** vs TF-IDF 0.473
(data v2: 0.521 vs 0.490). The new formats gave embeddings more same-meaning neighbors to find.

### Burst summarizer (Qwen2.5-0.5B-Instruct, 20 held-out bursts)

| | Prompt only | LoRA (120 bursts, 1 epoch, ~76 min CPU) |
|---|---|---|
| Exact 3-line format | 0.00 | **0.35** |
| Exact event count | 0.00 | **0.15** |
| Correct category mentioned | **0.70** | 0.50 |
| ROUGE-L vs reference | 0.107 | **0.347** |
| Seconds per burst | 14.6 | 10.5 |

LoRA trained 4 MB of adapter weights (rank 8, attention projections only). It taught the model the
*shape* of the note: `Incident: / Scope: / Action:`, short, no markdown. But it did not teach it to
*count or read* the burst. It often writes a plausible-looking number ("10 events in 10 min") and
sometimes the wrong category, e.g. a burst of interface flaps summarized as "unexpected login
attempts". Category accuracy dropped from 0.70 to 0.50: it learned the format at the expense of
faithfulness. A 0.5B model with 120 examples is too small for this. In practice, compute counts and
categories with code (or the classifier) and let the LLM only phrase the note.

### Takeaways
- **A cheap baseline can win.** TF-IDF character n-grams beat the fine-tuned transformer on unseen
  formats, because they catch shared tokens like `IKE`, `AUTH` and `%SYS`.
- **Fine-tuning on narrow data teaches shortcuts, and more formats fix part of it.** With only 2
  `vpn` formats DistilBERT learned surface cues; adding formats lifted it from 0.475 to 0.628.
- **Each method fails differently.** Zero-shot is still the only model that finds some unseen
  `threat` lines.
- **NER transfers better than classification.** Entities such as IPs and interfaces look alike in
  every format.
- **LoRA teaches format, not facts.** A small adapter fixed the output shape quickly, but a 0.5B model
  still invents counts. Structure is cheap to learn; faithfulness is not.
- **Real-world performance will differ** from both splits. `evaluate_file.py` exists to measure that
  on real, anonymized logs.

Full numbers are in `reports/*.json`. The model cards are in `docs/model_cards/`.

---

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # add ,stretch (Phase 7), ,serve (MCP) and ,web (console)
pytest -q                          # 66 tests
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
python -m logsense.api                      # Test & performance console at http://127.0.0.1:8000
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

**Evaluate on your own logs**: the easiest way is the console's **Label** tab, which anonymizes,
labels and exports the file for you. You can also write a `.jsonl` file yourself, with one
`{"text": ..., "label": ...}` per line, using anonymized lines only. `evaluate_file` reports accuracy, macro-F1 and per-class F1 for the
labels present in the file.

---

## Test & performance console
```bash
python -m logsense.api            # open http://127.0.0.1:8000 · API docs at /api/docs
```
The backend loads each model on first use and keeps it in memory. The summarizer adds about 2 GB,
so only the first summary is slow. It listens on 127.0.0.1 only.

| Tab | What you can do |
|---|---|
| **Live** | Start a simulated syslog stream: mostly routine traffic, with incident bursts pinned to one device (a flapping switch, a brute-force run, a scanner). Each line is classified and its entities extracted as it arrives, with severity colors, filters, a ground-truth toggle that flags model mistakes, and CSV export. The dashboard shows counts by category and vendor, events/min, top talkers, and **alerts** when a category repeats within a window (rules in `configs/default.yaml` → `api.alerts`). |
| **Incident drill-down** | Click any event or alert. A side panel shows the highlighted line, entities, the 5 most similar past events, the related events in the window, and **Generate incident note**, which runs the LoRA summarizer and shows the classifier's counts next to it. |
| **Try it** | Paste lines or load labeled samples; see category, confidence, entities and per-model ms. Click a result to drill down. |
| **Summarize** | Generate a sample burst with its reference note; compare LoRA and prompt-only notes with format, count and category checks. |
| **Label** | Build the labeled `.jsonl` for real-log evaluation: paste raw logs, then **anonymize** them (IPs → RFC 5737 / 10.x, users → `userNN`, hostnames, MACs, emails and PAN-OS serials, replaced consistently). Label each line with keys `1`–`7`, exclude lines that fit no category with `X`, then **download** the file or **evaluate** directly. Model predictions stay hidden until you evaluate, so they can't bias your labels. |
| **Performance** | Latency and throughput per model on labeled lines, with accuracy; runs stack up. |
| **Evaluation** | All offline metrics (tables, per-class bars, confusion heatmap), plus upload-and-score for a labeled file. |
| **Models** | Paths, sizes, load times, process memory, categories and severities. |

Extras: a light/dark/auto theme, keyboard shortcuts (press `?`), toasts, copy buttons and a
backend status light. The page needs no internet: plain HTML/CSS/JS, no CDN.

| Endpoint | What it does |
|---|---|
| `POST /api/analyze` | Lines → category + confidence, entities, similar events, ms per model |
| `GET /api/samples` | Random labeled lines from `test` / `test_unseen` |
| `POST /api/benchmark` | Lines/s, ms/line and batch p50/p95/max per model, plus accuracy on the labeled sample |
| `POST /api/evaluate` | Upload a labeled `.jsonl`; returns accuracy, macro-F1, per-class F1, confusion matrix, mistakes |
| `GET /api/sample-burst` | A synthetic burst with its reference note and facts |
| `POST /api/summarize` | Burst → note (`mode`: `lora` / `prompt`), the classifier's counts, and checks if facts are given |
| `POST /api/feed` | Next simulated syslog events (stateless: the client passes back the clock and any open burst) |
| `POST /api/anonymize` | Lines → anonymized lines, plus counts of what was replaced |
| `GET /api/config` | Labels, entity types, severities and alert rules for the page |
| `GET /api/metrics` · `GET /api/models` | Saved reports · model paths, sizes, load times, process memory |

Measured on this 16 GB CPU laptop (100 `test_unseen` lines, batch 16, models already loaded):

| Model | Lines/s | ms/line | Note |
|---|---|---|---|
| Classifier (DistilBERT) | 27.8 | 36 | accuracy 0.71 on the sample, matching the offline 0.710 |
| NER (DistilBERT) | 22.7 | 44 | the pipeline runs one line at a time |
| Search (MiniLM + FAISS) | 102.5 | 10 | the embedding pass dominates; the FAISS lookup is near-instant |
| Summarizer (Qwen 0.5B + LoRA) | – | ~5–10 s / burst | process memory goes from ~1.2 GB to ~3 GB |

The summarizer tab shows its known weakness live. On a burst of 7 interface flaps plus 3 noise
lines, the classifier counted 7 `link_event` and 3 `noise` correctly, while the LoRA note claimed
"10 SSH connection failures".

---

## Project layout
```
configs/default.yaml   labels, entity types, model names, hyperparameters, paths
src/logsense/
  config.py            load YAML config, label2id / id2label
  generate.py          Phase 1: 62-template synthetic log generator + DatasetDict
  baseline.py          Phase 2: zero-shot NLI and TF-IDF + LogReg
  metrics.py           accuracy, macro-F1, per-class F1, confusion matrix
  train_classifier.py  Phase 3: DistilBERT sequence classification
  train_ner.py         Phase 4: DistilBERT token classification (BIO)
  search.py            Phase 5: MiniLM embeddings + FAISS
  api.py               FastAPI backend of the test & performance console
  anonymize.py         consistent, pattern-based anonymizer for real syslog lines
  simulate.py          live syslog feed simulator (noise + incident bursts)
  web/                 console page: app.js, live.js, label.js (no build step, works offline)
  app.py               Phase 6: Gradio Blocks demo
  summarize.py         Phase 7: Qwen2.5 prompting vs LoRA
  serve.py             Phase 8: MCP tool server
  evaluate_file.py     Phase 8: score models on a labeled .jsonl
tests/                 66 pytest tests, one module per source module
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
- **Classifier gaps.** On unseen templates DistilBERT still scores 0 on `threat` (IPS signature read
  as `vpn`) and `noise` (CDP mismatch read as `threat`), and 0.48 on `vpn`.
- **NER.** Bare numbers get confused: unseen VLAN F1 is 0.11 (VLAN IDs in parentheses are
  missed), and MAC precision is 0.44 (other hex-like tokens get tagged as MACs).
- **Fixed taxonomy.** Events outside the seven categories, such as posture, RADIUS server
  dead/alive and ISE alarms, would need a new category.
- **Summarizer.** The LoRA model follows the note format 35% of the time but makes up counts and
  sometimes the category; don't trust its numbers.

## What's left
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
MIT (see [LICENSE](LICENSE))
