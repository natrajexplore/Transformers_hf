# LogSense

Transformer-based syslog triage for **Cisco IOS, ISE, Catalyst 9800 WLC, and Palo Alto PAN-OS** logs,
built with the Hugging Face stack.

Paste a log line → get its **category**, **extracted entities** (IPs, users, interfaces, MACs, rules),
and **similar past events**.

## Status
| Phase | What | Status |
|---|---|---|
| 1 | Synthetic labeled dataset (53 templates, held-out-template splits) | ✅ |
| 2 | Zero-shot + TF-IDF baselines | ✅ |
| 3 | Fine-tuned classifier (DistilBERT) | ✅ |
| 4 | NER entity extraction (BIO, seqeval) | ✅ |
| 5 | Similar-event search (MiniLM + FAISS) | ✅ |
| 6 | Gradio demo, model cards, Space folder | ✅ (Hub upload not done — see `space/README.md`) |
| 7 | Burst summarizer: prompting vs LoRA (stretch) | ⚠️ prompting done; LoRA run pending (needs a terminal run, see below) |
| 8 | MCP tool server; real-log evaluation script | ✅ code — needs your anonymized lab logs |

Full plan: [docs/PHASES.md](docs/PHASES.md)

## Results (honest version)

Everything is trained on **synthetic** logs. Each category's `test` split comes from the same
templates as training, so every model scores ~1.0 there — that measures memorization.
`test_unseen` holds out whole message templates and is the number that matters.

| `test_unseen` | Zero-shot (bart-large-mnli) | TF-IDF + LogReg | DistilBERT fine-tuned |
|---|---|---|---|
| Classification macro-F1 | 0.369 | **0.632** | 0.475 |

| `test_unseen` | Result |
|---|---|
| NER entity micro-F1 | 0.863 (USER 1.00, INTERFACE 1.00, IP 0.93, VLAN 0.58, MAC 0.43) |
| Search: top-1 neighbor has same category | MiniLM 0.521 vs TF-IDF 0.490 |

Burst summarizer (Qwen2.5-0.5B-Instruct, prompting only, 20 held-out bursts): correct
category mentioned 0.65, exact event count 0.00, exact 3-line format 0.00, ROUGE-L 0.117 —
it writes long free-form markdown instead of the requested note. The LoRA comparison is pending.

Takeaways: a cheap TF-IDF baseline beats the fine-tuned transformer on unseen formats;
fine-tuning on narrow data teaches format shortcuts. Details in `reports/*.json` and the model
cards in `docs/model_cards/`.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # add ,stretch for Phase 7 and ,serve for the MCP server
pytest -q
```

## Commands
```bash
python -m logsense.generate                 # Phase 1: data/processed + data/raw/logs.jsonl
python -m logsense.baseline --zero-shot-limit 100   # Phase 2 (zero-shot is slow on CPU)
python -m logsense.train_classifier         # Phase 3 (~25 min CPU)
python -m logsense.train_ner                # Phase 4 (~35 min CPU)
python -m logsense.search                   # Phase 5: build index + evaluate
python -m logsense.search --query "<log line>"
python -m logsense.app                      # Phase 6: Gradio demo at http://127.0.0.1:7860
python -m logsense.summarize                # Phase 7: prompting vs LoRA (needs .[stretch])
python -m logsense.serve                    # Phase 8: MCP server over stdio (needs .[serve])
python -m logsense.evaluate_file --input lab.jsonl   # Phase 8: score on your labeled logs
```
Long training runs are best started from your own terminal; on a 16 GB laptop, run them one at
a time.

Use LogSense from Claude Code (or any MCP client):
```bash
claude mcp add logsense -- .venv/Scripts/python -m logsense.serve
```

## Building with Claude Code
```bash
claude
```
Then inside Claude Code:
- `/next-phase` — picks up the next unfinished phase, explains the concepts, proposes a plan
- `/explain <concept>` — e.g. `/explain attention mask`, `/explain BIO tagging`
- Press **Shift+Tab** for plan mode before approving any phase
- `/clear` between phases; `CLAUDE.md` carries the project context forward

## Data privacy
All sample data is synthetic. Real lab logs must be anonymized before they enter `data/raw/`
(RFC 5737 documentation IPs, placeholder usernames and hostnames).

## License
Apache-2.0 (see [LICENSE](LICENSE))
