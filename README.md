# LogSense

Transformer-based syslog triage for **Cisco IOS, ISE, Catalyst 9800 WLC, and Palo Alto PAN-OS** logs,
built with the Hugging Face stack.

Paste a log line → get its **category**, **extracted entities** (IPs, users, interfaces, MACs, rules),
and **similar past events**.

## Status
| Phase | What | Status |
|---|---|---|
| 1 | Synthetic labeled dataset | ⬜ |
| 2 | Zero-shot + TF-IDF baselines | ⬜ |
| 3 | Fine-tuned classifier | ⬜ |
| 4 | NER entity extraction | ⬜ |
| 5 | Similar-event search (FAISS) | ⬜ |
| 6 | Gradio demo + HF Hub | ⬜ |
| 7 | LoRA burst summarizer (stretch) | ⬜ |

Full plan: [docs/PHASES.md](docs/PHASES.md)

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                          # should pass: config tests
huggingface-cli login              # only needed for Phase 6
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
MIT
