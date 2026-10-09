# LogSense Build Plan

One phase per Claude Code session. For each phase:
1. Start Claude Code, press **Shift+Tab** to enter plan mode.
2. Paste the phase prompt (or run `/next-phase`).
3. Review the plan, ask questions, approve.
4. Run tests, check "Done when", commit, then `/clear` before the next phase.

---

## Phase 1 — Synthetic labeled dataset
**Learn:** `datasets` library, DatasetDict, ClassLabel features, train/val/test splits

**Prompt:**
> Implement Phase 1 from docs/PHASES.md. Build `src/logsense/generate.py` that creates realistic,
> anonymized syslog lines for Cisco IOS, ISE, C9800 WLC, and PAN-OS, labeled with the categories in
> configs/default.yaml. Use real message formats from the Domain notes in CLAUDE.md, vary fields
> randomly, and add some hard examples (e.g. a successful login is `noise`, not `auth_failure`).
> Also record entity character spans (IP, USER, INTERFACE, MAC, RULE, VLAN) for Phase 4.
> Save a HF DatasetDict to data/processed/ with stratified train/val/test splits. Add tests.

**Done when:** `python -m logsense.generate` produces ~3,000 rows, balanced classes, tests pass,
and you can print 10 random examples that look like real logs.

---

## Phase 2 — Baselines
**Learn:** `pipeline()`, zero-shot classification, why baselines matter

**Prompt:**
> Implement Phase 2. In `src/logsense/baseline.py`, evaluate two baselines on the test split:
> (a) `pipeline("zero-shot-classification")` with the model in config, and (b) TF-IDF + LogisticRegression.
> Report accuracy, macro-F1, per-class F1. Save results to reports/baseline.json. Explain in comments
> how zero-shot NLI classification works.

**Done when:** you have two numbers to beat.

---

## Phase 3 — Fine-tune the classifier
**Learn:** AutoTokenizer, AutoModelForSequenceClassification, label2id/id2label, Trainer,
TrainingArguments, `evaluate`, early stopping

**Prompt:**
> Implement Phase 3. In `src/logsense/train_classifier.py`, fine-tune the classifier model from config
> on our dataset using Trainer. Tokenize with truncation to max_length from config. Compute accuracy
> and macro-F1 with `evaluate`. Save the best model to models/classifier/, plus a confusion matrix
> PNG and metrics JSON in reports/. Add a `predict(texts)` helper. Comment every TrainingArguments choice.

**Done when:** fine-tuned model beats both baselines; you can explain what each training arg does.

---

## Phase 4 — Entity extraction (NER)
**Learn:** token classification, BIO tagging, subword alignment (`word_ids()`), seqeval

**Prompt:**
> Implement Phase 4. In `src/logsense/train_ner.py`, convert the entity char spans from Phase 1 into
> BIO labels aligned to subword tokens, fine-tune AutoModelForTokenClassification, and evaluate with
> seqeval (per-entity precision/recall/F1). Add an `extract_entities(text)` helper that returns
> grouped entities. Explain the -100 label trick for subword tokens.

**Done when:** entity F1 is reported per type and `extract_entities` works on a pasted PAN-OS line.

---

## Phase 5 — Similar-event search
**Learn:** sentence embeddings, cosine similarity, FAISS indexing

**Prompt:**
> Implement Phase 5. In `src/logsense/search.py`, embed the training logs with the embedding model
> from config, build a FAISS index saved to models/search/, and add `find_similar(text, k)` returning
> the top-k logs with scores and categories. Include a quick quality check: for test logs, how often
> does the top-1 neighbor share the same category?

**Done when:** searching an ISE auth failure returns other auth failures.

---

## Phase 6 — Demo app + Hugging Face Hub
**Learn:** Gradio, `push_to_hub`, model cards, HF Spaces

**Prompt:**
> Implement Phase 6. Build `src/logsense/app.py` — a Gradio app where I paste one or more log lines
> and see category + confidence, extracted entities (highlighted), and similar past events.
> Write model cards for the classifier and NER models. Prepare a `space/` folder (app.py +
> requirements.txt) ready for a HF Space. Ask me before pushing anything to the Hub.

**Done when:** `python -m logsense.app` runs locally and the Space folder is ready.

---

## Phase 7 (stretch) — Burst summarizer with LoRA
**Learn:** PEFT/LoRA, chat templates, small instruct models

**Prompt:**
> Implement Phase 7. Use a small instruct model (≤1.5B params) to summarize a burst of 10–50 logs into
> a 3-line incident note. First try prompting only; then, if useful, fine-tune with LoRA via `peft`
> on synthetic (burst → summary) pairs. Compare before/after on 20 held-out bursts.

---

## Phase 8 (stretch) — Real data + integrations
- Swap in anonymized logs from your lab; compare synthetic vs real metrics (great LinkedIn post)
- Wrap `classify` / `extract_entities` / `find_similar` as an MCP server or FastAPI service
  so n8n or CrewAI agents can call LogSense as a tool
