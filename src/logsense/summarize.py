"""Phase 7 (stretch): summarize a burst of logs into a 3-line incident note with a small LLM.

Step 1 prompts the instruct model as-is; step 2 fine-tunes it with LoRA on synthetic
(burst -> note) pairs; both are scored on the same held-out bursts.

Hugging Face concepts used here
-------------------------------
* Instruct / chat models — trained to follow instructions given as a list of messages
  (``{"role": "system"|"user"|"assistant", "content": ...}``).
* Chat templates — each model family expects those messages wrapped in its own special tokens
  (Qwen uses ``<|im_start|>user ... <|im_end|>``). ``tokenizer.apply_chat_template`` does the
  wrapping, and ``add_generation_prompt=True`` appends the "assistant, your turn" marker.
* ``model.generate`` — produces text one token at a time; greedy decoding (``do_sample=False``)
  makes the output deterministic.
* LoRA / PEFT — instead of updating all ~500M weights, freeze them and learn a small low-rank
  update ``W + (alpha/r) * B @ A`` for chosen layers (here the attention projections). Only
  ~0.2% of the parameters train, so it fits on a CPU, and the result is a few-MB "adapter".
* Label masking again — like the -100 trick in Phase 4, prompt tokens get label -100 so the
  loss only teaches the model to write the summary, not to repeat the logs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from logsense.config import load_config
from logsense.generate import TEMPLATES, render

INSTRUCTION = (
    "Summarize this burst of network syslog messages as a 3-line incident note.\n"
    "Line 1 starts with 'Incident:' - what happened and how many events.\n"
    "Line 2 starts with 'Scope:' - how many devices and the most frequent message.\n"
    "Line 3 starts with 'Action:' - the recommended next step.\n\nLogs:\n"
)

_HOST = re.compile(r"^\w{3}\s+\d+ [\d:]+ (\S+?):? ")
_SIGNATURES = [
    (re.compile(r"%([A-Z0-9_]+-\d-[A-Z0-9_]+)"), "{}"),  # Cisco IOS / C9800 mnemonic
    (re.compile(r" (\d{4,5}) (?:NOTICE|INFO|WARN) "), "ISE {}"),  # ISE message code
    (re.compile(r",(TRAFFIC|THREAT|SYSTEM|CONFIG|GLOBALPROTECT|AUTH),"), "PAN-OS {}"),
]


def adapter_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["models_dir"]) / "summarizer_lora"


def signature(line: str) -> str:
    """A short message-type name for a log line, e.g. ``LINK-3-UPDOWN`` or ``ISE 5400``."""
    for pattern, fmt in _SIGNATURES:
        if m := pattern.search(line):
            return fmt.format(m.group(1))
    return "unknown"


def host(line: str) -> str:
    m = _HOST.match(line)
    return m.group(1) if m else "unknown"


def make_burst(rng: random.Random, cfg: dict[str, Any]) -> dict[str, Any]:
    """One burst: mostly events of one incident category, padded with routine noise, all
    within a few minutes. Returns the lines plus the facts the reference note is built from."""
    s = cfg["summarizer"]
    vendors = set(cfg["data"]["vendors"])
    label = rng.choice(sorted(s["incidents"]))
    total = rng.randint(s["burst_min"], s["burst_max"])
    count = max(1, round(total * rng.uniform(0.6, 0.9)))

    start = datetime(2026, 3, 1, tzinfo=timezone.utc) + timedelta(minutes=rng.randrange(500_000))
    span = rng.randint(60, 900)  # the burst lasts 1-15 minutes
    events = []
    for i in range(total):
        pool_label = label if i < count else "noise"
        pool = [fn for v, fn in TEMPLATES[pool_label] if v in vendors]
        dt = start + timedelta(seconds=rng.randint(0, span))
        events.append((dt, pool_label, render(*rng.choice(pool)(rng, dt))[0]))
    events.sort(key=lambda e: e[0])

    incident = [text for _, lbl, text in events if lbl == label]
    times = [dt for dt, lbl, _ in events if lbl == label]
    return {
        "lines": [text for _, _, text in events],
        "facts": {
            "label": label,
            "count": count,
            "total": total,
            "minutes": max(1, math.ceil((times[-1] - times[0]).total_seconds() / 60)),
            "devices": len({host(t) for t in incident}),
            "top_message": Counter(signature(t) for t in incident).most_common(1)[0][0],
        },
    }


def make_bursts(cfg: dict[str, Any], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return [make_burst(rng, cfg) for _ in range(n)]


def reference_summary(facts: dict[str, Any], cfg: dict[str, Any]) -> str:
    inc = cfg["summarizer"]["incidents"][facts["label"]]
    devices = f"{facts['devices']} device" + ("s" if facts["devices"] != 1 else "")
    return (
        f"Incident: {facts['count']} {inc['phrase']} events within {facts['minutes']} min "
        f"({facts['count']} of {facts['total']} logs in the burst).\n"
        f"Scope: {devices} affected; most frequent message: {facts['top_message']}.\n"
        f"Action: {inc['action']}."
    )


def build_messages(lines: list[str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a concise network operations assistant."},
        {"role": "user", "content": INSTRUCTION + "\n".join(lines)},
    ]


def generate_summary(model: Any, tokenizer: Any, lines: list[str], max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(
        build_messages(lines), add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    with torch.no_grad():
        out = model.generate(
            **prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy -> reproducible
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, prompt["input_ids"].shape[1] :]  # drop the echoed prompt
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def score(summary: str, facts: dict[str, Any], cfg: dict[str, Any]) -> dict[str, bool]:
    """Simple checkable facts: right shape, right event count, right category mentioned."""
    lines = [ln.strip() for ln in summary.splitlines() if ln.strip()]
    keywords = cfg["summarizer"]["incidents"][facts["label"]]["keywords"]
    incident_line = next((ln for ln in lines if ln.startswith("Incident:")), "")
    return {
        "format_ok": len(lines) == 3
        and [ln.split(":")[0] for ln in lines] == ["Incident", "Scope", "Action"],
        # the event count must be the first number in the Incident line, not just anywhere
        "count_ok": re.match(rf"Incident:\D*\b{facts['count']}\b", incident_line) is not None,
        # whole words only: "ap" must not match "approved"
        "category_ok": any(re.search(rf"\b{re.escape(k)}\b", summary.lower()) for k in keywords),
    }


def evaluate_model(
    model: Any, tokenizer: Any, bursts: list[dict[str, Any]], cfg: dict[str, Any]
) -> dict[str, Any]:
    import evaluate as hf_evaluate

    preds, checks = [], []
    t0 = time.perf_counter()
    for i, b in enumerate(bursts):
        preds.append(
            generate_summary(model, tokenizer, b["lines"], cfg["summarizer"]["max_new_tokens"])
        )
        checks.append(score(preds[-1], b["facts"], cfg))
        print(f"  summarized {i + 1}/{len(bursts)}", flush=True)
    refs = [reference_summary(b["facts"], cfg) for b in bursts]
    rouge = hf_evaluate.load("rouge").compute(predictions=preds, references=refs)
    return {
        "n": len(bursts),
        "seconds_per_burst": round((time.perf_counter() - t0) / len(bursts), 1),
        **{k: round(sum(c[k] for c in checks) / len(checks), 3) for k in checks[0]},
        "rougeL": round(float(rouge["rougeL"]), 3),
        "examples": [{"reference": r, "prediction": p} for r, p in zip(refs[:3], preds[:3])],
    }


def tokenize_pair(tokenizer: Any, lines: list[str], summary: str) -> dict:
    """input_ids = prompt + summary; labels = -100 over the prompt, so only the summary is learned.

    Never truncated: cutting the end would drop the summary (the only trained part) and its EOS;
    train_lora skips examples that are too long instead.
    """
    prompt_ids = tokenizer.apply_chat_template(build_messages(lines), add_generation_prompt=True)
    if not isinstance(prompt_ids, list):  # transformers 5 returns a BatchEncoding
        prompt_ids = prompt_ids["input_ids"]
    answer_ids = tokenizer(summary + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def train_lora(
    model: Any, tokenizer: Any, bursts: list[dict[str, Any]], cfg: dict[str, Any]
) -> Any:
    from peft import LoraConfig, get_peft_model

    s = cfg["summarizer"]
    lora = LoraConfig(
        r=s["lora"]["r"],
        lora_alpha=s["lora"]["alpha"],
        lora_dropout=s["lora"]["dropout"],
        target_modules=s["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)  # freezes the base weights, adds trainable A/B matrices
    model.print_trainable_parameters()

    examples = [
        tokenize_pair(tokenizer, b["lines"], reference_summary(b["facts"], cfg)) for b in bursts
    ]
    fits = [ex for ex in examples if len(ex["input_ids"]) <= s["max_length"]]
    print(f"training on {len(fits)}/{len(examples)} bursts (skipped > {s['max_length']} tokens)")
    train_ds = Dataset.from_list(fits)
    args = TrainingArguments(
        output_dir=str(adapter_dir(cfg).parent / "summarizer_checkpoints"),
        num_train_epochs=s["epochs"],
        per_device_train_batch_size=s["batch_size"],
        gradient_accumulation_steps=s["gradient_accumulation_steps"],
        learning_rate=s["learning_rate"],
        warmup_steps=s["warmup_ratio"],  # transformers 5: a float < 1 is a fraction of steps
        lr_scheduler_type="linear",
        logging_steps=s["logging_steps"],
        save_strategy="no",  # adapter is saved explicitly below
        seed=cfg["seed"],
        report_to="none",
        use_cpu=s.get("use_cpu", False),
        dataloader_num_workers=0,
        # Gradient checkpointing: keep only some layer activations for the backward pass and
        # recompute the rest. ~30% slower, but far less memory on 2-3k-token bursts.
        gradient_checkpointing=s["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},  # works with frozen LoRA inputs
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        # pads input_ids with the pad token and labels with -100
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )
    trainer.train()
    model.save_pretrained(str(adapter_dir(cfg)))  # writes only the small adapter weights
    return model


def run(cfg: dict[str, Any], skip_lora: bool = False) -> dict[str, Any]:
    set_seed(cfg["seed"])
    s = cfg["summarizer"]
    tokenizer = AutoTokenizer.from_pretrained(s["model"])
    model = AutoModelForCausalLM.from_pretrained(s["model"], dtype=torch.float32).eval()

    # Different seeds -> the eval bursts are never seen in training.
    eval_bursts = make_bursts(cfg, s["eval_bursts"], seed=cfg["seed"] + 1)
    report: dict[str, Any] = {
        "model": s["model"],
        "burst_size": [s["burst_min"], s["burst_max"]],
        "prompt_only": evaluate_model(model, tokenizer, eval_bursts, cfg),
    }
    out = Path(cfg["paths"]["reports_dir"]) / "summarizer.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")  # saved before the long step

    if not skip_lora:
        t0 = time.perf_counter()
        model = train_lora(model, tokenizer, make_bursts(cfg, s["train_bursts"], cfg["seed"]), cfg)
        report["lora_train_seconds"] = round(time.perf_counter() - t0, 1)
        report["lora"] = evaluate_model(model.eval(), tokenizer, eval_bursts, cfg)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 7: burst summarizer (prompting + LoRA).")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--skip-lora", action="store_true", help="Only evaluate prompting")
    args = parser.parse_args(argv)
    report = run(load_config(args.config), skip_lora=args.skip_lora)
    for name in ("prompt_only", "lora"):
        if name in report:
            m = report[name]
            print(
                f"\n[{name}] format={m['format_ok']} count={m['count_ok']} "
                f"category={m['category_ok']} rougeL={m['rougeL']} ({m['seconds_per_burst']}s/burst)"
            )
            print(m["examples"][0]["prediction"])


if __name__ == "__main__":
    main()
