"""Test & performance console: a FastAPI backend plus a static web page for the trained models.

    python -m logsense.api          # -> http://127.0.0.1:8000

The page (``src/logsense/web/``) calls the JSON endpoints below; anything else (curl, n8n,
scripts) can call them too.

Concepts worth knowing when you measure transformer models
----------------------------------------------------------
* Load cost vs inference cost — ``from_pretrained`` reads hundreds of MB of weights from disk
  once; each later call only runs the forward pass. The first request is therefore much slower,
  so models load lazily on first use and the benchmark does an untimed warm-up batch.
* Batching — a forward pass over 16 lines costs far less than 16 single-line passes, because the
  matrix multiplications are shared and padded together. ``batch_size`` in the benchmark shows
  that trade-off between latency (time per request) and throughput (lines per second).
* PEFT adapters can be switched off — the summarizer loads the Qwen base model once with the LoRA
  adapter on top; ``model.disable_adapter()`` temporarily runs the plain base model, so prompt-only
  and LoRA can be compared without loading two copies of the LLM.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from logsense import evaluate_file as ef
from logsense.anonymize import anonymize_lines
from logsense.config import load_config
from logsense.metrics import classification_metrics
from logsense.search import find_similar_batch, index_dir
from logsense.simulate import feed_chunk
from logsense.summarize import adapter_dir, make_bursts, reference_summary, score
from logsense.train_classifier import model_dir as classifier_dir
from logsense.train_classifier import predict
from logsense.train_ner import extract_entities
from logsense.train_ner import model_dir as ner_dir

WEB_DIR = Path(__file__).parent / "web"
REPORTS = ("baseline", "classifier", "ner", "search", "summarizer")
FAST_MODELS = ("classifier", "ner", "search")


class AnalyzeIn(BaseModel):
    lines: list[str] = Field(min_length=1)
    k: int = Field(5, ge=0, le=20)  # 0 = skip similar-event search (Live view)


class LinesIn(BaseModel):
    lines: list[str] = Field(min_length=1)


class FeedIn(BaseModel):
    n: int = Field(8, ge=1, le=50)
    t: float | None = None  # simulated clock from the previous call; None = now
    seed: int = 0
    incident: dict[str, Any] | None = None  # open burst returned by the previous call


class BenchmarkIn(BaseModel):
    split: Literal["test", "test_unseen"] = "test_unseen"
    n_lines: int = Field(100, ge=1)
    batch_size: int = Field(16, ge=1, le=128)
    models: list[Literal["classifier", "ner", "search"]] = list(FAST_MODELS)


class SummarizeIn(BaseModel):
    lines: list[str] = Field(min_length=1)
    mode: Literal["lora", "prompt"] = "lora"
    facts: dict[str, Any] | None = None  # from /api/sample-burst: lets us score the summary


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def _memory_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return round(psutil.Process().memory_info().rss / 2**20, 1)


def _size_mb(path: Path) -> float:
    files = [path] if path.is_file() else path.rglob("*")
    return round(sum(f.stat().st_size for f in files if f.is_file()) / 2**20, 1)


def sample_rows(
    cfg: dict[str, Any], split: str, n: int, seed: int = 0
) -> tuple[list[str], list[str]]:
    """``n`` random (text, label) rows from a saved dataset split — labeled test data."""
    from datasets import load_from_disk

    path = Path(cfg["data"]["processed_dir"])
    if not path.exists():
        raise HTTPException(404, f"{path} not found; run python -m logsense.generate first")
    ds = load_from_disk(str(path))[split]
    idx = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
    rows = ds.select(idx)
    names = ds.features["label"].names
    return list(rows["text"]), [names[i] for i in rows["label"]]


class Summarizer:
    """Qwen base model + LoRA adapter, loaded once. Generation is serialized by a lock."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = cfg["summarizer"]["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
        self.has_adapter = adapter_dir(cfg).exists()
        if self.has_adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir(cfg)))
        self.model = model.eval()

    def summarize(self, lines: list[str], use_lora: bool, max_new_tokens: int) -> str:
        from logsense.summarize import generate_summary

        if use_lora and not self.has_adapter:
            raise HTTPException(409, "No LoRA adapter in models/summarizer_lora; run summarize")
        if self.has_adapter and not use_lora:
            with self.model.disable_adapter():  # plain base model, same weights in memory
                return generate_summary(self.model, self.tokenizer, lines, max_new_tokens)
        return generate_summary(self.model, self.tokenizer, lines, max_new_tokens)


def create_app(cfg: dict[str, Any] | None = None) -> FastAPI:
    cfg = cfg or load_config()
    limits = cfg["api"]
    app = FastAPI(title="LogSense console", docs_url="/api/docs", openapi_url="/api/openapi.json")
    first_call: dict[str, float] = {}  # model -> seconds of its first call (load + run)
    summarizer: list[Summarizer] = []  # filled on first /api/summarize
    llm_lock, bench_lock = threading.Lock(), threading.Lock()

    def timed(name: str, fn: Callable[[], Any]) -> tuple[Any, float]:
        t0 = time.perf_counter()
        out = fn()
        seconds = time.perf_counter() - t0
        first_call.setdefault(name, round(seconds, 2))
        return out, seconds

    def clean(lines: list[str], limit: int) -> list[str]:
        lines = [ln.strip() for ln in lines if ln.strip()]
        if not lines:
            raise HTTPException(422, "no non-empty lines")
        if len(lines) > limit:
            raise HTTPException(422, f"at most {limit} lines per request")
        return lines

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "loaded": sorted(first_call)}

    @app.post("/api/analyze")
    def analyze(body: AnalyzeIn) -> dict[str, Any]:
        lines = clean(body.lines, limits["max_lines"])
        preds, t_cls = timed("classifier", lambda: predict(lines, cfg=cfg))
        ents, t_ner = timed("ner", lambda: [extract_entities(ln, cfg=cfg) for ln in lines])
        sims, t_sim = [[] for _ in lines], 0.0
        if body.k:
            sims, t_sim = timed("search", lambda: find_similar_batch(lines, k=body.k, cfg=cfg))
        return {
            "results": [
                {
                    "line": ln,
                    "category": p["label"],
                    "confidence": round(p["score"], 4),
                    "entities": e,
                    "similar": s,
                }
                for ln, p, e, s in zip(lines, preds, ents, sims, strict=True)
            ],
            "timing_ms": {
                "classifier": round(t_cls * 1000, 1),
                "ner": round(t_ner * 1000, 1),
                "search": round(t_sim * 1000, 1),
            },
        }

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        """What the page needs to render: labels, entity types, severity, alert rules."""
        return {
            "labels": cfg["labels"],
            "entity_types": cfg["entity_types"],
            "severity": limits["severity"],
            "alerts": limits["alerts"],
            "limits": {k: limits[k] for k in ("max_lines", "max_burst_lines", "max_eval_lines")},
        }

    @app.post("/api/anonymize")
    def anonymize(body: LinesIn) -> dict[str, Any]:
        lines = [ln.rstrip() for ln in body.lines if ln.strip()]
        if len(lines) > limits["max_eval_lines"]:
            raise HTTPException(422, f"at most {limits['max_eval_lines']} lines")
        return anonymize_lines(lines)

    @app.post("/api/feed")
    def feed(body: FeedIn) -> dict[str, Any]:
        return feed_chunk(cfg, body.n, body.t or time.time(), body.seed, body.incident)

    @app.get("/api/samples")
    def samples(
        split: Literal["test", "test_unseen"] = "test_unseen",
        n: int = Query(5, ge=1, le=50),
        seed: int = 0,
    ) -> dict[str, Any]:
        texts, labels = sample_rows(cfg, split, n, seed)
        return {"split": split, "rows": [{"text": t, "label": lb} for t, lb in zip(texts, labels)]}

    @app.post("/api/benchmark")
    def benchmark(body: BenchmarkIn) -> dict[str, Any]:
        n = min(body.n_lines, limits["benchmark_max_lines"])
        texts, gold = sample_rows(cfg, body.split, n)
        runners: dict[str, Callable[[list[str]], Any]] = {
            "classifier": lambda b: predict(b, cfg=cfg),
            "ner": lambda b: [extract_entities(t, cfg=cfg) for t in b],  # pipeline is per line
            "search": lambda b: find_similar_batch(b, cfg=cfg),
        }
        batches = [texts[i : i + body.batch_size] for i in range(0, len(texts), body.batch_size)]
        results = []
        with bench_lock:  # one benchmark at a time, or the numbers measure each other
            for name in dict.fromkeys(body.models):
                run = runners[name]
                timed(name, lambda r=run: r(batches[0]))  # warm-up: loads weights, not timed
                times, outputs = [], []
                for batch in batches:
                    t0 = time.perf_counter()
                    outputs.append(run(batch))
                    times.append(time.perf_counter() - t0)
                total = sum(times)
                row = {
                    "model": name,
                    "lines": len(texts),
                    "batches": len(batches),
                    "total_s": round(total, 3),
                    "lines_per_s": round(len(texts) / total, 1) if total else None,
                    "ms_per_line": round(total / len(texts) * 1000, 2),
                    "batch_ms": {
                        "p50": round(_percentile(times, 0.5) * 1000, 1),
                        "p95": round(_percentile(times, 0.95) * 1000, 1),
                        "max": round(max(times) * 1000, 1),
                    },
                }
                if name == "classifier":  # the sample is labeled, so report accuracy too
                    preds = [p["label"] for out in outputs for p in out]
                    present = [lb for lb in cfg["labels"] if lb in set(gold) | set(preds)]
                    m = classification_metrics(gold, preds, present)
                    row["accuracy"] = round(m["accuracy"], 3)
                    row["macro_f1"] = round(m["macro_f1"], 3)
                results.append(row)
        return {
            "split": body.split,
            "batch_size": body.batch_size,
            "results": results,
            "memory_mb": _memory_mb(),
        }

    @app.post("/api/evaluate")
    def evaluate(file: Annotated[UploadFile, File()]) -> dict[str, Any]:
        try:
            raw = file.file.read().decode("utf-8-sig")
            texts, gold = ef.parse_labeled(raw.splitlines(), cfg["labels"])
        except (UnicodeDecodeError, ValueError) as err:
            raise HTTPException(422, str(err)) from err
        if not texts:
            raise HTTPException(422, "file has no rows")
        if len(texts) > limits["max_eval_lines"]:
            raise HTTPException(422, f"at most {limits['max_eval_lines']} rows")
        metrics, preds = ef.score_labeled(texts, gold, cfg)
        mistakes = [
            {"text": t, "label": g, "predicted": p}
            for t, g, p in zip(texts, gold, preds, strict=True)
            if g != p
        ]
        return {"filename": file.filename, **metrics, "mistakes": mistakes[:200]}

    @app.get("/api/sample-burst")
    def sample_burst(seed: int | None = None) -> dict[str, Any]:
        seed = random.randrange(10**6) if seed is None else seed
        burst = make_bursts(cfg, 1, seed=seed)[0]
        return {**burst, "seed": seed, "reference": reference_summary(burst["facts"], cfg)}

    @app.post("/api/summarize")
    def summarize(body: SummarizeIn) -> dict[str, Any]:
        lines = clean(body.lines, limits["max_burst_lines"])
        with llm_lock:  # one generation at a time: the LLM is the memory-heavy part
            if not summarizer:
                (s, _) = timed("summarizer", lambda: Summarizer(cfg))
                summarizer.append(s)
            t0 = time.perf_counter()
            text = summarizer[0].summarize(
                lines, body.mode == "lora", cfg["summarizer"]["max_new_tokens"]
            )
            seconds = round(time.perf_counter() - t0, 1)
        # Facts computed by the classifier, to compare against what the LLM claims.
        counts = Counter(p["label"] for p in predict(lines, cfg=cfg))
        out: dict[str, Any] = {
            "summary": text,
            "mode": body.mode,
            "seconds": seconds,
            "classifier_counts": dict(counts.most_common()),
        }
        if body.facts:
            out["checks"] = score(text, body.facts, cfg)
        return out

    @app.get("/api/metrics")
    def metrics() -> dict[str, Any]:
        reports_dir = Path(cfg["paths"]["reports_dir"])
        out = {}
        for name in REPORTS:
            path = reports_dir / f"{name}.json"
            out[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        return out

    @app.get("/api/models")
    def models() -> dict[str, Any]:
        entries = [
            ("classifier", "text classification", cfg["models"]["classifier"], classifier_dir(cfg)),
            ("ner", "token classification (BIO)", cfg["models"]["ner"], ner_dir(cfg)),
            ("search", "sentence embeddings + FAISS", cfg["models"]["embeddings"], index_dir(cfg)),
            (
                "summarizer",
                "causal LM + LoRA adapter",
                cfg["summarizer"]["model"],
                adapter_dir(cfg),
            ),
        ]
        return {
            "models": [
                {
                    "name": name,
                    "task": task,
                    "base_model": base,
                    "path": str(path),
                    "exists": path.exists(),
                    "local_size_mb": _size_mb(path) if path.exists() else None,
                    "first_call_s": first_call.get(name),
                }
                for name, task, base, path in entries
            ],
            "labels": cfg["labels"],
            "entity_types": cfg["entity_types"],
            "memory_mb": _memory_mb(),
        }

    # Mounted last so /api/* routes win; html=True serves index.html at "/".
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the LogSense test & performance console.")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--host", default=None, help="Default from config (127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Default from config (8000)")
    args = parser.parse_args(argv)

    import uvicorn

    cfg = load_config(args.config)
    host, port = args.host or cfg["api"]["host"], args.port or cfg["api"]["port"]
    print(f"LogSense console: http://{host}:{port}  (API docs: /api/docs)")
    uvicorn.run(create_app(cfg), host=host, port=port)


if __name__ == "__main__":
    main()
