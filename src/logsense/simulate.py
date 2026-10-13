"""Live syslog feed simulator for the console's Live view.

Real syslog is mostly routine ``noise`` with occasional stray events, and incidents arrive as
bursts: one device flapping, one user failing login after login, one scanner hitting a
firewall. This module produces that shape from the Phase 1 templates, with a simulated clock.

It is stateless: the client passes back the clock (``t``) and any open incident, so the
server keeps no per-viewer state.
"""

from __future__ import annotations

import random
import re
from datetime import datetime, timezone
from typing import Any

from logsense.generate import TEMPLATES, render, template_name

_HEADER_HOST = re.compile(r"^([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+)([^\s:]+)")


def _pool(label: str, vendors: set[str], vendor: str | None = None) -> list[tuple[str, Any]]:
    pool = [(v, fn) for v, fn in TEMPLATES[label] if v in vendors]
    if vendor:
        pool = [(v, fn) for v, fn in pool if v == vendor] or pool
    return pool


def feed_chunk(
    cfg: dict[str, Any],
    n: int,
    t: float,
    seed: int,
    incident: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Next ``n`` events after simulated time ``t`` (epoch seconds).

    ``incident`` = {"label", "vendor", "host", "left"} carries an open burst across calls.
    Returns {"events": [...], "t": new clock, "incident": open burst or None}.
    """
    f = cfg["api"]["feed"]
    vendors = set(cfg["data"]["vendors"])
    # Seed from the clock too, so every chunk differs but a replay with the same inputs repeats.
    rng = random.Random(f"{seed}:{t:.3f}")
    incident_labels = [lb for lb in cfg["labels"] if lb != "noise"]
    events = []
    for _ in range(n):
        if incident is None and rng.random() < f["incident_rate"]:
            label = rng.choice(incident_labels)
            vendor = rng.choice(sorted({v for v, _ in _pool(label, vendors)}))
            incident = {
                "label": label,
                "vendor": vendor,
                "host": None,
                "left": rng.randint(f["burst_min"], f["burst_max"]),
            }
        in_burst = incident is not None and rng.random() < 0.8
        if in_burst:
            label, vendor = incident["label"], incident["vendor"]
            t += rng.uniform(0.2, 4.0)  # bursts are dense
        else:
            label = "noise" if rng.random() < f["noise_share"] else rng.choice(incident_labels)
            vendor = None
            t += rng.expovariate(1 / f["mean_gap_s"])
        v, fn = rng.choice(_pool(label, vendors, vendor))
        dt = datetime.fromtimestamp(t, tz=timezone.utc)
        text, _ = render(*fn(rng, dt))
        if in_burst:
            # One incident = one device: pin the sending host for the whole burst.
            m = _HEADER_HOST.match(text)
            if m:
                incident["host"] = incident["host"] or m.group(2)
                text = m.group(1) + incident["host"] + text[m.end() :]
            incident["left"] -= 1
        events.append(
            {
                "ts": round(t, 3),
                "text": text,
                "label": label,  # generator ground truth, for checking the model
                "vendor": v,
                "template": template_name(fn),
                "incident": in_burst,
            }
        )
        if incident is not None and incident["left"] <= 0:
            incident = None
    return {"events": events, "t": round(t, 3), "incident": incident}
