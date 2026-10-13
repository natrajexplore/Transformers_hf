from logsense.config import load_config
from logsense.simulate import feed_chunk

T0 = 1_760_000_000.0


def test_feed_is_time_ordered_and_reproducible():
    cfg = load_config()
    a = feed_chunk(cfg, 20, T0, seed=1)
    b = feed_chunk(cfg, 20, T0, seed=1)
    assert a == b
    ts = [e["ts"] for e in a["events"]]
    assert ts == sorted(ts) and ts[0] > T0 and a["t"] == ts[-1]
    assert all(e["label"] in cfg["labels"] for e in a["events"])


def test_open_incident_continues_on_one_host():
    cfg = load_config()
    inc = {"label": "link_event", "vendor": "cisco_ios", "host": None, "left": 6}
    r = feed_chunk(cfg, 30, T0, seed=2, incident=inc)
    burst = [e for e in r["events"] if e["incident"]][:6]  # a later, new burst may follow
    assert len(burst) == 6 and {e["label"] for e in burst} == {"link_event"}
    assert len({e["text"].split()[3] for e in burst}) == 1  # one device for the whole burst


def test_mostly_noise():
    cfg = load_config()
    labels = [e["label"] for e in feed_chunk(cfg, 300, T0, seed=3)["events"]]
    assert labels.count("noise") / len(labels) > 0.4
