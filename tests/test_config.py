from logsense.config import label_maps, load_config


def test_config_loads():
    cfg = load_config()
    assert "auth_failure" in cfg["labels"]
    assert cfg["models"]["classifier"]


def test_label_maps_roundtrip():
    cfg = load_config()
    label2id, id2label = label_maps(cfg)
    assert len(label2id) == len(cfg["labels"])
    for name, idx in label2id.items():
        assert id2label[idx] == name
