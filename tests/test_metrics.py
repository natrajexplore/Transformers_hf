from logsense.metrics import classification_metrics


def test_known_values():
    labels = ["a", "b", "c"]
    y_true = ["a", "a", "b", "c"]
    y_pred = ["a", "b", "b", "c"]
    m = classification_metrics(y_true, y_pred, labels)
    assert m["n"] == 4
    assert m["accuracy"] == 0.75
    assert m["per_class_f1"]["c"] == 1.0
    assert abs(m["per_class_f1"]["a"] - 2 / 3) < 1e-9
    assert m["confusion_matrix"]["matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_missing_class_scores_zero():
    m = classification_metrics(["a", "b"], ["a", "a"], ["a", "b", "c"])
    assert m["per_class_f1"]["b"] == 0.0
    assert m["per_class_f1"]["c"] == 0.0
