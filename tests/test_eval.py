from __future__ import annotations

from scripts.evaluate_ner import score


def test_evaluate_aligns_by_text_when_present() -> None:
    gold = [
        {"text": "Hello", "entities": [{"label": "ORG", "text": "Hello", "start": 0, "end": 5}]},
        {"text": "World", "entities": [{"label": "ORG", "text": "World", "start": 0, "end": 5}]},
    ]
    pred = [
        {"text": "world", "entities": [{"label": "ORG", "text": "World", "start": 0, "end": 5}]},
        {"text": "hello", "entities": [{"label": "ORG", "text": "Hello", "start": 0, "end": 5}]},
    ]

    out = score(gold, pred)
    assert out["relaxed"]["micro"]["f1"] == 1.0
    assert out["strict"]["micro"]["f1"] == 1.0

