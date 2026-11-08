"""Checks for the router and metric math on hand-sized inputs.

Run: uv run python -m tests.test_eval
"""
import pandas as pd

from src.eval import accuracy, choose_threshold, ece, macro_f1
from src.router import apply


def test_metrics():
    y = pd.Series(["a", "a", "b", "b"])
    p = pd.Series(["a", "b", "b", "b"])
    assert accuracy(y, p) == 0.75
    assert accuracy(y, p, w=[0, 0, 1, 1]) == 1.0
    # a: tp=1 fp=0 fn=1 -> 2/3;  b: tp=2 fp=1 fn=0 -> 4/5
    assert abs(macro_f1(y, p, labels=["a", "b"]) - (2 / 3 + 4 / 5) / 2) < 1e-12
    # one bin at 0.9 confidence, half right -> |0.5 - 0.9|
    assert abs(ece([0.9, 0.9], [1, 0]) - 0.4) < 1e-12


def test_router():
    df = pd.DataFrame({
        "canonical_product": ["x", "y", "z"],
        "system1_pred": ["q", "y", "q"],
        "system1_confidence": [0.2, 0.9, 0.6],
        "system2_pred": ["x", "q", "z"],
    })
    assert apply(df, 0.5).tolist() == ["x", "y", "q"]
    assert apply(df, 0.7).tolist() == ["x", "y", "z"]
    # Sol alone gets 2/3; 0.21 is the first grid step that escalates the 0.2 row and
    # reaches 2/3 (x, y right), so it beats escalating everything.
    assert choose_threshold(df, target=1.0) == 0.21


def test_system1_confidence_is_answer_probability():
    from src.models import system1

    class FakeAgent:  # Laya's choice answer: 'confidence' is entropy, 'answer_confidence' is max(p)
        def system_one(self, text, questions):
            return {"answers": {"product": {"choice": "a", "confidence": 0.3, "answer_confidence": 0.8,
                                            "probabilities": {"a": 0.8, "b": 0.2}}}}

    assert system1.classify(FakeAgent(), "text")["confidence"] == 0.8


if __name__ == "__main__":
    test_metrics()
    test_router()
    test_system1_confidence_is_answer_probability()
    print("ok")
