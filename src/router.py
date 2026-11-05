"""Fast/slow router: keep Laya's label when it's confident, otherwise ask Sol.

`apply` routes saved System 1/System 2 predictions (what src/eval.py sweeps).
`Router` does the same live, calling Sol only for complaints that need it.
"""
import sys

import numpy as np
import pandas as pd

# Lowest threshold reaching 97% of Sol's val accuracy on the real class mix; rerun
# `python -m src.eval` to re-pick it after the taxonomy or either model changes.
THRESHOLD = 0.41


def escalate(confidence, threshold: float = THRESHOLD) -> np.ndarray:
    return np.asarray(confidence) < threshold


def apply(df: pd.DataFrame, threshold: float = THRESHOLD) -> pd.Series:
    """Routed labels from a frame with system1_pred, system1_confidence and system2_pred."""
    return df["system2_pred"].where(escalate(df["system1_confidence"], threshold), df["system1_pred"])


class Router:
    def __init__(self, threshold: float = THRESHOLD):
        from laya import load
        from openai import OpenAI

        from .models import system1

        self.agent = load(system1.MODEL)
        self.client = OpenAI()
        self.threshold = threshold

    def classify(self, narrative: str) -> dict:
        from .models import system1, system2

        ans = system1.classify(self.agent, narrative)
        if not escalate(ans["confidence"], self.threshold):
            return {"label": ans["choice"], "system": 1, "confidence": ans["confidence"]}
        return {"label": system2.classify(self.client, narrative), "system": 2, "confidence": ans["confidence"]}


if __name__ == "__main__":
    router = Router()
    for text in sys.argv[1:]:
        print(router.classify(text))
