"""Fast/slow router: keep the fast system's label when it's confident, otherwise ask Sol.

`apply` routes saved System 1/System 2 predictions (what src/eval.py sweeps).
`Router` does the same live, calling Sol only for complaints that need it. The fast
system is Laya, the supervised TF-IDF model (src/models/tfidf.py) or a local model run
by src/models/slm.py:

    python -m src.router "narrative..."                          # Laya -> Sol
    python -m src.router --fast tfidf "narrative..."             # TF-IDF -> Sol
    python -m src.router --fast qwen3.5-4b-4bit "narrative..."   # Qwen -> Sol
"""
import argparse

import numpy as np
import pandas as pd

# Per fast system: the lowest threshold reaching 97% of Sol's val accuracy on the real
# class mix. Rerun `python -m src.eval [--fast NAME]` to re-pick after the taxonomy or
# a model changes. TF-IDF alone already beats Sol on val, which makes that rule pick 0
# (never ask Sol), so its threshold is the one maximizing val real-mix routed accuracy
# (the sweep in eval_tfidf_to_sol.json): 0.61, sending ~10% to Sol.
THRESHOLDS = {"laya": 0.41, "qwen3.5-4b-4bit": 0.84, "tfidf": 0.61}
THRESHOLD = THRESHOLDS["laya"]


def escalate(confidence, threshold: float = THRESHOLD) -> np.ndarray:
    return np.asarray(confidence) < threshold


def apply(df: pd.DataFrame, threshold: float = THRESHOLD) -> pd.Series:
    """Routed labels from a frame with system1_pred, system1_confidence and system2_pred."""
    return df["system2_pred"].where(escalate(df["system1_confidence"], threshold), df["system1_pred"])


class Router:
    def __init__(self, fast: str = "laya", threshold: float | None = None):
        from openai import OpenAI

        from .models import system2  # loads .env, so OpenAI() finds the key

        if threshold is None:
            assert fast in THRESHOLDS, f"no threshold picked for {fast}: run python -m src.eval --fast {fast}"
            threshold = THRESHOLDS[fast]
        if fast == "laya":
            from laya import load

            from .models import system1

            agent = load(system1.MODEL)
            self.fast = lambda text: system1.classify(agent, text)
        elif fast in ("tfidf", "tfidf-balanced"):
            from .models import tfidf

            model = tfidf.load_model()
            self.fast = lambda text: tfidf.classify(model, text, real_prior=fast == "tfidf")
        else:
            from .models import slm

            model = slm.load_model(fast)
            self.fast = lambda text: slm.classify(model, text)
        self.client = OpenAI()
        self.threshold = threshold

    def classify(self, narrative: str) -> dict:
        from .models import system2

        ans = self.fast(narrative)
        if not escalate(ans["confidence"], self.threshold):
            return {"label": ans["choice"], "system": 1, "confidence": ans["confidence"]}
        return {"label": system2.classify(self.client, narrative), "system": 2, "confidence": ans["confidence"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", default="laya", help='"laya" or a model folder name run by src/models/slm.py')
    parser.add_argument("--threshold", type=float, help="override the picked threshold")
    parser.add_argument("narratives", nargs="+")
    args = parser.parse_args()
    router = Router(args.fast, args.threshold)
    for text in args.narratives:
        print(router.classify(text))
