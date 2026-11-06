"""Supervised fast system: TF-IDF + logistic regression trained on train.parquet.

Unlike Laya and the local SLMs this one learns from labelled complaints, which is most
of what it takes on this task: trained in minutes, ~0.1 ms per complaint, and on the
real class mix close to Sol on its own.

It trains on up to `n_per_class` complaints per class with class weights balancing the
classes, so its raw probabilities assume a uniform prior. Two prediction sets are saved
in the slm format (slm_pred / slm_confidence / slm_probabilities) so `src.eval --fast`
and `src.router --fast` take either name:

    tfidf           probabilities reweighted to the real class shares in train.parquet
                    (the one for routing real traffic)
    tfidf-balanced  the raw, uniform-prior probabilities

    python -m src.models.tfidf              # train, save models/tfidf.joblib, predict val + test
    python -m src.models.tfidf 8000 10      # n_per_class, C

Defaults were picked on val only (test untouched): 8,000 or 30,000 per class x C in
{1, 3, 10, 30}. 30,000 x C=3 was best on real-mix accuracy (0.898) and balanced accuracy
(0.808); larger C overfits, and 30,000 per class is ~280k complaints, as far as building
the bigram vocabulary goes comfortably in 16 GB.
"""
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..data.taxonomy import LABELS
from .system1 import MODELS_DIR, PROCESSED_DIR

MODEL_PATH = MODELS_DIR / "tfidf.joblib"


N_PER_CLASS, C = 30_000, 3.0


def train(n_per_class: int = N_PER_CLASS, c: float = C, seed: int = 0) -> dict:
    tr = pd.read_parquet(PROCESSED_DIR / "train.parquet", columns=["complaint_id", "narrative", "canonical_product"])
    tr = tr[tr["canonical_product"].isin(LABELS)]
    held_out = set().union(*(pd.read_parquet(PROCESSED_DIR / f"{s}.parquet", columns=["complaint_id"])["complaint_id"]
                             for s in ("val", "test")))
    assert held_out.isdisjoint(tr["complaint_id"]), "train.parquet overlaps the eval sets"
    prior = tr["canonical_product"].value_counts(normalize=True)
    sample = pd.concat([g.sample(min(len(g), n_per_class), random_state=seed) for _, g in tr.groupby("canonical_product")])

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True, dtype=np.float32)
    clf = LogisticRegression(C=c, max_iter=1000, class_weight="balanced")
    clf.fit(vec.fit_transform(sample["narrative"]), sample["canonical_product"])
    return {"vec": vec, "clf": clf, "prior": prior.reindex(clf.classes_).to_numpy(),
            "n_per_class": n_per_class, "c": c, "n_train": len(sample)}


def load_model(path=MODEL_PATH) -> dict:
    return joblib.load(path)


def probabilities(model: dict, texts, real_prior: bool = True) -> np.ndarray:
    """Rows of per-class probabilities in model['clf'].classes_ order."""
    p = model["clf"].predict_proba(model["vec"].transform(texts))
    if real_prior:
        p = p * model["prior"]
        p /= p.sum(axis=1, keepdims=True)
    return p


def classify(model: dict, narrative: str, real_prior: bool = True) -> dict:
    """Same shape as system1.classify / slm.classify, so the router can use it."""
    p = probabilities(model, [narrative], real_prior)[0]
    classes = list(model["clf"].classes_)
    best = int(p.argmax())
    return {"choice": classes[best], "confidence": float(p[best]), "probabilities": dict(zip(classes, map(float, p)))}


def predict_split(model: dict, split: str) -> None:
    df = pd.read_parquet(PROCESSED_DIR / f"{split}.parquet", columns=["complaint_id", "narrative", "canonical_product"])
    classes = list(model["clf"].classes_)
    for name, real_prior in (("tfidf", True), ("tfidf-balanced", False)):
        p = probabilities(model, df["narrative"], real_prior)
        out = df[["complaint_id", "canonical_product"]].copy()
        out["slm_pred"] = [classes[i] for i in p.argmax(axis=1)]
        out["slm_confidence"] = p.max(axis=1)
        out["slm_probabilities"] = [dict(zip(classes, map(float, row))) for row in p]
        out.to_parquet(PROCESSED_DIR / f"{split}_{name}.parquet", index=False)
        acc = (out["slm_pred"] == out["canonical_product"]).mean()
        print(f"{split}_{name}: balanced accuracy {acc:.3f}", file=sys.stderr)


if __name__ == "__main__":
    args = [float(a) for a in sys.argv[1:]]
    t0 = time.perf_counter()
    model = train(int(args[0]) if args else N_PER_CLASS, args[1] if len(args) > 1 else C)
    print(f"trained on {model['n_train']:,} complaints (n_per_class={model['n_per_class']}, C={model['c']}) "
          f"in {time.perf_counter() - t0:.0f}s", file=sys.stderr)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    for split_name in ("val", "test"):
        predict_split(model, split_name)
