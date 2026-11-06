"""Evaluate a fast system, a slow system and the router on saved val/test predictions.

Defaults to Laya -> Sol; --fast / --slow swap in a local model run by src/models/slm.py.
The threshold is picked on val -- the lowest one whose routed accuracy reaches
`target` x slow-alone accuracy -- and test is reported once at that threshold.
Every metric comes in two class mixes: "balanced" (the 200/class eval sets as-is)
and "real" (reweighted to each class's share of the full narrative dataset).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data.preprocess import OUT_PATH
from .data.taxonomy import LABELS
from .router import apply, escalate

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
GRID = np.round(np.arange(0, 1.0001, 0.01), 2)
MIXES = ("real", "balanced")


# name -> (file suffix, prediction column, confidence column); any other name is a
# local model folder run by src/models/slm.py.
SAVED = {"laya": ("system1", "system1_pred", "system1_confidence"), "sol": ("system2", "system2_pred", None)}


def predictions(split: str, name: str) -> pd.DataFrame:
    """complaint_id, pred and (if the system reports one) confidence for one system."""
    file, pred, conf = SAVED.get(name, (name, "slm_pred", "slm_confidence"))
    cols = ["complaint_id", pred] + ([conf] if conf else [])
    return pd.read_parquet(PROCESSED_DIR / f"{split}_{file}.parquet", columns=cols).rename(
        columns={pred: "pred", conf: "confidence"})


def load(split: str, fast: str = "laya", slow: str = "sol") -> pd.DataFrame:
    """Labels joined with the fast system's predictions and confidence and the slow system's predictions."""
    f = predictions(split, fast)
    assert "confidence" in f, f"{fast} reports no confidence, so it can't be the fast system"
    s = predictions(split, slow)[["complaint_id", "pred"]]
    y = pd.read_parquet(PROCESSED_DIR / f"{split}.parquet", columns=["complaint_id", "canonical_product"])
    return (y.merge(f.rename(columns={"pred": "system1_pred", "confidence": "system1_confidence"}), on="complaint_id", validate="1:1")
             .merge(s.rename(columns={"pred": "system2_pred"}), on="complaint_id", validate="1:1"))


def real_mix_weights(y: pd.Series, shares: pd.Series) -> np.ndarray:
    """Row weights that make each class count by its real-world share."""
    return (y.map(shares) / y.map(y.value_counts())).to_numpy()


def accuracy(y, p, w=None) -> float:
    return float(np.average(np.asarray(y) == np.asarray(p), weights=w))


def macro_f1(y, p, w=None, labels=LABELS) -> float:
    y, p = np.asarray(y), np.asarray(p)
    w = np.ones(len(y)) if w is None else np.asarray(w)
    scores = []
    for label in labels:
        tp = w[(p == label) & (y == label)].sum()
        fp = w[(p == label) & (y != label)].sum()
        fn = w[(p != label) & (y == label)].sum()
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(scores))


def ece(confidence, correct, w=None, bins: int = 10) -> float:
    """Expected calibration error over equal-width confidence bins."""
    conf, correct = np.asarray(confidence, float), np.asarray(correct, float)
    w = np.ones(len(conf)) if w is None else np.asarray(w)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if w[m].sum():
            total += w[m].sum() * abs(np.average(correct[m], weights=w[m]) - np.average(conf[m], weights=w[m]))
    return float(total / w.sum())


def choose_threshold(df: pd.DataFrame, w=None, target: float = 0.97) -> float:
    goal = target * accuracy(df["canonical_product"], df["system2_pred"], w)
    return next((float(t) for t in GRID if accuracy(df["canonical_product"], apply(df, t), w) >= goal), float(GRID[-1]))


def report(df: pd.DataFrame, threshold: float, w=None) -> dict:
    y, s1, s2 = df["canonical_product"], df["system1_pred"], df["system2_pred"]
    routed, sent = apply(df, threshold), escalate(df["system1_confidence"], threshold)
    fast_right = (s1 == y).to_numpy()
    return {
        "fast": {"accuracy": accuracy(y, s1, w), "macro_f1": macro_f1(y, s1, w),
                 "ece": ece(df["system1_confidence"], fast_right, w)},
        "slow": {"accuracy": accuracy(y, s2, w), "macro_f1": macro_f1(y, s2, w)},
        "router": {"accuracy": accuracy(y, routed, w), "macro_f1": macro_f1(y, routed, w),
                   "sent_to_slow": float(np.average(sent, weights=w))},
        "perfect_router": {"accuracy": float(np.average(fast_right | (s2 == y).to_numpy(), weights=w)),
                           "sent_to_slow": float(np.average(~fast_right, weights=w))},
    }


def sweep(df: pd.DataFrame, w=None) -> list[dict]:
    return [{"threshold": float(t),
             "sent_to_slow": float(np.average(escalate(df["system1_confidence"], t), weights=w)),
             "accuracy": accuracy(df["canonical_product"], apply(df, t), w)} for t in GRID]


def main(target: float, mix: str, fast: str = "laya", slow: str = "sol") -> dict:
    shares = pd.read_parquet(OUT_PATH, columns=["canonical_product"])["canonical_product"]
    shares = shares[shares.isin(LABELS)].value_counts(normalize=True)

    frames = {split: load(split, fast, slow) for split in ("val", "test")}
    weights = {split: {"real": real_mix_weights(df["canonical_product"], shares), "balanced": None}
               for split, df in frames.items()}

    t = choose_threshold(frames["val"], weights["val"][mix], target)
    results = {
        "threshold": t,
        "fast": fast,
        "slow": slow,
        "picked_on": {"split": "val", "mix": mix, "target": target},
        **{split: {m: report(df, t, weights[split][m]) for m in MIXES} for split, df in frames.items()},
        "sweep": {split: {m: sweep(df, weights[split][m]) for m in MIXES} for split, df in frames.items()},
    }
    default = (fast, slow) == ("laya", "sol")
    out = PROCESSED_DIR / ("eval.json" if default else f"eval_{fast}_to_{slow}.json")
    out.write_text(json.dumps(results, indent=2))

    print(f"{fast} -> {slow}, threshold {t:.2f}: lowest reaching {target:.0%} of {slow}'s val accuracy, {mix} mix\n")
    print(f"{'':14}{'fast':>8}{'slow':>8}{'router':>8}{'to slow':>9}{'perfect':>9}   fast ECE")
    for split in ("val", "test"):
        for m in MIXES:
            r = results[split][m]
            print(f"{split + ' ' + m:14}{r['fast']['accuracy']:8.1%}{r['slow']['accuracy']:8.1%}"
                  f"{r['router']['accuracy']:8.1%}{r['router']['sent_to_slow']:9.1%}"
                  f"{r['perfect_router']['accuracy']:9.1%}   {r['fast']['ece']:.3f}")
    print(f"\nfull results, macro-F1 and threshold sweep -> {out}", file=sys.stderr)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=float, default=0.97, help="share of slow-alone accuracy to reach on val")
    parser.add_argument("--mix", choices=MIXES, default="real", help="class mix used to pick the threshold")
    parser.add_argument("--fast", default="laya", help='fast system: "laya" or a model folder name run by src/models/slm.py')
    parser.add_argument("--slow", default="sol", help='slow system: "sol", "laya" or a model folder name run by src/models/slm.py')
    args = parser.parse_args()
    main(args.target, args.mix, args.fast, args.slow)
