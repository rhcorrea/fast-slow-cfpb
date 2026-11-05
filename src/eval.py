"""Evaluate Laya, Sol and the router on saved val/test predictions.

The threshold is picked on val -- the lowest one whose routed accuracy reaches
`target` x Sol-alone accuracy -- and test is reported once at that threshold.
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


def load(split: str, slow: str = "sol") -> pd.DataFrame:
    """Laya's predictions joined with the slow system's: Sol, or a local model from src/models/slm.py."""
    s1 = pd.read_parquet(
        PROCESSED_DIR / f"{split}_system1.parquet",
        columns=["complaint_id", "canonical_product", "system1_pred", "system1_confidence"],
    )
    if slow == "sol":
        s2 = pd.read_parquet(PROCESSED_DIR / f"{split}_system2.parquet", columns=["complaint_id", "system2_pred"])
    else:
        s2 = pd.read_parquet(PROCESSED_DIR / f"{split}_{slow}.parquet", columns=["complaint_id", "slm_pred"])
        s2 = s2.rename(columns={"slm_pred": "system2_pred"})
    return s1.merge(s2, on="complaint_id", validate="1:1")


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
    laya_right = (s1 == y).to_numpy()
    return {
        "laya": {"accuracy": accuracy(y, s1, w), "macro_f1": macro_f1(y, s1, w),
                 "ece": ece(df["system1_confidence"], laya_right, w)},
        "sol": {"accuracy": accuracy(y, s2, w), "macro_f1": macro_f1(y, s2, w)},
        "router": {"accuracy": accuracy(y, routed, w), "macro_f1": macro_f1(y, routed, w),
                   "sent_to_sol": float(np.average(sent, weights=w))},
        "perfect_router": {"accuracy": float(np.average(laya_right | (s2 == y).to_numpy(), weights=w)),
                           "sent_to_sol": float(np.average(~laya_right, weights=w))},
    }


def sweep(df: pd.DataFrame, w=None) -> list[dict]:
    return [{"threshold": float(t),
             "sent_to_sol": float(np.average(escalate(df["system1_confidence"], t), weights=w)),
             "accuracy": accuracy(df["canonical_product"], apply(df, t), w)} for t in GRID]


def main(target: float, mix: str, slow: str = "sol") -> dict:
    shares = pd.read_parquet(OUT_PATH, columns=["canonical_product"])["canonical_product"]
    shares = shares[shares.isin(LABELS)].value_counts(normalize=True)

    frames = {split: load(split, slow) for split in ("val", "test")}
    weights = {split: {"real": real_mix_weights(df["canonical_product"], shares), "balanced": None}
               for split, df in frames.items()}

    t = choose_threshold(frames["val"], weights["val"][mix], target)
    results = {
        "threshold": t,
        "slow": slow,
        "picked_on": {"split": "val", "mix": mix, "target": target},
        **{split: {m: report(df, t, weights[split][m]) for m in MIXES} for split, df in frames.items()},
        "sweep": {split: {m: sweep(df, weights[split][m]) for m in MIXES} for split, df in frames.items()},
    }
    out = PROCESSED_DIR / ("eval.json" if slow == "sol" else f"eval_{slow}.json")
    out.write_text(json.dumps(results, indent=2))

    print(f"threshold {t:.2f}: lowest reaching {target:.0%} of {slow}'s val accuracy, {mix} mix\n")
    print(f"{'':14}{'laya':>8}{'slow':>8}{'router':>8}{'to slow':>9}{'perfect':>9}   laya ECE")
    for split in ("val", "test"):
        for m in MIXES:
            r = results[split][m]
            print(f"{split + ' ' + m:14}{r['laya']['accuracy']:8.1%}{r['sol']['accuracy']:8.1%}"
                  f"{r['router']['accuracy']:8.1%}{r['router']['sent_to_sol']:9.1%}"
                  f"{r['perfect_router']['accuracy']:9.1%}   {r['laya']['ece']:.3f}")
    print(f"\nfull results, macro-F1 and threshold sweep -> {out}", file=sys.stderr)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=float, default=0.97, help="share of Sol-alone accuracy to reach on val")
    parser.add_argument("--mix", choices=MIXES, default="real", help="class mix used to pick the threshold")
    parser.add_argument("--slow", default="sol", help='slow system: "sol" or a model folder name run by src/models/slm.py')
    args = parser.parse_args()
    main(args.target, args.mix, args.slow)
