"""Stratified train/val/test split over the canonical Product taxonomy.

200/class held out for val, 200/class for test -- enough for stable accuracy/F1 without
running System 2 (LLM) over huge eval sets. Everything else lands in train, unused by
V1's zero-shot pipeline but reserved in case Laya gets fine-tuned later.
"""
import sys
from pathlib import Path

import pandas as pd

from .preprocess import OUT_PATH
from .taxonomy import LABELS

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
SEED = 42
N_VAL = 200
N_TEST = 200


def split() -> dict[str, Path]:
    df = pd.read_parquet(OUT_PATH)
    df = df[df["canonical_product"].isin(LABELS)]

    val_parts, test_parts, train_parts = [], [], []
    for _, g in df.groupby("canonical_product"):
        g = g.sample(frac=1, random_state=SEED)
        val_parts.append(g.iloc[:N_VAL])
        test_parts.append(g.iloc[N_VAL : N_VAL + N_TEST])
        train_parts.append(g.iloc[N_VAL + N_TEST :])

    out = {}
    for name, parts in [("train", train_parts), ("val", val_parts), ("test", test_parts)]:
        split_df = pd.concat(parts, ignore_index=True)
        path = PROCESSED_DIR / f"{name}.parquet"
        split_df.to_parquet(path, index=False)
        out[name] = path
    return out


if __name__ == "__main__":
    paths = split()
    for name, path in paths.items():
        n = len(pd.read_parquet(path, columns=["complaint_id"]))
        print(f"{name}: {n} rows -> {path}", file=sys.stderr)
