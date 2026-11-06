"""Does Laya's option budget cost accuracy? At the default head_max_len of 192 tokens each of the
ten label descriptions is cut to 17; at 256 none is, but the complaint gets less room.

    python -m src.experiments.head_budget 192 256     # val balanced accuracy per budget
"""
import sys
import time

import pandas as pd
from laya import load

from ..models import system1

if __name__ == "__main__":
    agent = load(system1.MODEL)
    df = pd.read_parquet(system1.PROCESSED_DIR / "val.parquet", columns=["narrative", "canonical_product"])
    for h in map(int, sys.argv[1:] or ["192", "256"]):
        t0 = time.perf_counter()
        pred = [agent.system_one(t, system1.QUESTIONS, head_max_len=h)["answers"]["product"]["choice"] for t in df["narrative"]]
        print(f"head_max_len={h}: val balanced accuracy {(df['canonical_product'] == pred).mean():.4f}, "
              f"{(time.perf_counter() - t0) / len(df):.3f} s/complaint", flush=True)
