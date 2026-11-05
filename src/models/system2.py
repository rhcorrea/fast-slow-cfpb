"""System 2: GPT-5.6 Sol fallback for narratives System 1 is unsure about.

Slow and expensive relative to Laya, but this is the "System 2" side of the fast/slow
split on purpose -- it's the accuracy ceiling the routing policy buys into when it pays
for a call. Structured outputs (a Literal over the canonical taxonomy) guarantee a valid
label back, so there's no free-text parsing to get wrong.
"""
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from ..data.taxonomy import LABELS
from .system1 import CRITERIA

load_dotenv()

INSTRUCTIONS = "You classify CFPB consumer complaints into one of these product categories:\n\n" + "\n".join(
    f"- {label}: {desc}" for label, desc in CRITERIA.items()
)


class ProductClassification(BaseModel):
    product: Literal[tuple(LABELS)]


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def classify(client: OpenAI, narrative: str) -> str:
    response = client.responses.parse(
        model="gpt-5.6-sol",
        reasoning={"effort": "low"},
        instructions=INSTRUCTIONS,
        input=narrative,
        text_format=ProductClassification,
    )
    return response.output_parsed.product


def run_split(name: str, client: OpenAI, n: int | None = None) -> Path:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}_system1.parquet")
    if n is not None:
        per_class = max(1, n // len(LABELS))
        df = pd.concat(
            [g.sample(min(len(g), per_class), random_state=42) for _, g in df.groupby("canonical_product")],
            ignore_index=True,
        )

    df["system2_pred"] = [classify(client, t) for t in df["narrative"]]

    out = PROCESSED_DIR / f"{name}_system2{'_pilot' if n else ''}.parquet"
    df.to_parquet(out, index=False)

    acc = (df["system2_pred"] == df["canonical_product"]).mean()
    print(f"{name}: n={len(df)} accuracy={acc:.3f} -> {out}", file=sys.stderr)
    return out


if __name__ == "__main__":
    client = OpenAI()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    splits = ("val",) if n else ("val", "test")  # pilot mode: val only
    for split_name in splits:
        run_split(split_name, client, n=n)
