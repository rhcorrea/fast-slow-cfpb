"""System 1: Laya zero-shot classification over the canonical Product taxonomy.

One `choice` question, ten canonical labels (src/data/taxonomy.py). No fine-tuning --
Laya is a pretrained decision model, so this is straight zero-shot inference plus its
built-in calibrated confidence, which is what the fast/slow routing policy will threshold
on later.
"""
import sys
from pathlib import Path

import pandas as pd
from laya import load

from ..data.taxonomy import LABELS

CRITERIA = {
    "Credit reporting": "errors on a credit report, disputes with a credit bureau, or identity theft on a credit file",
    "Debt collection": "being contacted by a debt collector or disputing a collection attempt",
    "Checking or savings account": "a checking or savings account: deposits, fees, or account access",
    "Mortgage": "a home loan or mortgage: servicing, foreclosure, or escrow",
    "Credit card or prepaid card": "a credit card or prepaid card: charges, fees, or rewards",
    "Money transfer, virtual currency, or money service": "sending money, a money transfer service, virtual currency, or a money order",
    "Student loan": "a federal or private student loan: repayment or servicing",
    "Vehicle loan or lease": "an auto loan or vehicle lease",
    "Payday loan, title loan, personal loan, or advance loan": "a payday loan, title loan, personal loan, or cash advance",
    "Debt or credit management": "debt settlement, credit counseling, or credit repair services",
}
assert set(CRITERIA) == set(LABELS), "CRITERIA must cover exactly the canonical taxonomy"

QUESTIONS = {
    "product": {
        "type": "choice",
        "instructions": "Which CFPB product category does this complaint concern?",
        "criteria": CRITERIA,
    }
}

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def run_split(name: str, agent) -> Path:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    preds, confs, probs = [], [], []
    for narrative in df["narrative"]:
        ans = agent.system_one(narrative, QUESTIONS)["answers"]["product"]
        preds.append(ans["choice"])
        confs.append(ans["confidence"])
        probs.append(ans["probabilities"])

    df["system1_pred"] = preds
    df["system1_confidence"] = confs
    df["system1_probabilities"] = probs
    assert set(preds) <= set(LABELS)

    out = PROCESSED_DIR / f"{name}_system1.parquet"
    df.to_parquet(out, index=False)

    acc = (df["system1_pred"] == df["canonical_product"]).mean()
    print(f"{name}: accuracy={acc:.3f} -> {out}", file=sys.stderr)
    return out


if __name__ == "__main__":
    agent = load("convaiinnovations/laya")
    for split_name in ("val", "test"):
        run_split(split_name, agent)
