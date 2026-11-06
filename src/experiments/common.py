"""Shared pieces for the Laya platform experiments: the issue-type question and test-set scoring.

Issue type is a second field for the multi-question test. CFPB's 173 `issue` values mostly
differ by product era, so they're grouped into six types that cut across products. The first
matching rule wins; unmatched issues (advertising, customer service, "other features"...) get
None and aren't scored, which leaves 88% of the test set.
"""
import re

import numpy as np
import pandas as pd

from ..data.taxonomy import LABELS
from ..models import system1

P = system1.PROCESSED_DIR
RESULTS = P / "laya_curve.jsonl"

ISSUE_TYPES = {  # label: (criteria for Laya, regex over the lowercased CFPB issue)
    "Credit report problem": ("wrong information on a credit report, misuse of a report, or a credit bureau's dispute handling",
                              r"report|investigation|credit monitoring|fraud alert|security freeze"),
    "Payment or repayment problem": ("trouble making payments, paying off, or falling behind on a loan, mortgage or bill",
                                     r"struggling|repay|unable to pay|payment|payoff|loan modification|foreclosure|repossession|pay your bill"),
    "Debt collection conduct": ("how a collector is collecting a debt: debt not owed, notices, threats, or contact tactics",
                                r"collect|debt not owed|about debt|verification of debt|threaten|legal action|false statements|communication tactics|improper contact|electronic communications"),
    "Fraud or disputed transaction": ("fraud, a scam, or an unauthorized, disputed or failed transaction",
                                      r"fraud|scam|unauthori|purchase shown|transaction problem|billing dispute|purchase or transfer|money was not available"),
    "Unexpected fees or charges": ("unexpected fees, interest, or charges on an account",
                                   r"fee|interest|charging your account|funds being low|settlement process"),
    "Account or loan access": ("opening, closing, getting, or managing an account, card or loan",
                               r"opening|closing|getting a|getting the|applying|managing|dealing with|end of the loan|trouble using|accessing funds|deposits and withdrawals|mobile wallet|application"),
}
_RULES = [(k, re.compile(rx)) for k, (_, rx) in ISSUE_TYPES.items()]
ISSUE_Q = {"type": "choice", "instructions": "What kind of problem is the consumer complaining about?",
           "criteria": {k: v[0] for k, v in ISSUE_TYPES.items()}}
MULTI = {**system1.QUESTIONS, "issue_type": ISSUE_Q}


def issue_type(issue):
    s = (issue or "").lower()
    return next((k for k, rx in _RULES if rx.search(s)), None)


def train_meta():
    tr = pd.read_parquet(P / "train.parquet", columns=["complaint_id", "canonical_product", "issue"])
    tr = tr[tr["canonical_product"].isin(LABELS)]
    return tr, tr["canonical_product"].value_counts(normalize=True)


def sample(tr, n, seed):
    """The rows tfidf.train picks: n per class, in groupby order, random_state=seed."""
    return pd.concat([g.sample(min(len(g), n), random_state=seed) for _, g in tr.groupby("canonical_product")])


def with_text(rows):
    # Read narratives for the sampled ids only, so the 3.85M-row column never sits in memory.
    t = pd.read_parquet(P / "train.parquet", columns=["complaint_id", "narrative"],
                        filters=[("complaint_id", "in", list(rows["complaint_id"]))])
    return rows.merge(t, on="complaint_id", validate="1:1")


def test_frame():
    df = pd.read_parquet(P / "test.parquet", columns=["complaint_id", "narrative", "canonical_product", "issue"])
    df["issue_type"] = df["issue"].map(issue_type)
    return df


def product_scores(y, probs, shares):
    """Balanced accuracy, real-mix accuracy raw, and real-mix after reweighting the
    probabilities to the real class shares. probs: one {label: p} per row."""
    labels = list(LABELS)
    p = np.array([[d[k] for k in labels] for d in probs])
    raw = np.array(labels)[p.argmax(1)]
    prior = np.array(labels)[(p * shares.reindex(labels).to_numpy()).argmax(1)]
    w = (y.map(shares) / y.map(y.value_counts())).to_numpy()
    acc = lambda pred, w=None: float(np.average(np.asarray(y) == pred, weights=w))
    return {"balanced": acc(raw), "real_raw": acc(raw, w), "real_prior": acc(prior, w)}


def issue_score(y, pred):
    m = y.notna().to_numpy()
    return float((np.asarray(y)[m] == np.asarray(pred)[m]).mean())
