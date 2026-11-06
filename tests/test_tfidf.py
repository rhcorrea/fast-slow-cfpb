"""Checks the TF-IDF model's real-prior reweighting on a toy model.

Run: uv run python -m tests.test_tfidf
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.models.tfidf import classify, probabilities


def test_real_prior():
    texts = ["loan loan", "loan card", "card card", "card loan"]
    labels = ["Loan", "Loan", "Card", "Card"]
    vec = TfidfVectorizer().fit(texts)
    clf = LogisticRegression().fit(vec.transform(texts), labels)  # classes_ sorted: Card, Loan
    model = {"vec": vec, "clf": clf, "prior": np.array([0.01, 0.99])}
    ambiguous = ["loan card"]
    raw, real = probabilities(model, ambiguous, False)[0], probabilities(model, ambiguous, True)[0]
    assert abs(real.sum() - 1) < 1e-9
    # Reweighting is p * prior, renormalised: odds move by exactly the prior ratio.
    assert abs((real[1] / real[0]) / (raw[1] / raw[0]) - 99) < 1e-6
    assert classify(model, "loan card")["choice"] == "Loan"
    assert classify(model, "card card", real_prior=False)["choice"] == "Card"


if __name__ == "__main__":
    test_real_prior()
    print("ok")
