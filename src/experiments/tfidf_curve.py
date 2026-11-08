"""TF-IDF on the same small samples Laya fine-tunes on: product and issue type, 5 seeds per size.

    python -m src.experiments.tfidf_curve        # appends to data/processed/laya_curve.jsonl
"""
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .common import RESULTS, issue_score, issue_type, product_scores, sample, test_frame, train_meta, with_text

if __name__ == "__main__":
    tr, shares = train_meta()
    te = test_frame()
    with RESULTS.open("a") as out:
        for n in (10, 50, 100, 500):
            for seed in range(5):
                s = with_text(sample(tr, n, seed))
                # min_df=3 would leave almost no vocabulary at 10 per class; 1 keeps every word seen.
                vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1 if n < 100 else 3, max_features=300_000,
                                      sublinear_tf=True, dtype=np.float32)
                X, Xt = vec.fit_transform(s["narrative"]), vec.transform(te["narrative"])
                clf = LogisticRegression(C=3.0, max_iter=1000, class_weight="balanced").fit(X, s["canonical_product"])
                probs = [dict(zip(clf.classes_, map(float, r))) for r in clf.predict_proba(Xt)]
                rec = {"model": "tfidf", "n": n, "seed": seed, **product_scores(te["canonical_product"], probs, shares)}
                it = s["issue"].map(issue_type)
                m = it.notna().to_numpy()
                clf2 = LogisticRegression(C=3.0, max_iter=1000, class_weight="balanced").fit(X[m], it[m])
                rec["issue_type"] = issue_score(te["issue_type"], clf2.predict(Xt))
                print(json.dumps(rec), file=out, flush=True)
                print(rec, flush=True)
