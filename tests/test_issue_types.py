"""Checks the issue-type grouping on CFPB issues whose wording overlaps several rules.

Run: uv run python -m tests.test_issue_types
"""
from src.experiments.common import ISSUE_TYPES, issue_type


def test_issue_type():
    cases = {
        "Loan modification,collection,foreclosure": "Payment or repayment problem",  # not collection
        "Problem with the payoff process at the end of the loan": "Payment or repayment problem",
        "Problem with fraud alerts or security freezes": "Credit report problem",  # not fraud
        "Cont'd attempts collect debt not owed": "Debt collection conduct",
        "Charged fees or interest you didn't expect": "Unexpected fees or charges",
        "Closing on a mortgage": "Account or loan access",
        "Fraud or scam": "Fraud or disputed transaction",
        "Advertising and marketing, including promotional offers": None,
        None: None,
    }
    for issue, want in cases.items():
        assert issue_type(issue) == want, (issue, issue_type(issue))
    assert set(ISSUE_TYPES) == {v for v in cases.values() if v} | {"Credit report problem"}


if __name__ == "__main__":
    test_issue_type()
    print("ok")
