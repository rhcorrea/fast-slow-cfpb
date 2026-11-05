"""Canonical CFPB Product taxonomy.

CFPB renamed the Product field three times across the archive's date range (2015 era,
2017 era, 2023 era -- confirmed empirically by cross-tabbing raw product x year on the
full 3.85M-row narrative dataset). Left uncanonicalized, the "same" real-world category
appears as 2-3 different class labels depending only on when the complaint was filed,
which would make Product classification partly a date-detection task. This collapses
21 raw labels down to 11 real categories.

Two labels are collapsed by judgment call, not an unambiguous rename, and are worth
revisiting: "Consumer Loan" (2015-2017, 9458 rows) folded into the payday/personal-loan
bucket rather than vehicle loan or its own category, and "Credit card"/"Prepaid card"
folded together with "Credit card or prepaid card" even though the two products are
arguably distinct (the eras overlap: "Prepaid card" appears in every year 2015-2026,
even during 2017-2023 when "Credit card or prepaid card" also exists).
"""

CANONICAL_PRODUCT = {
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit reporting": "Credit reporting",
    "Debt collection": "Debt collection",
    "Checking or savings account": "Checking or savings account",
    "Bank account or service": "Checking or savings account",
    "Mortgage": "Mortgage",
    "Credit card": "Credit card or prepaid card",
    "Credit card or prepaid card": "Credit card or prepaid card",
    "Prepaid card": "Credit card or prepaid card",
    "Money transfer, virtual currency, or money service": "Money transfer, virtual currency, or money service",
    "Money transfers": "Money transfer, virtual currency, or money service",
    "Virtual currency": "Money transfer, virtual currency, or money service",
    "Student loan": "Student loan",
    "Vehicle loan or lease": "Vehicle loan or lease",
    "Payday loan, title loan, personal loan, or advance loan": "Payday loan, title loan, personal loan, or advance loan",
    "Payday loan, title loan, or personal loan": "Payday loan, title loan, personal loan, or advance loan",
    "Payday loan": "Payday loan, title loan, personal loan, or advance loan",
    "Consumer Loan": "Payday loan, title loan, personal loan, or advance loan",
    "Debt or credit management": "Debt or credit management",
    "Other financial service": "Other financial service",
}

# Credit repair and debt settlement got their own product ("Debt or credit management")
# only in 2023. Before that, credit repair was filed under credit reporting (2017-2023) or
# "Other financial service" (2015-2017), and debt settlement under money services
# (2017-2023) or "Other financial service" (2015-2017). Mapping on sub-product, whatever
# the product, keeps one label per real-world category across all three eras.
CANONICAL_SUB_PRODUCT = {
    "Credit repair services": "Debt or credit management",
    "Credit repair": "Debt or credit management",
    "Debt settlement": "Debt or credit management",
}


def canonicalize(df):
    """canonical_product for a frame with raw `product` and `sub_product` columns."""
    return df["sub_product"].map(CANONICAL_SUB_PRODUCT).fillna(df["product"].map(CANONICAL_PRODUCT))

# "Other financial service" has 292 rows total (0.01%) across 11 years -- too sparse to
# reliably stratify into train/val/test. Excluded from the modeling taxonomy for V1.
LABELS = sorted({v for v in CANONICAL_PRODUCT.values() if v != "Other financial service"})
