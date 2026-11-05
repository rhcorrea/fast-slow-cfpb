"""Filter the CFPB narrative archive down to what this experiment needs.

Each of the 21 export CSVs contains every complaint for its period, most with an
empty narrative -- only the minority with consent-and-narrative are useful here.
Reads straight out of each zip (chunked, so peak memory stays bounded) rather
than extracting ~5GB of raw CSVs to disk first.
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd

from .download import EXPORTS, RAW_DIR
from .taxonomy import canonicalize

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT_PATH = PROCESSED_DIR / "complaints_with_narrative.parquet"
CHUNKSIZE = 200_000

# Verified against a live export file's header before trusting these.
COLUMNS = {
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Consumer complaint narrative": "narrative",
    "Company": "company",
    "State": "state",
    "Complaint ID": "complaint_id",
}


def _filtered_rows(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            header = pd.read_csv(f, nrows=0).columns.tolist()
        missing = [c for c in COLUMNS if c not in header]
        if missing:
            raise ValueError(
                f"expected columns not found in {zip_path.name}: {missing}\n"
                f"actual header: {header}\n"
                "CFPB may have renamed columns -- update COLUMNS in preprocess.py."
            )
        with zf.open(csv_name) as f:
            for chunk in pd.read_csv(
                f, usecols=list(COLUMNS), dtype=str, chunksize=CHUNKSIZE, low_memory=False
            ):
                chunk = chunk.rename(columns=COLUMNS)
                chunk = chunk[chunk["narrative"].notna() & (chunk["narrative"].str.strip() != "")]
                if len(chunk):
                    yield chunk


def preprocess() -> Path:
    parts = []
    for i, name in enumerate(EXPORTS, 1):
        zip_path = RAW_DIR / f"{name}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"{zip_path} not found -- run download.py first")
        print(f"[{i}/{len(EXPORTS)}] filtering {zip_path.name}")
        parts.extend(_filtered_rows(zip_path))

    df = pd.concat(parts, ignore_index=True)
    df["canonical_product"] = canonicalize(df)
    assert df["canonical_product"].notna().all(), "unmapped product label -- update taxonomy.py"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    return OUT_PATH


if __name__ == "__main__":
    path = preprocess()
    print(f"done: {path}", file=sys.stderr)
