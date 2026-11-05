"""Fetch the CFPB Consumer Complaint Database Narratives Archive.

CFPB stopped publishing narratives in the live database on 2026-08-14 (see
data/README.md). The narrative field only survives in this official historical
archive: 21 chronological zip files, Dec 2011 - Aug 2026, one CSV each.
https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/

Zips are cached in data/raw/ (gitignored) but never extracted to disk -- each CSV
member is ~150-300MB uncompressed and there are 21 of them (~5GB+ total), so
preprocess.py reads straight out of the zip instead.
"""
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://files.consumerfinance.gov/f/documents/"
EXPORTS = [
    "CCDB_Export_1_December_2011_through_April_2018",
    "CCDB_Export_2_May_2018_through_April_2021",
    "CCDB_Export_3_May_2021_through_October_2022",
    "CCDB_Export_4_November_2022_through_August_2023",
    "CCDB_Export_5_September_2023_through_March_2024",
    "CCDB_Export_6_April_2024_through_July_2024",
    "CCDB_Export_7_August_2024_through_October_2024",
    "CCDB_Export_8_November_2024_through_December_2024",
    "CCDB_Export_9_January_2025_through_February_2025",
    "CCDB_Export_10_March_2025_through_April_2025",
    "CCDB_Export_11_May_2025_through_June_2025",
    "CCDB_Export_12_July_2025_through_August_2025",
    "CCDB_Export_13_September_2025_through_October_2025",
    "CCDB_Export_14_November_2025_through_December_2025",
    "CCDB_Export_15_January_2026_through_February_2026",
    "CCDB_Export_16_March_2026",
    "CCDB_Export_17_April_2026",
    "CCDB_Export_18_May_2026",
    "CCDB_Export_19_June_2026",
    "CCDB_Export_20_July_2026",
    "CCDB_Export_21_August_2026",
]
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _report(block_count: int, block_size: int, total_size: int) -> None:
    done = block_count * block_size
    pct = min(100, done * 100 // total_size) if total_size > 0 else 0
    print(f"\r  {pct}% ({done / 1e6:.0f} MB)", end="", flush=True)


def download() -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, name in enumerate(EXPORTS, 1):
        dest = RAW_DIR / f"{name}.zip"
        if dest.exists():
            print(f"[{i}/{len(EXPORTS)}] already have {dest.name}")
        else:
            print(f"[{i}/{len(EXPORTS)}] {dest.name}")
            urllib.request.urlretrieve(BASE_URL + name + ".zip", dest, reporthook=_report)
            print()
        paths.append(dest)
    return paths


if __name__ == "__main__":
    paths = download()
    total = sum(p.stat().st_size for p in paths)
    print(f"done: {len(paths)} files, {total / 1e9:.2f} GB", file=sys.stderr)
