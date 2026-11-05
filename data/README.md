# Data

Not committed to this repo (see `.gitignore`). Reproduce locally:

```bash
uv run python -m src.data.download     # 21 zips, ~1.43 GB -> data/raw/
uv run python -m src.data.preprocess   # -> data/processed/complaints_with_narrative.parquet
```

## Source

[CFPB Consumer Complaint Database Narratives Archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/),
public domain.

**Not the live bulk CSV** (`files.consumerfinance.gov/ccdb/complaints.csv.zip`) --
CFPB ceased publishing complaint narratives in the live Consumer Complaint Database on
2026-08-14. That endpoint has no narrative column as of this writing. The Narratives
Archive is the only surviving source: a frozen, official snapshot covering
Dec 2011 - Aug 2026, split into 21 chronological zips. This project therefore evaluates
on a fixed historical dataset, not a live-updating one -- worth stating explicitly in
any writeup.

`download.py` is idempotent -- safe to re-run, skips files already present.
`preprocess.py` streams each zip's CSV directly (never extracts the ~5GB of raw
per-period CSVs to disk) and keeps only rows with a non-empty narrative.
