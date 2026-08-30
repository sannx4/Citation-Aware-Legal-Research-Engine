import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from src.config import SEED_URLS_PATH


def crawl_seed_urls() -> list[dict]:
    if not SEED_URLS_PATH.exists():
        raise FileNotFoundError(f"Seed URL file not found: {SEED_URLS_PATH}")

    dataframe = pd.read_csv(SEED_URLS_PATH)

    rows = []

    for _, row in dataframe.iterrows():
        rows.append(
            {
                "document_id": row["document_id"],
                "source": row["source"],
                "document_type": row["document_type"],
                "title": row["title"],
                "court": row["court"],
                "year": str(row["year"]),
                "url": row["url"],
                "file_name": row["file_name"],
                "status": "pending",
            }
        )

    return rows