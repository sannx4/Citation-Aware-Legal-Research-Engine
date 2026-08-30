import csv
import re
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    CASE_CITATIONS_PATH,
    STATUTE_MENTIONS_PATH,
    NORMALIZED_CITATIONS_PATH,
)


def normalize_text(value: str) -> str:
    value = str(value).lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = value.replace(".", "")
    value = value.replace(",", "")
    return value


def create_slug(value: str) -> str:
    value = normalize_text(value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value


def normalize_connector(connector: str) -> str:
    return "v"


def normalize_case_citations(case_citations: pd.DataFrame) -> list[dict]:
    rows = []

    for _, row in case_citations.iterrows():
        party_1 = normalize_text(row["party_1"])
        party_2 = normalize_text(row["party_2"])

        canonical_name = f"{party_1} v {party_2}"
        canonical_id = f"CASE_REF_{create_slug(canonical_name)}"

        rows.append(
            {
                "source_case_id": row["case_id"],
                "chunk_id": row["chunk_id"],
                "mention_type": "CASE",
                "original_mention": row["citation_text"],
                "canonical_name": canonical_name,
                "canonical_id": canonical_id,
                "party_1": party_1,
                "party_2": party_2,
                "reference_number": "",
                "confidence": row.get("confidence", 0.70),
                "resolution_status": "RESOLVED",
            }
        )

    return rows


def normalize_statute_mentions(statute_mentions: pd.DataFrame) -> list[dict]:
    rows = []

    for _, row in statute_mentions.iterrows():
        mention_type = row["mention_type"]
        reference_number = str(row["reference_number"])

        canonical_name = f"{mention_type.lower()} {reference_number}"
        canonical_id = f"{mention_type}_{reference_number}".upper()

        rows.append(
            {
                "source_case_id": row["case_id"],
                "chunk_id": row["chunk_id"],
                "mention_type": mention_type,
                "original_mention": row["mention_text"],
                "canonical_name": canonical_name,
                "canonical_id": canonical_id,
                "party_1": "",
                "party_2": "",
                "reference_number": reference_number,
                "confidence": 0.90,
                "resolution_status": "RESOLVED",
            }
        )

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_case_id",
        "chunk_id",
        "mention_type",
        "original_mention",
        "canonical_name",
        "canonical_id",
        "party_1",
        "party_2",
        "reference_number",
        "confidence",
        "resolution_status",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main() -> None:
    all_rows = []

    if CASE_CITATIONS_PATH.exists():
        case_citations = pd.read_csv(CASE_CITATIONS_PATH)
        all_rows.extend(normalize_case_citations(case_citations))

    if STATUTE_MENTIONS_PATH.exists():
        statute_mentions = pd.read_csv(STATUTE_MENTIONS_PATH)
        all_rows.extend(normalize_statute_mentions(statute_mentions))

    write_csv(all_rows, NORMALIZED_CITATIONS_PATH)

    print("\nDay 10 Citation Normalization Report")
    print("=" * 70)
    print(f"Normalized citations created: {len(all_rows)}")
    print(f"Output saved to: {NORMALIZED_CITATIONS_PATH}")
    print("=" * 70)

    for row in all_rows[:10]:
        print(
            f"{row['mention_type']} | "
            f"{row['original_mention']} -> "
            f"{row['canonical_id']}"
        )


if __name__ == "__main__":
    main()