import csv
import re
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    CASE_METADATA_PATH,
    CITATION_TREATMENTS_PATH,
    CITATION_RESOLUTION_PATH,
)


KNOWN_CITATION_MAP = {
    "(1978) 1 SCC 248": "CASE_002",
    "AIR 1950 SC 27": "CASE_003",
}


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).upper()


def load_case_titles() -> dict:
    metadata = pd.read_csv(CASE_METADATA_PATH)

    title_map = {}

    for _, row in metadata.iterrows():
        title = normalize(row.get("title", ""))
        title_map[title] = row.get("case_id", "")

    return title_map


def resolve_citation(citation_text: str, title_map: dict) -> tuple[str, str]:
    citation_text_clean = str(citation_text).strip()

    if citation_text_clean in KNOWN_CITATION_MAP:
        return KNOWN_CITATION_MAP[citation_text_clean], "known_reporter_citation_map"

    normalized = normalize(citation_text_clean)

    if normalized in title_map:
        return title_map[normalized], "exact_case_title_match"

    return "CASE_REF_" + re.sub(r"[^A-Z0-9]+", "_", normalized).strip("_"), "unresolved_reference"


def main() -> None:
    if not CITATION_TREATMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CITATION_TREATMENTS_PATH}. Run citation_treatment.py first."
        )

    treatments = pd.read_csv(CITATION_TREATMENTS_PATH)
    title_map = load_case_titles()

    rows = []

    for _, row in treatments.iterrows():
        cited_text = row.get("cited_case_text", "")
        resolved_case_id, reason = resolve_citation(cited_text, title_map)

        rows.append(
            {
                "case_id": row.get("case_id", ""),
                "cited_case_text": cited_text,
                "resolved_case_id": resolved_case_id,
                "treatment": row.get("treatment", ""),
                "polarity": row.get("polarity", ""),
                "confidence": row.get("confidence", ""),
                "resolution_reason": reason,
            }
        )

    CITATION_RESOLUTION_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CITATION_RESOLUTION_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case_id",
                "cited_case_text",
                "resolved_case_id",
                "treatment",
                "polarity",
                "confidence",
                "resolution_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nCitation Resolution Report")
    print("=" * 70)
    print(f"Citations checked: {len(rows)}")
    print(f"Output saved to: {CITATION_RESOLUTION_PATH}")
    print("=" * 70)

    for row in rows:
        print(
            f"{row['cited_case_text']} -> {row['resolved_case_id']} | "
            f"{row['resolution_reason']}"
        )


if __name__ == "__main__":
    main()
