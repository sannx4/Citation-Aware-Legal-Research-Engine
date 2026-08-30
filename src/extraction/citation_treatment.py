import csv
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    CITATION_CONTEXTS_PATH,
    ADVANCED_CASE_CITATIONS_PATH,
    CASE_CITATIONS_PATH,
    CITATION_TREATMENTS_PATH,
)

from src.extraction.treatment_classifier import classify_treatment


def load_citation_contexts() -> pd.DataFrame:
    if CITATION_CONTEXTS_PATH.exists():
        return pd.read_csv(CITATION_CONTEXTS_PATH)

    raise FileNotFoundError(
        f"Citation contexts not found: {CITATION_CONTEXTS_PATH}. "
        "Run advanced citation extraction first."
    )


def load_citation_metadata() -> pd.DataFrame:
    if ADVANCED_CASE_CITATIONS_PATH.exists():
        return pd.read_csv(ADVANCED_CASE_CITATIONS_PATH)

    if CASE_CITATIONS_PATH.exists():
        return pd.read_csv(CASE_CITATIONS_PATH)

    raise FileNotFoundError(
        "No citation metadata found. Run citation extraction first."
    )


def safe_get(row: dict, key: str) -> str:
    value = row.get(key, "")

    if pd.isna(value):
        return ""

    return str(value)


def build_treatment_rows(
    contexts: pd.DataFrame,
    citations: pd.DataFrame,
) -> list[dict]:
    rows = []

    citation_lookup = {}

    if "citation_id" in citations.columns:
        for _, citation_row in citations.iterrows():
            citation_lookup[str(citation_row.get("citation_id", ""))] = citation_row.to_dict()

    for _, context_row in contexts.iterrows():
        context = safe_get(context_row, "context")
        citation_text = safe_get(context_row, "citation_text")
        citation_id = safe_get(context_row, "citation_id")

        treatment = classify_treatment(context)

        metadata = citation_lookup.get(citation_id, {})

        rows.append(
            {
                "citation_id": citation_id,
                "case_id": safe_get(context_row, "case_id"),
                "chunk_id": safe_get(context_row, "chunk_id"),
                "cited_case_text": citation_text,
                "citation_type": safe_get(metadata, "citation_type"),
                "role": safe_get(metadata, "role"),
                "treatment": treatment["treatment"],
                "polarity": treatment["polarity"],
                "confidence": treatment["confidence"],
                "matched_pattern": treatment["matched_pattern"],
                "context": context,
            }
        )

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "citation_id",
        "case_id",
        "chunk_id",
        "cited_case_text",
        "citation_type",
        "role",
        "treatment",
        "polarity",
        "confidence",
        "matched_pattern",
        "context",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    contexts = load_citation_contexts()
    citations = load_citation_metadata()

    rows = build_treatment_rows(contexts, citations)
    write_csv(rows, CITATION_TREATMENTS_PATH)

    positive = sum(1 for row in rows if row["polarity"] == "POSITIVE")
    negative = sum(1 for row in rows if row["polarity"] == "NEGATIVE")
    neutral = sum(1 for row in rows if row["polarity"] == "NEUTRAL")

    print("\nGap 5/18/20/29 Citation Treatment Detection Report")
    print("=" * 70)
    print(f"Citation contexts checked: {len(rows)}")
    print(f"Positive treatments: {positive}")
    print(f"Negative treatments: {negative}")
    print(f"Neutral treatments: {neutral}")
    print(f"Output saved to: {CITATION_TREATMENTS_PATH}")
    print("=" * 70)

    for row in rows[:10]:
        print(
            f"{row['case_id']} --{row['treatment']}--> "
            f"{row['cited_case_text']} | "
            f"polarity={row['polarity']} | "
            f"confidence={row['confidence']}"
        )


if __name__ == "__main__":
    main()
