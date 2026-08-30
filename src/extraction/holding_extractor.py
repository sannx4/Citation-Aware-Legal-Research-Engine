import csv
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    ADVANCED_CLEAN_CASES_PATH,
    CLEAN_CASES_PATH,
    CASE_OUTCOMES_PATH,
)

from src.extraction.outcome_classifier import classify_case_outcome


HOLDING_PATTERNS = [
    r"\bit\s+is\s+held\s+that\s+(.{20,500})",
    r"\bwe\s+hold\s+that\s+(.{20,500})",
    r"\bthe\s+court\s+held\s+that\s+(.{20,500})",
    r"\bit\s+was\s+held\s+that\s+(.{20,500})",
]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def get_input_cases() -> list[dict]:
    if ADVANCED_CLEAN_CASES_PATH.exists():
        return read_jsonl(ADVANCED_CLEAN_CASES_PATH)

    return read_jsonl(CLEAN_CASES_PATH)


def extract_holding(text: str) -> dict:
    cleaned = " ".join(text.split())

    for pattern in HOLDING_PATTERNS:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)

        if match:
            holding = match.group(0).strip()

            return {
                "holding_text": holding[:700],
                "holding_found": "yes",
                "holding_confidence": 0.90,
            }

    return {
        "holding_text": "",
        "holding_found": "no",
        "holding_confidence": 0.0,
    }


def get_case_text(record: dict) -> str:
    return (
        record.get("pii_masked_text")
        or record.get("advanced_cleaned_text")
        or record.get("cleaned_text")
        or record.get("raw_text")
        or ""
    )


def build_case_outcomes(records: list[dict]) -> list[dict]:
    rows = []

    for record in records:
        text = get_case_text(record)

        holding = extract_holding(text)
        outcome = classify_case_outcome(text)

        rows.append(
            {
                "case_id": record.get("case_id", record.get("document_id", "")),
                "title": record.get("title", ""),
                "court": record.get("court", ""),
                "year": record.get("year", ""),
                "outcome": outcome["outcome"],
                "outcome_confidence": outcome["confidence"],
                "outcome_evidence": outcome["evidence_sentence"],
                "holding_found": holding["holding_found"],
                "holding_confidence": holding["holding_confidence"],
                "holding_text": holding["holding_text"],
            }
        )

    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "title",
        "court",
        "year",
        "outcome",
        "outcome_confidence",
        "outcome_evidence",
        "holding_found",
        "holding_confidence",
        "holding_text",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    records = get_input_cases()
    rows = build_case_outcomes(records)

    write_csv(rows, CASE_OUTCOMES_PATH)

    print("\nGap 29 Case Outcome / Holding Extraction Report")
    print("=" * 70)
    print(f"Cases processed: {len(records)}")
    print(f"Output saved to: {CASE_OUTCOMES_PATH}")
    print("=" * 70)

    for row in rows[:10]:
        print(
            f"{row['case_id']} | "
            f"outcome={row['outcome']} | "
            f"holding={row['holding_found']}"
        )


if __name__ == "__main__":
    main()
