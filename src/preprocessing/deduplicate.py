import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import CLEAN_CASES_PATH, REPORTS_DIR


DEDUP_REPORT_PATH = REPORTS_DIR / "dedup_report.csv"


def read_jsonl(input_path: Path) -> list[dict]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = []

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def normalize_for_hash(text: str) -> str:
    return " ".join(text.lower().split())


def generate_text_hash(text: str) -> str:
    normalized_text = normalize_for_hash(text)
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def find_exact_duplicates(records: list[dict]) -> list[dict]:
    seen_hashes = {}
    report_rows = []

    for record in records:
        case_id = record.get("case_id")
        title = record.get("title")
        cleaned_text = record.get("cleaned_text", "")

        text_hash = generate_text_hash(cleaned_text)

        if text_hash in seen_hashes:
            original = seen_hashes[text_hash]

            report_rows.append(
                {
                    "duplicate_case_id": case_id,
                    "duplicate_title": title,
                    "original_case_id": original["case_id"],
                    "original_title": original["title"],
                    "duplicate_type": "exact_duplicate",
                    "text_hash": text_hash,
                    "duplicate_text_length": len(cleaned_text),
                }
            )
        else:
            seen_hashes[text_hash] = {
                "case_id": case_id,
                "title": title,
                "text_hash": text_hash,
            }

    return report_rows


def write_dedup_report(report_rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "duplicate_case_id",
        "duplicate_title",
        "original_case_id",
        "original_title",
        "duplicate_type",
        "text_hash",
        "duplicate_text_length",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in report_rows:
            writer.writerow(row)


def main() -> None:
    records = read_jsonl(CLEAN_CASES_PATH)
    report_rows = find_exact_duplicates(records)
    write_dedup_report(report_rows, DEDUP_REPORT_PATH)

    print("\nDay 6 Deduplication Report")
    print("=" * 70)
    print(f"Input cases checked: {len(records)}")
    print(f"Exact duplicates found: {len(report_rows)}")
    print(f"Dedup report saved to: {DEDUP_REPORT_PATH}")
    print("=" * 70)

    if report_rows:
        for row in report_rows:
            print(
                f"{row['duplicate_case_id']} duplicates "
                f"{row['original_case_id']}"
            )
    else:
        print("No exact duplicates found.")


if __name__ == "__main__":
    main()