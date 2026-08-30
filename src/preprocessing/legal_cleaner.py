import csv
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    RAW_EXTRACTED_CASES_PATH,
    ADVANCED_CLEAN_CASES_PATH,
    ADVANCED_PREPROCESSING_REPORT_PATH,
)

from src.preprocessing.pii_mask import mask_pii


PAGE_NUMBER_PATTERN = re.compile(r"^\s*(?:page\s*)?\d+\s*(?:of\s*\d+)?\s*$", re.I)
MULTI_SPACE_PATTERN = re.compile(r"[ \t]+")
MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def remove_page_numbers(text: str) -> str:
    lines = []

    for line in text.splitlines():
        if PAGE_NUMBER_PATTERN.match(line.strip()):
            continue
        lines.append(line)

    return "\n".join(lines)


def remove_repeated_headers(lines: list[str], min_repeats: int = 3) -> list[str]:
    normalized_counts = {}

    for line in lines:
        normalized = " ".join(line.strip().lower().split())

        if len(normalized) < 5:
            continue

        normalized_counts[normalized] = normalized_counts.get(normalized, 0) + 1

    repeated = {
        line
        for line, count in normalized_counts.items()
        if count >= min_repeats
    }

    cleaned_lines = []

    for line in lines:
        normalized = " ".join(line.strip().lower().split())

        if normalized in repeated:
            continue

        cleaned_lines.append(line)

    return cleaned_lines


def fix_broken_lines(text: str) -> str:
    lines = text.splitlines()
    fixed = []

    for line in lines:
        current = line.strip()

        if not current:
            fixed.append("")
            continue

        if not fixed:
            fixed.append(current)
            continue

        previous = fixed[-1]

        if (
            previous
            and not previous.endswith((".", ":", ";", "?", "!", ")"))
            and current
            and current[0].islower()
        ):
            fixed[-1] = previous + " " + current
        else:
            fixed.append(current)

    return "\n".join(fixed)


def safe_clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = remove_page_numbers(text)

    lines = text.splitlines()
    lines = remove_repeated_headers(lines)
    text = "\n".join(lines)

    text = fix_broken_lines(text)
    text = MULTI_SPACE_PATTERN.sub(" ", text)
    text = MULTI_NEWLINE_PATTERN.sub("\n\n", text)

    return text.strip()


def build_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    output = []
    report_rows = []

    for record in records:
        raw_text = record.get("raw_text", "")

        cleaned_text = safe_clean_text(raw_text)
        masked_text, pii_counts = mask_pii(cleaned_text)

        new_record = {
            **record,
            "advanced_cleaned_text": cleaned_text,
            "pii_masked_text": masked_text,
            "pii_counts": pii_counts,
            "advanced_cleaned_length": len(cleaned_text),
            "pii_masked_length": len(masked_text),
        }

        output.append(new_record)

        report_rows.append(
            {
                "case_id": record.get("case_id", ""),
                "file_name": record.get("file_name", ""),
                "raw_length": len(raw_text),
                "advanced_cleaned_length": len(cleaned_text),
                "pii_masked_length": len(masked_text),
                "emails_masked": pii_counts["emails"],
                "phones_masked": pii_counts["phones"],
                "aadhaar_masked": pii_counts["aadhaar"],
                "pan_masked": pii_counts["pan"],
                "pincode_masked": pii_counts["pincode"],
                "minor_names_masked": pii_counts["minor_names"],
            }
        )

    return output, report_rows


def write_report(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "file_name",
        "raw_length",
        "advanced_cleaned_length",
        "pii_masked_length",
        "emails_masked",
        "phones_masked",
        "aadhaar_masked",
        "pan_masked",
        "pincode_masked",
        "minor_names_masked",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    records = read_jsonl(RAW_EXTRACTED_CASES_PATH)
    cleaned_records, report_rows = build_records(records)

    write_jsonl(cleaned_records, ADVANCED_CLEAN_CASES_PATH)
    write_report(report_rows, ADVANCED_PREPROCESSING_REPORT_PATH)

    print("\nAdvanced Legal Cleaner Report")
    print("=" * 70)
    print(f"Input records: {len(records)}")
    print(f"Output saved to: {ADVANCED_CLEAN_CASES_PATH}")
    print(f"Report saved to: {ADVANCED_PREPROCESSING_REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
