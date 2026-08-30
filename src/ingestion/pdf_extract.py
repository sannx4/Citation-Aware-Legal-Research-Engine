import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
import fitz  # PyMuPDF

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    RAW_CASES_DIR,
    RAW_EXTRACTED_CASES_PATH,
    REPORTS_DIR,
)


def read_txt_file(file_path: Path) -> tuple[str, list[dict]]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")

    pages = [
        {
            "page_number": 1,
            "text": text,
        }
    ]

    return text, pages


def extract_pdf_text(file_path: Path) -> tuple[str, list[dict]]:
    pages = []

    with fitz.open(file_path) as document:
        for page_index, page in enumerate(document, start=1):
            page_text = page.get_text("text")

            pages.append(
                {
                    "page_number": page_index,
                    "text": page_text,
                }
            )

    full_text = "\n".join(page["text"] for page in pages)

    return full_text, pages


def extract_text_by_file_type(file_path: Path) -> tuple[str, list[dict]]:
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return read_txt_file(file_path)

    if suffix == ".pdf":
        return extract_pdf_text(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")


def create_case_record(row: pd.Series, raw_text: str, pages: list[dict], file_path: Path) -> dict:
    return {
        "case_id": row["case_id"],
        "title": row["title"],
        "court": row["court"],
        "year": int(row["year"]),
        "source_type": row["source_type"],
        "file_name": row["file_name"],
        "file_path": str(file_path),
        "raw_text": raw_text,
        "pages": pages,
        "text_length": len(raw_text),
        "page_count": len(pages),
    }


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_error_report(errors: list[dict], error_path: Path) -> None:
    error_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["case_id", "file_name", "error"]

    with error_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for error in errors:
            writer.writerow(error)


def extract_cases(metadata_path: Path) -> None:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    extracted_records = []
    extraction_errors = []

    print("\nDay 3 PDF/Text Extraction Report")
    print("=" * 70)

    for _, row in metadata.iterrows():
        case_id = row["case_id"]
        file_name = row["file_name"]
        file_path = RAW_CASES_DIR / file_name

        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            raw_text, pages = extract_text_by_file_type(file_path)
            record = create_case_record(row, raw_text, pages, file_path)

            extracted_records.append(record)

            print(
                f"{case_id} | SUCCESS | "
                f"pages={len(pages)} | chars={len(raw_text)} | {file_name}"
            )

        except Exception as error:
            extraction_errors.append(
                {
                    "case_id": case_id,
                    "file_name": file_name,
                    "error": str(error),
                }
            )

            print(f"{case_id} | FAILED | {file_name} | {error}")

    write_jsonl(extracted_records, RAW_EXTRACTED_CASES_PATH)

    error_path = REPORTS_DIR / "extraction_errors.csv"
    write_error_report(extraction_errors, error_path)

    print("=" * 70)
    print(f"Extracted cases: {len(extracted_records)}")
    print(f"Failed cases: {len(extraction_errors)}")
    print(f"JSONL saved to: {RAW_EXTRACTED_CASES_PATH}")
    print(f"Error report saved to: {error_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract raw text from sample legal cases.")
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to case metadata CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    extract_cases(metadata_path)


if __name__ == "__main__":
    main()