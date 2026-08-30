import sys
import csv
from pathlib import Path

import fitz
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    MANIFEST_PATH,
    RAW_CASES_DIR,
    RAW_STATUTES_DIR,
    PDF_INTEGRITY_REPORT_PATH,
)

from src.ingestion.duplicate_detector import main as run_duplicate_detection


def get_file_path(document_type: str, file_name: str) -> Path:
    if str(document_type).lower() == "statute":
        return RAW_STATUTES_DIR / file_name

    return RAW_CASES_DIR / file_name


def looks_like_html(file_path: Path) -> bool:
    try:
        sample = file_path.read_bytes()[:500].lower()
        return b"<html" in sample or b"<!doctype html" in sample
    except Exception:
        return False


def validate_pdf(file_path: Path) -> dict:
    result = {
        "file_exists": False,
        "file_size": 0,
        "is_valid_pdf": False,
        "page_count": 0,
        "is_html_saved_as_pdf": False,
        "error": "",
    }

    try:
        if not file_path.exists():
            result["error"] = "file_not_found"
            return result

        result["file_exists"] = True
        result["file_size"] = file_path.stat().st_size

        if result["file_size"] == 0:
            result["error"] = "empty_file"
            return result

        result["is_html_saved_as_pdf"] = looks_like_html(file_path)

        if result["is_html_saved_as_pdf"]:
            result["error"] = "html_saved_as_pdf"
            return result

        with fitz.open(file_path) as document:
            result["page_count"] = document.page_count

            if document.page_count <= 0:
                result["error"] = "zero_pages"
                return result

            result["is_valid_pdf"] = True
            result["error"] = ""

    except Exception as error:
        result["error"] = str(error)

    return result


def write_report(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "document_id",
        "source",
        "document_type",
        "title",
        "court",
        "year",
        "file_name",
        "file_path",
        "file_exists",
        "file_size",
        "is_valid_pdf",
        "page_count",
        "is_html_saved_as_pdf",
        "error",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)
    report_rows = []

    for _, row in manifest.iterrows():
        document_type = str(row["document_type"])
        file_name = str(row["file_name"])
        file_path = get_file_path(document_type, file_name)

        validation = validate_pdf(file_path)

        report_rows.append(
            {
                "document_id": row["document_id"],
                "source": row["source"],
                "document_type": document_type,
                "title": row["title"],
                "court": row["court"],
                "year": row["year"],
                "file_name": file_name,
                "file_path": str(file_path),
                **validation,
            }
        )

    write_report(report_rows, PDF_INTEGRITY_REPORT_PATH)

    total = len(report_rows)
    valid = sum(1 for row in report_rows if row["is_valid_pdf"])
    invalid = total - valid

    print("\nGap 37 PDF Integrity Check Report")
    print("=" * 70)
    print(f"Total files checked: {total}")
    print(f"Valid PDFs: {valid}")
    print(f"Invalid PDFs: {invalid}")
    print(f"Report saved to: {PDF_INTEGRITY_REPORT_PATH}")
    print("=" * 70)

    for row in report_rows:
        status = "VALID" if row["is_valid_pdf"] else "INVALID"
        print(
            f"{row['document_id']} | {status} | "
            f"pages={row['page_count']} | "
            f"size={row['file_size']} | "
            f"error={row['error']}"
        )

    print("\nStarting Gap 36 Duplicate Detection after Gap 37...")
    print("=" * 70)

    run_duplicate_detection()


if __name__ == "__main__":
    main()
