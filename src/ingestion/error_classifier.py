import csv
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    FAILED_DOWNLOADS_PATH,
    PDF_INTEGRITY_REPORT_PATH,
    FAILED_DOWNLOADS_CLASSIFIED_PATH,
)


def classify_error(error_message: str) -> tuple[str, str]:
    error = str(error_message).lower()

    if "404" in error or "not found" in error:
        return "404_not_found", "no"

    if "403" in error or "forbidden" in error or "blocked" in error:
        return "403_blocked", "maybe"

    if "timeout" in error or "timed out" in error:
        return "timeout", "yes"

    if "connection" in error or "connectionerror" in error:
        return "connection_error", "yes"

    if "invalid url" in error or "missing schema" in error:
        return "invalid_url", "no"

    if "empty" in error or "empty_file" in error:
        return "empty_response", "yes"

    if "html_saved_as_pdf" in error:
        return "html_saved_as_pdf", "yes"

    if "not real pdf" in error or "invalid pdf" in error:
        return "invalid_pdf", "yes"

    if "parse" in error or "syntax" in error:
        return "parse_error", "maybe"

    if "file_not_found" in error:
        return "file_not_found", "maybe"

    return "unknown_error", "maybe"


def load_failed_download_errors() -> list[dict]:
    rows = []

    if not FAILED_DOWNLOADS_PATH.exists():
        return rows

    failed = pd.read_csv(FAILED_DOWNLOADS_PATH)

    for _, row in failed.iterrows():
        error_type, retryable = classify_error(row.get("error", ""))

        rows.append(
            {
                "document_id": row.get("document_id", ""),
                "url": row.get("url", ""),
                "file_name": row.get("file_name", ""),
                "source_stage": "download",
                "error_type": error_type,
                "retryable": retryable,
                "error_message": row.get("error", ""),
            }
        )

    return rows


def load_pdf_integrity_errors() -> list[dict]:
    rows = []

    if not PDF_INTEGRITY_REPORT_PATH.exists():
        return rows

    report = pd.read_csv(PDF_INTEGRITY_REPORT_PATH)

    invalid = report[
        ~report["is_valid_pdf"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    ]

    for _, row in invalid.iterrows():
        error_type, retryable = classify_error(row.get("error", ""))

        rows.append(
            {
                "document_id": row.get("document_id", ""),
                "url": "",
                "file_name": row.get("file_name", ""),
                "source_stage": "pdf_integrity",
                "error_type": error_type,
                "retryable": retryable,
                "error_message": row.get("error", ""),
            }
        )

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "document_id",
        "url",
        "file_name",
        "source_stage",
        "error_type",
        "retryable",
        "error_message",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = []

    rows.extend(load_failed_download_errors())
    rows.extend(load_pdf_integrity_errors())

    write_csv(rows, FAILED_DOWNLOADS_CLASSIFIED_PATH)

    retryable_count = sum(1 for row in rows if row["retryable"] == "yes")
    non_retryable_count = sum(1 for row in rows if row["retryable"] == "no")
    maybe_count = sum(1 for row in rows if row["retryable"] == "maybe")

    print("\nGap 38 Error Classification Report")
    print("=" * 70)
    print(f"Errors classified: {len(rows)}")
    print(f"Retryable: {retryable_count}")
    print(f"Non-retryable: {non_retryable_count}")
    print(f"Maybe retryable: {maybe_count}")
    print(f"Classified report saved to: {FAILED_DOWNLOADS_CLASSIFIED_PATH}")
    print("=" * 70)

    for row in rows[:10]:
        print(
            f"{row['document_id']} | "
            f"{row['source_stage']} | "
            f"{row['error_type']} | "
            f"retryable={row['retryable']}"
        )


if __name__ == "__main__":
    main()
