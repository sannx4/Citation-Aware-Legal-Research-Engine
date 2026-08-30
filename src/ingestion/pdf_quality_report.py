import csv
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    MANIFEST_PATH,
    PDF_INTEGRITY_REPORT_PATH,
    OCR_QUALITY_REPORT_PATH,
)

from src.ingestion.ocr_detect import get_file_path, detect_ocr_need


def load_valid_documents() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    if not PDF_INTEGRITY_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"PDF integrity report not found: {PDF_INTEGRITY_REPORT_PATH}. "
            "Run Gap 37 first."
        )

    manifest = pd.read_csv(MANIFEST_PATH)
    integrity = pd.read_csv(PDF_INTEGRITY_REPORT_PATH)

    valid = integrity[
        integrity["is_valid_pdf"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    ]

    merged = manifest.merge(
        valid[["document_id", "is_valid_pdf"]],
        on="document_id",
        how="inner",
    )

    return merged


def write_csv(rows: list[dict], output_path: Path) -> None:
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
        "page_count",
        "sample_char_count",
        "avg_chars_per_page",
        "pdf_type",
        "ocr_needed",
        "text_quality_score",
        "quality_status",
        "reason",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    documents = load_valid_documents()

    rows = []

    for _, row in documents.iterrows():
        document_type = str(row["document_type"])
        file_name = str(row["file_name"])
        file_path = get_file_path(document_type, file_name)

        quality = detect_ocr_need(file_path)

        rows.append(
            {
                "document_id": row["document_id"],
                "source": row["source"],
                "document_type": document_type,
                "title": row["title"],
                "court": row["court"],
                "year": row["year"],
                "file_name": file_name,
                "file_path": str(file_path),
                "page_count": quality["page_count"],
                "sample_char_count": quality["sample_char_count"],
                "avg_chars_per_page": round(quality["avg_chars_per_page"], 2),
                "pdf_type": quality["pdf_type"],
                "ocr_needed": quality["ocr_needed"],
                "text_quality_score": quality["text_quality_score"],
                "quality_status": quality["quality_status"],
                "reason": quality["reason"],
            }
        )

    write_csv(rows, OCR_QUALITY_REPORT_PATH)

    total = len(rows)
    needs_ocr = sum(1 for row in rows if row["ocr_needed"] == "yes")
    maybe_ocr = sum(1 for row in rows if row["ocr_needed"] == "maybe")
    good = sum(1 for row in rows if row["quality_status"] == "GOOD")

    print("\nGap 24 OCR and PDF Quality Report")
    print("=" * 70)
    print(f"Documents checked: {total}")
    print(f"Good text PDFs: {good}")
    print(f"OCR needed: {needs_ocr}")
    print(f"Manual review / maybe OCR: {maybe_ocr}")
    print(f"Report saved to: {OCR_QUALITY_REPORT_PATH}")
    print("=" * 70)

    for row in rows[:10]:
        print(
            f"{row['document_id']} | "
            f"{row['pdf_type']} | "
            f"ocr={row['ocr_needed']} | "
            f"score={row['text_quality_score']} | "
            f"{row['file_name']}"
        )


if __name__ == "__main__":
    main()
