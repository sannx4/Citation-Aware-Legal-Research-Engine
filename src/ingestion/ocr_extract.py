import json
import sys
from pathlib import Path

import fitz
import pandas as pd
import pytesseract
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import OCR_QUALITY_REPORT_PATH, OCR_TEXT_DIR, OCR_EXTRACTED_TEXT_PATH


def render_page_to_image(page: fitz.Page, zoom: float = 2.0) -> Image.Image:
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix)

    image = Image.frombytes(
        "RGB",
        [pixmap.width, pixmap.height],
        pixmap.samples,
    )

    return image


def ocr_pdf(file_path: Path) -> tuple[str, list[dict]]:
    pages = []

    with fitz.open(file_path) as document:
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            image = render_page_to_image(page)

            text = pytesseract.image_to_string(image, lang="eng")

            pages.append(
                {
                    "page_number": page_index + 1,
                    "text": text,
                    "char_count": len(text.strip()),
                }
            )

    full_text = "\n".join(page["text"] for page in pages)

    return full_text, pages


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    if not OCR_QUALITY_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"OCR quality report not found: {OCR_QUALITY_REPORT_PATH}. "
            "Run pdf_quality_report.py first."
        )

    quality = pd.read_csv(OCR_QUALITY_REPORT_PATH)

    ocr_rows = quality[
        quality["ocr_needed"]
        .astype(str)
        .str.lower()
        .isin(["yes"])
    ]

    OCR_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    output_records = []

    print("\nGap 24 OCR Extraction Report")
    print("=" * 70)

    for _, row in ocr_rows.iterrows():
        document_id = str(row["document_id"])
        file_name = str(row["file_name"])
        file_path = Path(str(row["file_path"]))

        try:
            ocr_text, pages = ocr_pdf(file_path)

            text_output_path = OCR_TEXT_DIR / f"{document_id}.txt"
            text_output_path.write_text(ocr_text, encoding="utf-8")

            output_records.append(
                {
                    "document_id": document_id,
                    "file_name": file_name,
                    "file_path": str(file_path),
                    "ocr_text_path": str(text_output_path),
                    "ocr_text": ocr_text,
                    "page_count": len(pages),
                    "char_count": len(ocr_text.strip()),
                    "pages": pages,
                    "status": "success",
                    "error": "",
                }
            )

            print(
                f"SUCCESS | {document_id} | "
                f"pages={len(pages)} | chars={len(ocr_text.strip())}"
            )

        except Exception as error:
            output_records.append(
                {
                    "document_id": document_id,
                    "file_name": file_name,
                    "file_path": str(file_path),
                    "ocr_text_path": "",
                    "ocr_text": "",
                    "page_count": 0,
                    "char_count": 0,
                    "pages": [],
                    "status": "failed",
                    "error": str(error),
                }
            )

            print(f"FAILED | {document_id} | {error}")

    write_jsonl(output_records, OCR_EXTRACTED_TEXT_PATH)

    print("=" * 70)
    print(f"OCR candidates: {len(ocr_rows)}")
    print(f"OCR JSONL saved to: {OCR_EXTRACTED_TEXT_PATH}")
    print(f"OCR text folder: {OCR_TEXT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
