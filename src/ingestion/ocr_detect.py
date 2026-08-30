import sys
from pathlib import Path

import fitz

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import RAW_CASES_DIR, RAW_STATUTES_DIR


def get_file_path(document_type: str, file_name: str) -> Path:
    if str(document_type).lower() == "statute":
        return RAW_STATUTES_DIR / file_name

    return RAW_CASES_DIR / file_name


def extract_text_sample(file_path: Path, max_pages: int = 3) -> dict:
    result = {
        "page_count": 0,
        "sample_text": "",
        "sample_char_count": 0,
        "avg_chars_per_page": 0.0,
        "error": "",
    }

    try:
        with fitz.open(file_path) as document:
            result["page_count"] = document.page_count

            texts = []

            pages_to_check = min(document.page_count, max_pages)

            for page_index in range(pages_to_check):
                page = document.load_page(page_index)
                text = page.get_text("text")
                texts.append(text)

            sample_text = "\n".join(texts)
            sample_char_count = len(sample_text.strip())

            result["sample_text"] = sample_text
            result["sample_char_count"] = sample_char_count

            if pages_to_check > 0:
                result["avg_chars_per_page"] = sample_char_count / pages_to_check

    except Exception as error:
        result["error"] = str(error)

    return result


def calculate_text_quality_score(avg_chars_per_page: float) -> float:
    score = avg_chars_per_page / 2000

    if score < 0:
        return 0.0

    if score > 1:
        return 1.0

    return round(score, 4)


def classify_pdf_quality(sample: dict) -> dict:
    page_count = sample["page_count"]
    sample_char_count = sample["sample_char_count"]
    avg_chars_per_page = sample["avg_chars_per_page"]
    error = sample["error"]

    text_quality_score = calculate_text_quality_score(avg_chars_per_page)

    if error:
        return {
            "pdf_type": "unreadable_pdf",
            "ocr_needed": "yes",
            "text_quality_score": 0.0,
            "quality_status": "ERROR",
            "reason": error,
        }

    if page_count == 0:
        return {
            "pdf_type": "empty_pdf",
            "ocr_needed": "yes",
            "text_quality_score": 0.0,
            "quality_status": "EMPTY",
            "reason": "zero_pages",
        }

    if sample_char_count == 0:
        return {
            "pdf_type": "scanned_pdf",
            "ocr_needed": "yes",
            "text_quality_score": 0.0,
            "quality_status": "NEEDS_OCR",
            "reason": "no_extractable_text",
        }

    if avg_chars_per_page < 100:
        return {
            "pdf_type": "possible_scanned_or_bad_ocr_pdf",
            "ocr_needed": "yes",
            "text_quality_score": text_quality_score,
            "quality_status": "LOW_TEXT",
            "reason": "very_low_text_density",
        }

    if avg_chars_per_page < 500:
        return {
            "pdf_type": "low_quality_text_pdf",
            "ocr_needed": "maybe",
            "text_quality_score": text_quality_score,
            "quality_status": "REVIEW",
            "reason": "low_text_density",
        }

    return {
        "pdf_type": "text_pdf",
        "ocr_needed": "no",
        "text_quality_score": text_quality_score,
        "quality_status": "GOOD",
        "reason": "extractable_text_found",
    }


def detect_ocr_need(file_path: Path) -> dict:
    sample = extract_text_sample(file_path)
    classification = classify_pdf_quality(sample)

    return {
        **sample,
        **classification,
    }
