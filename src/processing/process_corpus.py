from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_CASES_DIR = RAW_DIR / "sample_cases"

CANONICAL_MANIFEST_PATH = RAW_DIR / "manifest_deduped.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TEXT_OUTPUT_DIR = PROCESSED_DIR / "text"
DOCUMENT_OUTPUT_DIR = PROCESSED_DIR / "documents"

PROCESSING_MANIFEST_PATH = (
    PROCESSED_DIR / "corpus_processing_manifest.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"
QUALITY_REPORT_PATH = (
    REPORTS_DIR / "corpus_text_quality_report.csv"
)


# ============================================================
# Extraction / quality thresholds
# ============================================================

# A page with fewer characters than this is considered nearly empty.
MIN_PAGE_CHARACTERS = 80

# Very low average text density often indicates scanned/image PDFs.
MIN_AVG_CHARS_PER_PAGE = 250

# More than this fraction of nearly-empty pages is suspicious.
MAX_EMPTY_PAGE_RATIO = 0.40

# If too little of the extracted text is alphabetic, extraction
# may be corrupted.
MIN_ALPHA_RATIO = 0.45

# Documents under this total character count are suspicious.
MIN_DOCUMENT_CHARACTERS = 500


# ============================================================
# Processing manifest fields
# ============================================================

PROCESSING_FIELDS = [
    "document_id",
    "file_name",
    "file_path",
    "sha256",
    "pages",
    "characters",
    "words",
    "empty_pages",
    "empty_page_ratio",
    "avg_chars_per_page",
    "alpha_ratio",
    "quality",
    "needs_ocr",
    "status",
    "error",
    "text_output_path",
    "document_output_path",
    "processed_at",
]


# ============================================================
# Data structures
# ============================================================

@dataclass
class PageResult:
    page_number: int
    characters: int
    words: int
    empty: bool
    text: str


@dataclass
class QualityResult:
    quality: str
    needs_ocr: bool
    characters: int
    words: int
    pages: int
    empty_pages: int
    empty_page_ratio: float
    avg_chars_per_page: float
    alpha_ratio: float
    reasons: list[str]


# ============================================================
# Generic helpers
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc,
    ).isoformat()


def ensure_directories() -> None:
    TEXT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOCUMENT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def sha256_file(
    path: Path,
) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def safe_int(
    value: Any,
    default: int | None = None,
) -> int | None:
    try:
        if value is None:
            return default

        value = str(value).strip()

        if not value:
            return default

        return int(float(value))

    except (TypeError, ValueError):
        return default


def first_nonempty(
    row: dict[str, Any],
    keys: list[str],
    default: str = "",
) -> str:
    for key in keys:
        value = row.get(key)

        if value is None:
            continue

        value = str(value).strip()

        if value:
            return value

    return default


# ============================================================
# Text normalization
# ============================================================

def normalize_page_text(
    text: str,
) -> str:
    """
    Conservative normalization.

    Important:
    We do NOT aggressively rewrite legal text because downstream
    citation extraction and evidence retrieval should preserve the
    judgment wording as much as possible.
    """

    if not text:
        return ""

    text = text.replace(
        "\x00",
        "",
    )

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    # Normalize horizontal whitespace while preserving line structure.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Remove excessive blank lines.
    text = re.sub(
        r"\n{4,}",
        "\n\n\n",
        text,
    )

    # Remove trailing whitespace.
    lines = [
        line.rstrip()
        for line in text.splitlines()
    ]

    return "\n".join(
        lines
    ).strip()


def combine_pages(
    pages: list[PageResult],
) -> str:
    """
    Build the canonical document text.

    Page markers are retained because later citation/evidence
    verification may need page provenance.
    """

    blocks: list[str] = []

    for page in pages:
        blocks.append(
            f"--- PAGE {page.page_number} ---"
        )

        if page.text:
            blocks.append(
                page.text
            )

        blocks.append("")

    return "\n".join(
        blocks
    ).strip()


# ============================================================
# PDF resolution
# ============================================================

def resolve_pdf_path(
    row: dict[str, Any],
) -> Path:
    """
    Resolve a canonical manifest row to an actual PDF.

    Supports several possible manifest column names because ingestion
    manifests may evolve over time.
    """

    possible_path = first_nonempty(
        row,
        [
            "file_path",
            "path",
            "pdf_path",
            "local_path",
        ],
    )

    if possible_path:
        candidate = Path(
            possible_path
        )

        if not candidate.is_absolute():
            candidate = (
                PROJECT_ROOT
                / candidate
            )

        if candidate.exists():
            return candidate

    file_name = first_nonempty(
        row,
        [
            "file_name",
            "filename",
            "pdf_file",
        ],
    )

    if file_name:
        candidate = (
            RAW_CASES_DIR
            / file_name
        )

        if candidate.exists():
            return candidate

    document_id = first_nonempty(
        row,
        [
            "document_id",
            "case_id",
            "id",
        ],
    )

    if document_id:
        candidate = (
            RAW_CASES_DIR
            / f"{document_id}.pdf"
        )

        if candidate.exists():
            return candidate

        matches = list(
            RAW_CASES_DIR.glob(
                f"{document_id}*.pdf"
            )
        )

        if len(matches) == 1:
            return matches[0]

    raise FileNotFoundError(
        f"Could not resolve PDF path for "
        f"{document_id or '<unknown>'}"
    )


# ============================================================
# Manifest loading
# ============================================================

def load_canonical_manifest() -> list[dict[str, str]]:
    if not CANONICAL_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Canonical manifest not found:\n"
            f"{CANONICAL_MANIFEST_PATH}"
        )

    with CANONICAL_MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        rows = [
            dict(row)
            for row in reader
        ]

    return rows


def load_previous_processing_manifest(
) -> dict[str, dict[str, str]]:
    if not PROCESSING_MANIFEST_PATH.exists():
        return {}

    with PROCESSING_MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        result: dict[
            str,
            dict[str, str],
        ] = {}

        for row in reader:
            document_id = (
                row.get(
                    "document_id",
                    "",
                ).strip()
            )

            if document_id:
                result[
                    document_id
                ] = dict(row)

        return result


# ============================================================
# PDF extraction
# ============================================================

def extract_pdf_pages(
    pdf_path: Path,
) -> list[PageResult]:
    pages: list[PageResult] = []

    with fitz.open(
        pdf_path
    ) as document:

        for index, page in enumerate(
            document,
            start=1,
        ):
            raw_text = page.get_text(
                "text"
            )

            text = normalize_page_text(
                raw_text
            )

            characters = len(
                text
            )

            words = len(
                re.findall(
                    r"\b\S+\b",
                    text,
                )
            )

            empty = (
                characters
                < MIN_PAGE_CHARACTERS
            )

            pages.append(
                PageResult(
                    page_number=index,
                    characters=characters,
                    words=words,
                    empty=empty,
                    text=text,
                )
            )

    return pages


# ============================================================
# Text quality
# ============================================================

def calculate_alpha_ratio(
    text: str,
) -> float:
    if not text:
        return 0.0

    visible_chars = [
        char
        for char in text
        if not char.isspace()
    ]

    if not visible_chars:
        return 0.0

    alpha_chars = sum(
        1
        for char in visible_chars
        if char.isalpha()
    )

    return (
        alpha_chars
        / len(visible_chars)
    )


def evaluate_text_quality(
    pages: list[PageResult],
    document_text: str,
) -> QualityResult:

    page_count = len(
        pages
    )

    characters = sum(
        page.characters
        for page in pages
    )

    words = sum(
        page.words
        for page in pages
    )

    empty_pages = sum(
        1
        for page in pages
        if page.empty
    )

    if page_count:
        empty_page_ratio = (
            empty_pages
            / page_count
        )

        avg_chars_per_page = (
            characters
            / page_count
        )

    else:
        empty_page_ratio = 1.0
        avg_chars_per_page = 0.0

    alpha_ratio = calculate_alpha_ratio(
        document_text
    )

    reasons: list[str] = []

    needs_ocr = False

    # --------------------------------------------------------
    # OCR heuristics
    # --------------------------------------------------------

    if characters < MIN_DOCUMENT_CHARACTERS:
        reasons.append(
            "very_low_document_text"
        )

        needs_ocr = True

    if (
        avg_chars_per_page
        < MIN_AVG_CHARS_PER_PAGE
    ):
        reasons.append(
            "low_text_density"
        )

        needs_ocr = True

    if (
        empty_page_ratio
        > MAX_EMPTY_PAGE_RATIO
    ):
        reasons.append(
            "high_empty_page_ratio"
        )

        needs_ocr = True

    if (
        alpha_ratio
        < MIN_ALPHA_RATIO
    ):
        reasons.append(
            "low_alphabetic_ratio"
        )

    # --------------------------------------------------------
    # Overall quality label
    # --------------------------------------------------------

    if not pages:
        quality = "failed"

    elif needs_ocr:
        quality = "poor"

    elif (
        alpha_ratio
        < MIN_ALPHA_RATIO
    ):
        quality = "review"

    else:
        quality = "good"

    return QualityResult(
        quality=quality,
        needs_ocr=needs_ocr,
        characters=characters,
        words=words,
        pages=page_count,
        empty_pages=empty_pages,
        empty_page_ratio=round(
            empty_page_ratio,
            4,
        ),
        avg_chars_per_page=round(
            avg_chars_per_page,
            2,
        ),
        alpha_ratio=round(
            alpha_ratio,
            4,
        ),
        reasons=reasons,
    )


# ============================================================
# Structured document creation
# ============================================================

def build_document_json(
    row: dict[str, Any],
    pdf_path: Path,
    pdf_hash: str,
    pages: list[PageResult],
    document_text: str,
    quality: QualityResult,
) -> dict[str, Any]:

    document_id = first_nonempty(
        row,
        [
            "document_id",
            "case_id",
            "id",
        ],
    )

    year = safe_int(
        first_nonempty(
            row,
            [
                "year",
                "case_year",
                "decision_year",
            ],
        )
    )

    return {
        "document_id": document_id,

        "title": first_nonempty(
            row,
            [
                "title",
                "case_title",
                "name",
            ],
        ),

        "court": first_nonempty(
            row,
            [
                "court",
                "court_name",
            ],
        ),

        "year": year,

        "source": first_nonempty(
            row,
            [
                "source",
                "source_name",
            ],
            default="ecourts",
        ),

        "source_url": first_nonempty(
            row,
            [
                "url",
                "source_url",
                "pdf_url",
            ],
        ),

        "file_name": pdf_path.name,

        "file_path": str(
            pdf_path
        ),

        "sha256": pdf_hash,

        "pages": quality.pages,

        "characters": (
            quality.characters
        ),

        "words": quality.words,

        "empty_pages": (
            quality.empty_pages
        ),

        "empty_page_ratio": (
            quality.empty_page_ratio
        ),

        "avg_chars_per_page": (
            quality.avg_chars_per_page
        ),

        "alpha_ratio": (
            quality.alpha_ratio
        ),

        "quality": quality.quality,

        "needs_ocr": (
            quality.needs_ocr
        ),

        "quality_reasons": (
            quality.reasons
        ),

        "extraction_method": (
            "pymupdf"
        ),

        "processed_at": utc_now(),

        "text": document_text,

        "page_data": [
            asdict(page)
            for page in pages
        ],
    }


# ============================================================
# Output writing
# ============================================================

def write_text_output(
    document_id: str,
    text: str,
) -> Path:

    path = (
        TEXT_OUTPUT_DIR
        / f"{document_id}.txt"
    )

    path.write_text(
        text,
        encoding="utf-8",
    )

    return path


def write_document_output(
    document_id: str,
    data: dict[str, Any],
) -> Path:

    path = (
        DOCUMENT_OUTPUT_DIR
        / f"{document_id}.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return path


def write_processing_manifest(
    rows: list[dict[str, Any]],
) -> None:

    with PROCESSING_MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=PROCESSING_FIELDS,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row.get(
                        key,
                        "",
                    )
                    for key
                    in PROCESSING_FIELDS
                }
            )


def write_quality_report(
    rows: list[dict[str, Any]],
) -> None:

    fields = [
        "document_id",
        "pages",
        "characters",
        "words",
        "empty_pages",
        "empty_page_ratio",
        "avg_chars_per_page",
        "alpha_ratio",
        "quality",
        "needs_ocr",
        "status",
        "error",
    ]

    with QUALITY_REPORT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row.get(
                        key,
                        "",
                    )
                    for key in fields
                }
            )


# ============================================================
# Incremental processing
# ============================================================

def can_skip_document(
    document_id: str,
    pdf_hash: str,
    previous: dict[str, dict[str, str]],
    force: bool,
) -> bool:

    if force:
        return False

    old = previous.get(
        document_id
    )

    if not old:
        return False

    if (
        old.get("status")
        != "success"
    ):
        return False

    if (
        old.get("sha256")
        != pdf_hash
    ):
        return False

    text_path = (
        TEXT_OUTPUT_DIR
        / f"{document_id}.txt"
    )

    json_path = (
        DOCUMENT_OUTPUT_DIR
        / f"{document_id}.json"
    )

    return (
        text_path.exists()
        and json_path.exists()
    )


# ============================================================
# Per-document processing
# ============================================================

def process_document(
    row: dict[str, Any],
    previous: dict[str, dict[str, str]],
    force: bool,
) -> tuple[
    dict[str, Any],
    str,
]:

    document_id = first_nonempty(
        row,
        [
            "document_id",
            "case_id",
            "id",
        ],
    )

    if not document_id:
        raise ValueError(
            "Manifest row has no document_id."
        )

    pdf_path = resolve_pdf_path(
        row
    )

    pdf_hash = sha256_file(
        pdf_path
    )

    # --------------------------------------------------------
    # Skip unchanged canonical documents
    # --------------------------------------------------------

    if can_skip_document(
        document_id=document_id,
        pdf_hash=pdf_hash,
        previous=previous,
        force=force,
    ):

        old = dict(
            previous[
                document_id
            ]
        )

        return old, "skipped"

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    pages = extract_pdf_pages(
        pdf_path
    )

    document_text = combine_pages(
        pages
    )

    quality = evaluate_text_quality(
        pages,
        document_text,
    )

    # --------------------------------------------------------
    # Save structured outputs
    # --------------------------------------------------------

    document_json = build_document_json(
        row=row,
        pdf_path=pdf_path,
        pdf_hash=pdf_hash,
        pages=pages,
        document_text=document_text,
        quality=quality,
    )

    text_path = write_text_output(
        document_id,
        document_text,
    )

    json_path = write_document_output(
        document_id,
        document_json,
    )

    result = {
        "document_id": document_id,
        "file_name": pdf_path.name,
        "file_path": str(
            pdf_path
        ),
        "sha256": pdf_hash,
        "pages": quality.pages,
        "characters": (
            quality.characters
        ),
        "words": quality.words,
        "empty_pages": (
            quality.empty_pages
        ),
        "empty_page_ratio": (
            quality.empty_page_ratio
        ),
        "avg_chars_per_page": (
            quality.avg_chars_per_page
        ),
        "alpha_ratio": (
            quality.alpha_ratio
        ),
        "quality": (
            quality.quality
        ),
        "needs_ocr": str(
            quality.needs_ocr
        ).lower(),
        "status": "success",
        "error": "",
        "text_output_path": str(
            text_path
        ),
        "document_output_path": str(
            json_path
        ),
        "processed_at": utc_now(),
    }

    return result, "processed"


# ============================================================
# Corpus processing
# ============================================================

def process_corpus(
    force: bool = False,
    limit: int | None = None,
    target_document_id: str | None = None,
) -> None:

    ensure_directories()

    canonical_rows = (
        load_canonical_manifest()
    )

    previous = (
        load_previous_processing_manifest()
    )

    # --------------------------------------------------------
    # Optional single-document processing
    # --------------------------------------------------------

    if target_document_id:
        canonical_rows = [
            row
            for row in canonical_rows
            if first_nonempty(
                row,
                [
                    "document_id",
                    "case_id",
                    "id",
                ],
            )
            == target_document_id
        ]

        if not canonical_rows:
            raise ValueError(
                f"Document not found in "
                f"canonical manifest: "
                f"{target_document_id}"
            )

    # --------------------------------------------------------
    # Optional limit
    # --------------------------------------------------------

    if limit is not None:
        canonical_rows = (
            canonical_rows[
                :limit
            ]
        )

    print()
    print(
        "Production Canonical Corpus Processing"
    )
    print("=" * 70)

    print(
        f"Canonical documents selected: "
        f"{len(canonical_rows)}"
    )

    print(
        f"Input manifest: "
        f"{CANONICAL_MANIFEST_PATH}"
    )

    print(
        f"Processed output: "
        f"{PROCESSED_DIR}"
    )

    print("=" * 70)

    results_by_id: dict[
        str,
        dict[str, Any],
    ] = {
        document_id: dict(row)
        for document_id, row
        in previous.items()
    }

    processed_count = 0
    skipped_count = 0
    failed_count = 0
    ocr_count = 0
    poor_count = 0
    review_count = 0

    # --------------------------------------------------------
    # Process canonical corpus
    # --------------------------------------------------------

    for index, row in enumerate(
        canonical_rows,
        start=1,
    ):

        document_id = first_nonempty(
            row,
            [
                "document_id",
                "case_id",
                "id",
            ],
            default=(
                f"UNKNOWN_{index}"
            ),
        )

        print(
            f"[{index}/{len(canonical_rows)}] "
            f"{document_id}",
            end="",
        )

        try:
            result, action = (
                process_document(
                    row=row,
                    previous=previous,
                    force=force,
                )
            )

            results_by_id[
                document_id
            ] = result

            if action == "skipped":
                skipped_count += 1

                print(
                    " | SKIPPED unchanged"
                )

                continue

            processed_count += 1

            quality = result.get(
                "quality",
                "",
            )

            needs_ocr = (
                str(
                    result.get(
                        "needs_ocr",
                        "",
                    )
                ).lower()
                == "true"
            )

            if needs_ocr:
                ocr_count += 1

            if quality == "poor":
                poor_count += 1

            elif quality == "review":
                review_count += 1

            print(
                f" | {quality.upper()}"
                f" | pages="
                f"{result['pages']}"
                f" | chars="
                f"{result['characters']}"
                f" | needs_ocr="
                f"{result['needs_ocr']}"
            )

        except Exception as exc:
            failed_count += 1

            error_message = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            results_by_id[
                document_id
            ] = {
                "document_id": (
                    document_id
                ),
                "file_name": "",
                "file_path": "",
                "sha256": "",
                "pages": "",
                "characters": "",
                "words": "",
                "empty_pages": "",
                "empty_page_ratio": "",
                "avg_chars_per_page": "",
                "alpha_ratio": "",
                "quality": "failed",
                "needs_ocr": "",
                "status": "failed",
                "error": error_message,
                "text_output_path": "",
                "document_output_path": "",
                "processed_at": (
                    utc_now()
                ),
            }

            print(
                f" | FAILED | "
                f"{error_message}"
            )

    # --------------------------------------------------------
    # Write processing state
    # --------------------------------------------------------

    ordered_results = sorted(
        results_by_id.values(),
        key=lambda row: row.get(
            "document_id",
            "",
        ),
    )

    write_processing_manifest(
        ordered_results
    )

    write_quality_report(
        ordered_results
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    successful_total = sum(
        1
        for row in ordered_results
        if row.get("status")
        == "success"
    )

    total_pages = sum(
        safe_int(
            row.get("pages"),
            0,
        )
        or 0
        for row in ordered_results
        if row.get("status")
        == "success"
    )

    total_characters = sum(
        safe_int(
            row.get(
                "characters"
            ),
            0,
        )
        or 0
        for row in ordered_results
        if row.get("status")
        == "success"
    )

    print()
    print(
        "Corpus Processing Report"
    )
    print("=" * 70)

    print(
        f"Canonical documents in state: "
        f"{len(ordered_results)}"
    )

    print(
        f"Successful documents: "
        f"{successful_total}"
    )

    print(
        f"Processed this run: "
        f"{processed_count}"
    )

    print(
        f"Skipped unchanged: "
        f"{skipped_count}"
    )

    print(
        f"Failed this run: "
        f"{failed_count}"
    )

    print(
        f"Needs OCR this run: "
        f"{ocr_count}"
    )

    print(
        f"Poor quality this run: "
        f"{poor_count}"
    )

    print(
        f"Review quality this run: "
        f"{review_count}"
    )

    print(
        f"Total extracted pages: "
        f"{total_pages}"
    )

    print(
        f"Total extracted characters: "
        f"{total_characters:,}"
    )

    print(
        f"Processing manifest: "
        f"{PROCESSING_MANIFEST_PATH}"
    )

    print(
        f"Quality report: "
        f"{QUALITY_REPORT_PATH}"
    )

    print(
        f"Text directory: "
        f"{TEXT_OUTPUT_DIR}"
    )

    print(
        f"Document directory: "
        f"{DOCUMENT_OUTPUT_DIR}"
    )

    print("=" * 70)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Process canonical legal judgments "
            "into structured text documents."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Reprocess documents even when "
            "the PDF hash is unchanged."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N "
            "canonical documents."
        ),
    )

    parser.add_argument(
        "--document-id",
        default=None,
        help=(
            "Process one canonical "
            "document only."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    try:
        process_corpus(
            force=args.force,
            limit=args.limit,
            target_document_id=(
                args.document_id
            ),
        )

    except Exception as exc:
        print()
        print(
            "Corpus processing failed"
        )
        print("=" * 70)
        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )
        print("=" * 70)

        sys.exit(1)


if __name__ == "__main__":
    main()