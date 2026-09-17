import csv
import sys
from pathlib import Path

import fitz

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import (
    MANIFEST_PATH,
    RAW_CASES_DIR,
    RAW_STATUTES_DIR,
    PDF_INTEGRITY_REPORT_PATH,
    SEED_URLS_PATH,
)

from src.ingestion.duplicate_detector import (
    main as run_duplicate_detection,
)


# ============================================================
# Metadata helpers
# ============================================================

def clean(value) -> str:
    if value is None:
        return ""

    value = str(value).strip()

    if value.lower() == "nan":
        return ""

    return value


def load_csv_rows(
    path: Path,
) -> list[dict]:

    if not path.exists():
        return []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        return list(
            csv.DictReader(file)
        )


def build_metadata_index() -> dict[str, dict]:
    """
    Build metadata lookup from both:

    1. seed_urls.csv
       - new growing eCourts corpus

    2. manifest.csv
       - older corpus entries / statutes

    Lookup is keyed primarily by file_name.
    """

    metadata = {}

    # --------------------------------------------------------
    # Old manifest first
    # --------------------------------------------------------

    for row in load_csv_rows(
        MANIFEST_PATH
    ):

        file_name = clean(
            row.get("file_name")
        )

        if not file_name:
            continue

        metadata[file_name] = {
            "document_id": clean(
                row.get("document_id")
            ),
            "source": clean(
                row.get("source")
            ),
            "document_type": clean(
                row.get("document_type")
            ),
            "title": clean(
                row.get("title")
            ),
            "court": clean(
                row.get("court")
            ),
            "year": clean(
                row.get("year")
            ),
        }

    # --------------------------------------------------------
    # seed_urls.csv overrides manifest metadata
    # because it represents newer downloaded cases.
    # --------------------------------------------------------

    for row in load_csv_rows(
        SEED_URLS_PATH
    ):

        file_name = clean(
            row.get("file_name")
        )

        if not file_name:
            continue

        existing = metadata.get(
            file_name,
            {},
        )

        metadata[file_name] = {
            "document_id": (
                clean(row.get("document_id"))
                or existing.get(
                    "document_id",
                    "",
                )
            ),
            "source": (
                clean(row.get("source"))
                or existing.get(
                    "source",
                    "",
                )
            ),
            "document_type": (
                clean(
                    row.get(
                        "document_type"
                    )
                )
                or existing.get(
                    "document_type",
                    "",
                )
            ),
            "title": (
                clean(row.get("title"))
                or existing.get(
                    "title",
                    "",
                )
            ),
            "court": (
                clean(row.get("court"))
                or existing.get(
                    "court",
                    "",
                )
            ),
            "year": (
                clean(row.get("year"))
                or existing.get(
                    "year",
                    "",
                )
            ),
        }

    return metadata


# ============================================================
# Corpus discovery
# ============================================================

def discover_pdf_files() -> list[dict]:
    """
    Discover PDFs from the actual corpus directories.

    This means:

    20 PDFs    -> 20 discovered
    100 PDFs   -> 100 discovered
    1,000 PDFs -> 1,000 discovered

    We no longer depend on manifest row count.
    """

    discovered = []

    # --------------------------------------------------------
    # Judgments
    # --------------------------------------------------------

    if RAW_CASES_DIR.exists():

        for file_path in sorted(
            RAW_CASES_DIR.rglob("*.pdf")
        ):

            discovered.append(
                {
                    "file_path":
                        file_path,
                    "document_type":
                        "judgment",
                }
            )

    # --------------------------------------------------------
    # Statutes
    # --------------------------------------------------------

    if RAW_STATUTES_DIR.exists():

        for file_path in sorted(
            RAW_STATUTES_DIR.rglob("*.pdf")
        ):

            discovered.append(
                {
                    "file_path":
                        file_path,
                    "document_type":
                        "statute",
                }
            )

    return discovered


# ============================================================
# PDF validation
# ============================================================

def looks_like_html(
    file_path: Path,
) -> bool:

    try:

        with file_path.open(
            "rb"
        ) as file:

            sample = (
                file.read(1000)
                .lower()
            )

        return (
            b"<html" in sample
            or
            b"<!doctype html" in sample
        )

    except Exception:
        return False


def has_pdf_signature(
    file_path: Path,
) -> bool:

    try:

        with file_path.open(
            "rb"
        ) as file:

            header = file.read(5)

        return header.startswith(
            b"%PDF"
        )

    except Exception:
        return False


def validate_pdf(
    file_path: Path,
) -> dict:

    result = {
        "file_exists": False,
        "file_size": 0,
        "is_valid_pdf": False,
        "page_count": 0,
        "is_html_saved_as_pdf": False,
        "error": "",
    }

    try:

        # ----------------------------------------------------
        # Exists
        # ----------------------------------------------------

        if not file_path.exists():

            result["error"] = (
                "file_not_found"
            )

            return result

        result["file_exists"] = True

        # ----------------------------------------------------
        # Size
        # ----------------------------------------------------

        result["file_size"] = (
            file_path.stat().st_size
        )

        if result["file_size"] <= 0:

            result["error"] = (
                "empty_file"
            )

            return result

        # ----------------------------------------------------
        # HTML accidentally saved as PDF
        # ----------------------------------------------------

        result[
            "is_html_saved_as_pdf"
        ] = looks_like_html(
            file_path
        )

        if result[
            "is_html_saved_as_pdf"
        ]:

            result["error"] = (
                "html_saved_as_pdf"
            )

            return result

        # ----------------------------------------------------
        # PDF magic bytes
        # ----------------------------------------------------

        if not has_pdf_signature(
            file_path
        ):

            result["error"] = (
                "invalid_pdf_header"
            )

            return result

        # ----------------------------------------------------
        # PyMuPDF structural check
        # ----------------------------------------------------

        with fitz.open(
            file_path
        ) as document:

            result["page_count"] = (
                document.page_count
            )

            if (
                document.page_count
                <= 0
            ):

                result["error"] = (
                    "zero_pages"
                )

                return result

        result["is_valid_pdf"] = True
        result["error"] = ""

    except Exception as error:

        result["error"] = (
            f"pdf_open_error: {error}"
        )

    return result


# ============================================================
# Report
# ============================================================

def write_report(
    rows: list[dict],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Main integrity pipeline
# ============================================================

def main() -> None:

    metadata_index = (
        build_metadata_index()
    )

    discovered_files = (
        discover_pdf_files()
    )

    report_rows = []

    for item in discovered_files:

        file_path = item[
            "file_path"
        ]

        file_name = (
            file_path.name
        )

        document_type = item[
            "document_type"
        ]

        metadata = (
            metadata_index.get(
                file_name,
                {},
            )
        )

        document_id = (
            metadata.get(
                "document_id"
            )
            or file_path.stem
        )

        source = (
            metadata.get(
                "source"
            )
            or "local"
        )

        title = (
            metadata.get(
                "title"
            )
            or file_path.stem
        )

        court = (
            metadata.get(
                "court"
            )
            or ""
        )

        year = (
            metadata.get(
                "year"
            )
            or ""
        )

        # Prefer explicit metadata type,
        # otherwise use directory-derived type.
        document_type = (
            metadata.get(
                "document_type"
            )
            or document_type
        )

        validation = validate_pdf(
            file_path
        )

        report_rows.append(
            {
                "document_id":
                    document_id,

                "source":
                    source,

                "document_type":
                    document_type,

                "title":
                    title,

                "court":
                    court,

                "year":
                    year,

                "file_name":
                    file_name,

                "file_path":
                    str(file_path),

                **validation,
            }
        )

    # --------------------------------------------------------
    # Save integrity report
    # --------------------------------------------------------

    write_report(
        report_rows,
        PDF_INTEGRITY_REPORT_PATH,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total = len(
        report_rows
    )

    valid = sum(
        1
        for row in report_rows
        if row["is_valid_pdf"]
    )

    invalid = (
        total - valid
    )

    judgments = sum(
        1
        for row in report_rows
        if str(
            row["document_type"]
        ).lower()
        == "judgment"
    )

    statutes = sum(
        1
        for row in report_rows
        if str(
            row["document_type"]
        ).lower()
        == "statute"
    )

    total_pages = sum(
        int(
            row["page_count"]
            or 0
        )
        for row in report_rows
        if row["is_valid_pdf"]
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print(
        "\nGap 37 PDF Integrity Check Report"
    )

    print("=" * 70)

    print(
        f"Total files checked: "
        f"{total}"
    )

    print(
        f"Judgment PDFs: "
        f"{judgments}"
    )

    print(
        f"Statute PDFs: "
        f"{statutes}"
    )

    print(
        f"Valid PDFs: "
        f"{valid}"
    )

    print(
        f"Invalid PDFs: "
        f"{invalid}"
    )

    print(
        f"Total valid pages: "
        f"{total_pages}"
    )

    print(
        f"Report saved to: "
        f"{PDF_INTEGRITY_REPORT_PATH}"
    )

    print("=" * 70)

    for row in report_rows:

        status = (
            "VALID"
            if row[
                "is_valid_pdf"
            ]
            else "INVALID"
        )

        print(
            f"{row['document_id']} | "
            f"{status} | "
            f"pages={row['page_count']} | "
            f"size={row['file_size']} | "
            f"error={row['error']}"
        )

    # --------------------------------------------------------
    # Existing chained pipeline
    # --------------------------------------------------------

    print(
        "\nStarting Gap 36 Duplicate "
        "Detection after Gap 37..."
    )

    print("=" * 70)

    run_duplicate_detection()


if __name__ == "__main__":
    main()