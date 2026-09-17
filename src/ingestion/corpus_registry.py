import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import fitz


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import (
    RAW_CASES_DIR,
    CORPUS_REGISTRY_PATH,
    SEED_URLS_PATH,
)


# ---------------------------------------------------------
# Registry schema
# ---------------------------------------------------------

FIELDNAMES = [
    "document_id",
    "file_name",
    "file_path",
    "source",
    "document_type",
    "court",
    "year",
    "title",
    "sha256_hash",
    "file_size",
    "page_count",
    "is_valid_pdf",
    "is_duplicate",
    "duplicate_of",
    "status",
    "error",
    "registered_at",
]


# ---------------------------------------------------------
# Seed metadata
# ---------------------------------------------------------

def load_seed_metadata() -> dict[str, dict]:
    """
    Load metadata collected by the eCourts downloader.

    Returns:
        {
            "ECOURTS_000001": {
                "source": "...",
                "title": "...",
                "court": "...",
                "year": "..."
            }
        }
    """

    if not SEED_URLS_PATH.exists():
        return {}

    metadata = {}

    with SEED_URLS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            # Support a few possible historical column names.
            document_id = (
                row.get("document_id")
                or row.get("case_id")
                or row.get("id")
                or ""
            ).strip()

            if not document_id:
                continue

            metadata[document_id] = {
                "source": (
                    row.get("source", "")
                    or ""
                ).strip(),

                "title": (
                    row.get("title", "")
                    or row.get("case_title", "")
                    or ""
                ).strip(),

                "court": (
                    row.get("court", "")
                    or ""
                ).strip(),

                "year": (
                    row.get("year", "")
                    or ""
                ).strip(),
            }

    return metadata


# ---------------------------------------------------------
# Hashing
# ---------------------------------------------------------

def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hasher.update(block)

    return hasher.hexdigest()


# ---------------------------------------------------------
# PDF validation
# ---------------------------------------------------------

def validate_pdf(path: Path) -> dict:
    result = {
        "file_size": 0,
        "page_count": 0,
        "is_valid_pdf": False,
        "error": "",
    }

    try:
        if not path.exists():
            result["error"] = "file_not_found"
            return result

        result["file_size"] = path.stat().st_size

        if result["file_size"] <= 0:
            result["error"] = "empty_file"
            return result

        with path.open("rb") as file:
            header = file.read(5)

        if not header.startswith(b"%PDF"):
            result["error"] = "invalid_pdf_header"
            return result

        with fitz.open(path) as document:
            result["page_count"] = document.page_count

            if document.page_count <= 0:
                result["error"] = "zero_pages"
                return result

        result["is_valid_pdf"] = True

    except Exception as error:
        result["error"] = str(error)

    return result


# ---------------------------------------------------------
# Document ID
# ---------------------------------------------------------

def infer_document_id(
    path: Path,
    index: int,
) -> str:

    stem = path.stem.strip()

    if stem:
        return stem

    return f"CASE_{index:06d}"


# ---------------------------------------------------------
# Existing registry
# ---------------------------------------------------------

def load_existing_registry() -> dict[str, dict]:
    if not CORPUS_REGISTRY_PATH.exists():
        return {}

    existing = {}

    with CORPUS_REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            file_name = row.get(
                "file_name",
                "",
            ).strip()

            if file_name:
                existing[file_name] = row

    return existing


# ---------------------------------------------------------
# Build registry
# ---------------------------------------------------------

def build_registry() -> list[dict]:

    RAW_CASES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_metadata = load_seed_metadata()

    pdf_files = sorted(
        RAW_CASES_DIR.glob("*.pdf")
    )

    existing = load_existing_registry()

    seen_hashes = {}

    rows = []

    now = datetime.now().isoformat(
        timespec="seconds"
    )

    for index, path in enumerate(
        pdf_files,
        start=1,
    ):

        validation = validate_pdf(path)

        document_id = infer_document_id(
            path,
            index,
        )

        # Metadata captured during download.
        meta = seed_metadata.get(
            document_id,
            {},
        )

        # Metadata already present in previous registry.
        old = existing.get(
            path.name,
            {},
        )

        file_hash = ""

        if validation["is_valid_pdf"]:
            file_hash = sha256_file(path)

        # -------------------------------------------------
        # Duplicate detection
        # -------------------------------------------------

        is_duplicate = False
        duplicate_of = ""

        if file_hash:

            if file_hash in seen_hashes:

                is_duplicate = True

                duplicate_of = (
                    seen_hashes[file_hash]
                )

            else:

                seen_hashes[file_hash] = (
                    document_id
                )

        # -------------------------------------------------
        # Prefer seed metadata.
        # Fall back to old registry metadata.
        # Finally use defaults.
        # -------------------------------------------------

        source = (
            meta.get("source")
            or old.get("source")
            or "ecourts"
        )

        court = (
            meta.get("court")
            or old.get("court")
            or ""
        )

        year = (
            meta.get("year")
            or old.get("year")
            or ""
        )

        title = (
            meta.get("title")
            or old.get("title")
            or path.stem
        )

        # -------------------------------------------------
        # Status
        # -------------------------------------------------

        if is_duplicate:
            status = "duplicate"

        elif validation["is_valid_pdf"]:
            status = "ready"

        else:
            status = "invalid"

        # -------------------------------------------------
        # Registry row
        # -------------------------------------------------

        rows.append(
            {
                "document_id":
                    document_id,

                "file_name":
                    path.name,

                "file_path":
                    str(path),

                "source":
                    source,

                "document_type":
                    "judgment",

                "court":
                    court,

                "year":
                    year,

                "title":
                    title,

                "sha256_hash":
                    file_hash,

                "file_size":
                    validation["file_size"],

                "page_count":
                    validation["page_count"],

                "is_valid_pdf":
                    validation["is_valid_pdf"],

                "is_duplicate":
                    is_duplicate,

                "duplicate_of":
                    duplicate_of,

                "status":
                    status,

                "error":
                    validation["error"],

                "registered_at":
                    old.get(
                        "registered_at",
                        now,
                    ),
            }
        )

    return rows


# ---------------------------------------------------------
# Write registry
# ---------------------------------------------------------

def write_registry(
    rows: list[dict],
) -> None:

    CORPUS_REGISTRY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CORPUS_REGISTRY_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()

        writer.writerows(rows)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main() -> None:

    rows = build_registry()

    write_registry(rows)

    valid = sum(
        1
        for row in rows
        if row["status"] == "ready"
    )

    duplicates = sum(
        1
        for row in rows
        if row["status"] == "duplicate"
    )

    invalid = sum(
        1
        for row in rows
        if row["status"] == "invalid"
    )

    metadata_titles = sum(
        1
        for row in rows
        if row["title"]
        and row["title"]
        != row["document_id"]
    )

    metadata_courts = sum(
        1
        for row in rows
        if row["court"]
    )

    metadata_years = sum(
        1
        for row in rows
        if row["year"]
    )

    print("\nCorpus Registry Report")
    print("=" * 70)

    print(
        f"PDF files discovered: "
        f"{len(rows)}"
    )

    print(
        f"Valid unique judgments: "
        f"{valid}"
    )

    print(
        f"Duplicates: "
        f"{duplicates}"
    )

    print(
        f"Invalid PDFs: "
        f"{invalid}"
    )

    print(
        f"Cases with title metadata: "
        f"{metadata_titles}"
    )

    print(
        f"Cases with court metadata: "
        f"{metadata_courts}"
    )

    print(
        f"Cases with year metadata: "
        f"{metadata_years}"
    )

    print(
        f"Registry: "
        f"{CORPUS_REGISTRY_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()