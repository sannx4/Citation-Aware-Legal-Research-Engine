import csv
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import (
    MANIFEST_PATH,
    SEED_URLS_PATH,
    PDF_INTEGRITY_REPORT_PATH,
    DUPLICATE_DOWNLOADS_PATH,
    DEDUPED_MANIFEST_PATH,
)

from src.ingestion.hash_index import (
    main as run_hash_index,
)

from src.ingestion.error_classifier import (
    main as run_error_classifier,
)


# ============================================================
# Helpers
# ============================================================

def clean(value) -> str:
    if value is None:
        return ""

    value = str(value).strip()

    if value.lower() == "nan":
        return ""

    return value


def normalize_text(value: str) -> str:
    return " ".join(
        clean(value)
        .lower()
        .split()
    )


def normalize_url(url: str) -> str:
    return (
        clean(url)
        .split("#")[0]
        .rstrip("/")
    )


def sha256_file(
    file_path: Path,
) -> str:

    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            hasher.update(block)

    return hasher.hexdigest()


# ============================================================
# Metadata loading
# ============================================================

def load_optional_csv(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def build_url_index() -> dict[str, str]:
    """
    Build document_id -> URL mapping.

    New seed_urls.csv takes precedence over the
    historical manifest.
    """

    url_index = {}

    # --------------------------------------------------------
    # Historical manifest
    # --------------------------------------------------------

    manifest = load_optional_csv(
        MANIFEST_PATH
    )

    if not manifest.empty:

        for _, row in manifest.iterrows():

            document_id = clean(
                row.get(
                    "document_id",
                    "",
                )
            )

            url = normalize_url(
                row.get(
                    "url",
                    "",
                )
            )

            if document_id and url:
                url_index[
                    document_id
                ] = url

    # --------------------------------------------------------
    # Growing eCourts seed metadata
    # --------------------------------------------------------

    seeds = load_optional_csv(
        SEED_URLS_PATH
    )

    if not seeds.empty:

        for _, row in seeds.iterrows():

            document_id = clean(
                row.get(
                    "document_id",
                    "",
                )
            )

            url = normalize_url(
                row.get(
                    "url",
                    "",
                )
            )

            if document_id and url:
                url_index[
                    document_id
                ] = url

    return url_index


# ============================================================
# Load valid corpus
# ============================================================

def load_valid_corpus() -> pd.DataFrame:
    """
    PDF integrity report is now the authoritative
    source for physical files.

    20 valid PDFs -> 20 rows here.
    """

    if not PDF_INTEGRITY_REPORT_PATH.exists():
        raise FileNotFoundError(
            "Run PDF integrity check first. "
            f"Missing: {PDF_INTEGRITY_REPORT_PATH}"
        )

    report = pd.read_csv(
        PDF_INTEGRITY_REPORT_PATH
    )

    if "is_valid_pdf" not in report.columns:
        raise ValueError(
            "Integrity report is missing "
            "'is_valid_pdf' column."
        )

    valid_mask = (
        report["is_valid_pdf"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )

    valid = (
        report[
            valid_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    url_index = (
        build_url_index()
    )

    valid["url"] = (
        valid["document_id"]
        .astype(str)
        .map(
            lambda document_id:
                url_index.get(
                    document_id,
                    "",
                )
        )
    )

    return valid


# ============================================================
# Metadata duplicate key
# ============================================================

def judgment_key(
    row: pd.Series,
) -> str:
    """
    Secondary duplicate signal.

    We intentionally keep this conservative because
    generic court metadata can otherwise create
    false positives.
    """

    title = normalize_text(
        row.get(
            "title",
            "",
        )
    )

    court = normalize_text(
        row.get(
            "court",
            "",
        )
    )

    year = normalize_text(
        row.get(
            "year",
            "",
        )
    )

    # Title is mandatory for metadata matching.
    if not title:
        return ""

    return (
        f"{title}|"
        f"{court}|"
        f"{year}"
    )


# ============================================================
# Duplicate detection
# ============================================================

def detect_duplicates(
    corpus: pd.DataFrame,
) -> tuple[
    list[dict],
    pd.DataFrame,
]:
    """
    Detect one canonical duplicate relationship
    per physical document.

    Priority:

    1. same SHA-256
    2. same URL
    3. same judgment metadata

    SHA-256 is strongest and is what identifies
    ECOURTS_000016 as the same physical judgment
    as ECOURTS_000014.
    """

    seen_hashes = {}
    seen_urls = {}
    seen_judgments = {}

    duplicate_rows = []
    keep_indices = []

    for index, row in corpus.iterrows():

        document_id = clean(
            row.get(
                "document_id",
                "",
            )
        )

        file_name = clean(
            row.get(
                "file_name",
                "",
            )
        )

        file_path_value = clean(
            row.get(
                "file_path",
                "",
            )
        )

        url = normalize_url(
            row.get(
                "url",
                "",
            )
        )

        metadata_key = (
            judgment_key(row)
        )

        file_hash = ""

        if file_path_value:

            file_path = Path(
                file_path_value
            )

            if file_path.exists():

                file_hash = sha256_file(
                    file_path
                )

        duplicate_of = ""
        duplicate_type = ""
        duplicate_value = ""

        # ----------------------------------------------------
        # 1. Exact binary duplicate
        # ----------------------------------------------------

        if (
            file_hash
            and file_hash
            in seen_hashes
        ):

            duplicate_of = (
                seen_hashes[
                    file_hash
                ]
            )

            duplicate_type = (
                "same_file_hash"
            )

            duplicate_value = (
                file_hash
            )

        # ----------------------------------------------------
        # 2. Same source URL
        # ----------------------------------------------------

        elif (
            url
            and url
            in seen_urls
        ):

            duplicate_of = (
                seen_urls[url]
            )

            duplicate_type = (
                "same_url"
            )

            duplicate_value = url

        # ----------------------------------------------------
        # 3. Same normalized metadata
        # ----------------------------------------------------

        elif (
            metadata_key
            and metadata_key
            in seen_judgments
        ):

            duplicate_of = (
                seen_judgments[
                    metadata_key
                ]
            )

            duplicate_type = (
                "same_judgment_metadata"
            )

            duplicate_value = (
                metadata_key
            )

        # ----------------------------------------------------
        # Duplicate
        # ----------------------------------------------------

        if duplicate_of:

            duplicate_rows.append(
                {
                    "duplicate_document_id":
                        document_id,

                    "original_document_id":
                        duplicate_of,

                    "duplicate_type":
                        duplicate_type,

                    "duplicate_value":
                        duplicate_value,

                    "file_name":
                        file_name,
                }
            )

            continue

        # ----------------------------------------------------
        # Canonical document
        # ----------------------------------------------------

        keep_indices.append(
            index
        )

        if file_hash:

            seen_hashes[
                file_hash
            ] = document_id

        if url:

            seen_urls[
                url
            ] = document_id

        if metadata_key:

            seen_judgments[
                metadata_key
            ] = document_id

    deduped = (
        corpus
        .loc[
            keep_indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return (
        duplicate_rows,
        deduped,
    )


# ============================================================
# Write duplicate report
# ============================================================

def write_duplicate_report(
    rows: list[dict],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "duplicate_document_id",
        "original_document_id",
        "duplicate_type",
        "duplicate_value",
        "file_name",
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
# Main
# ============================================================

def main() -> None:

    corpus = load_valid_corpus()

    (
        duplicates,
        deduped_corpus,
    ) = detect_duplicates(
        corpus
    )

    # --------------------------------------------------------
    # Duplicate relationship report
    # --------------------------------------------------------

    write_duplicate_report(
        duplicates,
        DUPLICATE_DOWNLOADS_PATH,
    )

    # --------------------------------------------------------
    # Deduped corpus
    # --------------------------------------------------------

    DEDUPED_MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    deduped_corpus.to_csv(
        DEDUPED_MANIFEST_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    total_valid = len(
        corpus
    )

    duplicate_count = len(
        duplicates
    )

    unique_count = len(
        deduped_corpus
    )

    print(
        "\nGap 36 Duplicate Detection Report"
    )

    print("=" * 70)

    print(
        f"Valid PDFs checked: "
        f"{total_valid}"
    )

    print(
        f"Unique documents: "
        f"{unique_count}"
    )

    print(
        f"Duplicates found: "
        f"{duplicate_count}"
    )

    print(
        f"Deduped corpus rows: "
        f"{unique_count}"
    )

    print(
        f"Duplicate report saved to: "
        f"{DUPLICATE_DOWNLOADS_PATH}"
    )

    print(
        f"Deduped corpus saved to: "
        f"{DEDUPED_MANIFEST_PATH}"
    )

    print("=" * 70)

    for row in duplicates:

        print(
            f"{row['duplicate_document_id']} "
            f"duplicates "
            f"{row['original_document_id']} | "
            f"{row['duplicate_type']}"
        )

    # --------------------------------------------------------
    # Continue existing pipeline
    # --------------------------------------------------------

    print(
        "\nStarting Gap 39 Hash Index "
        "after Gap 36..."
    )

    print("=" * 70)

    run_hash_index()

    print(
        "\nStarting Gap 38 Error "
        "Classification after Gap 39..."
    )

    print("=" * 70)

    run_error_classifier()


if __name__ == "__main__":
    main()