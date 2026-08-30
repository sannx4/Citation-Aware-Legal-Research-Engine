import csv
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    MANIFEST_PATH,
    RAW_CASES_DIR,
    RAW_STATUTES_DIR,
    PDF_INTEGRITY_REPORT_PATH,
    DUPLICATE_DOWNLOADS_PATH,
    DEDUPED_MANIFEST_PATH,
)

from src.ingestion.hash_index import main as run_hash_index
from src.ingestion.error_classifier import main as run_error_classifier


def get_file_path(document_type: str, file_name: str) -> Path:
    if str(document_type).lower() == "statute":
        return RAW_STATUTES_DIR / file_name
    return RAW_CASES_DIR / file_name


def normalize_text(value: str) -> str:
    return " ".join(str(value).lower().strip().split())


def normalize_url(url: str) -> str:
    return str(url).strip().split("#")[0].rstrip("/")


def judgment_key(row: pd.Series) -> str:
    title = normalize_text(row.get("title", ""))
    court = normalize_text(row.get("court", ""))
    year = normalize_text(row.get("year", ""))
    return f"{title}|{court}|{year}"


def sha256_file(file_path: Path) -> str:
    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)

    return hasher.hexdigest()


def load_valid_pdf_ids() -> set[str]:
    if not PDF_INTEGRITY_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Run Gap 37 first. Missing: {PDF_INTEGRITY_REPORT_PATH}"
        )

    report = pd.read_csv(PDF_INTEGRITY_REPORT_PATH)

    valid = report[
        report["is_valid_pdf"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    ]

    return set(valid["document_id"].astype(str))


def detect_duplicates(
    manifest: pd.DataFrame,
    valid_ids: set[str],
) -> tuple[list[dict], pd.DataFrame]:

    seen_urls = {}
    seen_hashes = {}
    seen_judgments = {}

    duplicate_rows = []
    keep_ids = []

    for _, row in manifest.iterrows():
        document_id = str(row["document_id"])
        document_type = str(row["document_type"])
        file_name = str(row["file_name"])

        if document_id not in valid_ids:
            continue

        duplicate_found = False

        url = normalize_url(row.get("url", ""))

        if url and url in seen_urls:
            duplicate_rows.append(
                {
                    "duplicate_document_id": document_id,
                    "original_document_id": seen_urls[url],
                    "duplicate_type": "same_url",
                    "duplicate_value": url,
                    "file_name": file_name,
                }
            )
            duplicate_found = True
        elif url:
            seen_urls[url] = document_id

        file_path = get_file_path(document_type, file_name)

        if file_path.exists():
            file_hash = sha256_file(file_path)

            if file_hash in seen_hashes:
                duplicate_rows.append(
                    {
                        "duplicate_document_id": document_id,
                        "original_document_id": seen_hashes[file_hash],
                        "duplicate_type": "same_file_hash",
                        "duplicate_value": file_hash,
                        "file_name": file_name,
                    }
                )
                duplicate_found = True
            else:
                seen_hashes[file_hash] = document_id

        key = judgment_key(row)

        if key.strip("|") and key in seen_judgments:
            duplicate_rows.append(
                {
                    "duplicate_document_id": document_id,
                    "original_document_id": seen_judgments[key],
                    "duplicate_type": "same_judgment_metadata",
                    "duplicate_value": key,
                    "file_name": file_name,
                }
            )
            duplicate_found = True
        elif key.strip("|"):
            seen_judgments[key] = document_id

        if not duplicate_found:
            keep_ids.append(document_id)

    deduped_manifest = manifest[
        manifest["document_id"].astype(str).isin(keep_ids)
    ]

    return duplicate_rows, deduped_manifest


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "duplicate_document_id",
        "original_document_id",
        "duplicate_type",
        "duplicate_value",
        "file_name",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)
    valid_ids = load_valid_pdf_ids()

    duplicates, deduped_manifest = detect_duplicates(manifest, valid_ids)

    write_csv(duplicates, DUPLICATE_DOWNLOADS_PATH)

    DEDUPED_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    deduped_manifest.to_csv(DEDUPED_MANIFEST_PATH, index=False)

    print("\nGap 36 Duplicate Detection Report")
    print("=" * 70)
    print(f"Manifest rows checked: {len(manifest)}")
    print(f"Valid PDFs checked: {len(valid_ids)}")
    print(f"Duplicates found: {len(duplicates)}")
    print(f"Deduped manifest rows: {len(deduped_manifest)}")
    print(f"Duplicate report saved to: {DUPLICATE_DOWNLOADS_PATH}")
    print(f"Deduped manifest saved to: {DEDUPED_MANIFEST_PATH}")
    print("=" * 70)

    for row in duplicates[:10]:
        print(
            f"{row['duplicate_document_id']} duplicates "
            f"{row['original_document_id']} | "
            f"{row['duplicate_type']}"
        )

    print("\nStarting Gap 39 Hash Index after Gap 36...")
    print("=" * 70)
    run_hash_index()

    print("\nStarting Gap 38 Error Classification after Gap 39...")
    print("=" * 70)
    run_error_classifier()


if __name__ == "__main__":
    main()
