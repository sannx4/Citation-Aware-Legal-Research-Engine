import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    MANIFEST_PATH,
    RAW_CASES_DIR,
    RAW_STATUTES_DIR,
    PDF_INTEGRITY_REPORT_PATH,
    FILE_HASH_INDEX_PATH,
)


def get_file_path(document_type: str, file_name: str) -> Path:
    if str(document_type).lower() == "statute":
        return RAW_STATUTES_DIR / file_name

    return RAW_CASES_DIR / file_name


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


def load_old_hash_index() -> dict:
    if not FILE_HASH_INDEX_PATH.exists():
        return {}

    old_index = pd.read_csv(FILE_HASH_INDEX_PATH)

    result = {}

    for _, row in old_index.iterrows():
        result[str(row["document_id"])] = {
            "sha256_hash": str(row.get("sha256_hash", "")),
            "processed_at": str(row.get("processed_at", "")),
            "status": str(row.get("status", "")),
        }

    return result


def build_hash_index(manifest: pd.DataFrame, valid_ids: set[str]) -> list[dict]:
    old_index = load_old_hash_index()
    rows = []

    now = datetime.now().isoformat(timespec="seconds")

    for _, row in manifest.iterrows():
        document_id = str(row["document_id"])
        document_type = str(row["document_type"])
        file_name = str(row["file_name"])

        if document_id not in valid_ids:
            continue

        file_path = get_file_path(document_type, file_name)

        if not file_path.exists():
            rows.append(
                {
                    "document_id": document_id,
                    "file_name": file_name,
                    "file_path": str(file_path),
                    "sha256_hash": "",
                    "file_size": 0,
                    "processed_at": "",
                    "status": "missing_file",
                    "reprocess_needed": "yes",
                }
            )
            continue

        current_hash = sha256_file(file_path)
        file_size = file_path.stat().st_size

        old = old_index.get(document_id)

        if old and old["sha256_hash"] == current_hash:
            status = "unchanged"
            processed_at = old["processed_at"]
            reprocess_needed = "no"
        else:
            status = "new_or_changed"
            processed_at = now
            reprocess_needed = "yes"

        rows.append(
            {
                "document_id": document_id,
                "file_name": file_name,
                "file_path": str(file_path),
                "sha256_hash": current_hash,
                "file_size": file_size,
                "processed_at": processed_at,
                "status": status,
                "reprocess_needed": reprocess_needed,
            }
        )

    return rows


def write_hash_index(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "document_id",
        "file_name",
        "file_path",
        "sha256_hash",
        "file_size",
        "processed_at",
        "status",
        "reprocess_needed",
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

    rows = build_hash_index(manifest, valid_ids)

    write_hash_index(rows, FILE_HASH_INDEX_PATH)

    total = len(rows)
    unchanged = sum(1 for row in rows if row["status"] == "unchanged")
    changed = sum(1 for row in rows if row["status"] == "new_or_changed")
    missing = sum(1 for row in rows if row["status"] == "missing_file")

    print("\nGap 39 Hash Index Report")
    print("=" * 70)
    print(f"Valid documents indexed: {total}")
    print(f"Unchanged files: {unchanged}")
    print(f"New or changed files: {changed}")
    print(f"Missing files: {missing}")
    print(f"Hash index saved to: {FILE_HASH_INDEX_PATH}")
    print("=" * 70)

    for row in rows[:10]:
        print(
            f"{row['document_id']} | "
            f"{row['status']} | "
            f"reprocess={row['reprocess_needed']} | "
            f"{row['sha256_hash'][:12]}"
        )


if __name__ == "__main__":
    main()