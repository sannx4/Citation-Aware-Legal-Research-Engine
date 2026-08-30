import csv
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from src.config import (
    MANIFEST_PATH,
    DOWNLOAD_LOGS_PATH,
    FAILED_DOWNLOADS_PATH,
    RAW_CASES_DIR,
    RAW_STATUTES_DIR,
    CRAWL_RATE_REPORT_PATH,
)

from src.ingestion.source_monitor import run_source_schema_monitor
from src.ingestion.robots_checker import run_compliance_check
from src.ingestion.rate_limiter import RateLimiter
from src.ingestion.checkpoint_downloader import checkpoint_download_file

from src.ingestion.resume_downloads import (
    initialize_pending_documents,
    should_skip_document,
    mark_download_success,
    mark_download_failed,
    print_resume_report,
)


def get_output_dir(document_type: str) -> Path:
    if str(document_type).lower() == "statute":
        return RAW_STATUTES_DIR

    return RAW_CASES_DIR


def append_csv(row: dict, path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def run_downloads(limit: int | None = None) -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)

    if limit:
        manifest = manifest.head(limit)

    schema_ok = run_source_schema_monitor()

    if not schema_ok:
        print("\nSCHEMA WARNING")
        print("=" * 70)
        print("Possible source schema change detected.")
        print("Check reports/source_schema_changes.md before large-scale crawling.")
        print("Continuing because this is your local controlled pipeline.")
        print("=" * 70)

    compliance_ok = run_compliance_check()

    if not compliance_ok:
        print("\nCOMPLIANCE WARNING")
        print("=" * 70)
        print("Some URLs are disallowed or need manual review.")
        print("Check reports/crawl_compliance_report.md before continuing.")
        print("Continuing because this is your local controlled pipeline.")
        print("=" * 70)

    document_ids = [str(row["document_id"]) for _, row in manifest.iterrows()]
    initialize_pending_documents(document_ids)

    rate_limiter = RateLimiter()

    log_fields = [
        "document_id",
        "url",
        "file_name",
        "status",
        "timestamp",
        "error",
    ]

    print("\nGap 44 + Gap 42 + Gap 40 + Gap 41 + Gap 43 Production Download Manager")
    print("=" * 70)

    for _, row in manifest.iterrows():
        document_id = str(row["document_id"])
        url = str(row["url"])
        file_name = str(row["file_name"])
        document_type = str(row["document_type"])

        if should_skip_document(document_id):
            rate_limiter.log_event(
                document_id=document_id,
                url=url,
                action="skipped_completed",
            )

            print(f"SKIPPED | {document_id} | already completed")
            continue

        output_dir = get_output_dir(document_type)
        output_path = output_dir / file_name

        try:
            rate_limiter.wait(document_id=document_id, url=url)

            checkpoint_download_file(
                document_id=document_id,
                url=url,
                output_path=output_path,
            )

            append_csv(
                {
                    "document_id": document_id,
                    "url": url,
                    "file_name": file_name,
                    "status": "success",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "error": "",
                },
                DOWNLOAD_LOGS_PATH,
                log_fields,
            )

            mark_download_success(document_id)

            print(f"SUCCESS | {document_id} | {file_name}")

        except Exception as error:
            append_csv(
                {
                    "document_id": document_id,
                    "url": url,
                    "file_name": file_name,
                    "status": "failed",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "error": str(error),
                },
                FAILED_DOWNLOADS_PATH,
                log_fields,
            )

            mark_download_failed(document_id)

            print(f"FAILED | {document_id} | {error}")

    print_resume_report()
    print(f"\nRate report saved to: {CRAWL_RATE_REPORT_PATH}")


if __name__ == "__main__":
    run_downloads(limit=10)
