import csv
import sys
from datetime import datetime
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    PARTIAL_DOWNLOADS_DIR,
    DOWNLOAD_CHECKPOINTS_PATH,
)


def append_checkpoint(row: dict) -> None:
    DOWNLOAD_CHECKPOINTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "document_id",
        "url",
        "file_name",
        "final_path",
        "part_path",
        "status",
        "bytes_downloaded",
        "timestamp",
        "error",
    ]

    file_exists = DOWNLOAD_CHECKPOINTS_PATH.exists()

    with DOWNLOAD_CHECKPOINTS_PATH.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def is_completed_file(output_path: Path) -> bool:
    return output_path.exists() and output_path.stat().st_size > 0


def checkpoint_download_file(
    document_id: str,
    url: str,
    output_path: Path,
    timeout: int = 60,
    chunk_size: int = 1024 * 256,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PARTIAL_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    part_path = PARTIAL_DOWNLOADS_DIR / f"{output_path.name}.part"

    if is_completed_file(output_path):
        append_checkpoint(
            {
                "document_id": document_id,
                "url": url,
                "file_name": output_path.name,
                "final_path": str(output_path),
                "part_path": str(part_path),
                "status": "skipped_completed",
                "bytes_downloaded": output_path.stat().st_size,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "error": "",
            }
        )
        return

    bytes_downloaded = 0

    try:
        if part_path.exists():
            part_path.unlink()

        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()

            with part_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
                        bytes_downloaded += len(chunk)

        if bytes_downloaded == 0:
            raise ValueError("empty_response")

        part_path.replace(output_path)

        append_checkpoint(
            {
                "document_id": document_id,
                "url": url,
                "file_name": output_path.name,
                "final_path": str(output_path),
                "part_path": str(part_path),
                "status": "success",
                "bytes_downloaded": bytes_downloaded,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "error": "",
            }
        )

    except Exception as error:
        append_checkpoint(
            {
                "document_id": document_id,
                "url": url,
                "file_name": output_path.name,
                "final_path": str(output_path),
                "part_path": str(part_path),
                "status": "failed_partial",
                "bytes_downloaded": bytes_downloaded,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "error": str(error),
            }
        )

        raise
