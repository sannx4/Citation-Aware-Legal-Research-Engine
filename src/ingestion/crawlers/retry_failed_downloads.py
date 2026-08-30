import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from src.config import FAILED_DOWNLOADS_PATH
from src.ingestion.crawlers.download_manager import download_file, get_output_dir


def retry_failed_downloads() -> None:
    if not FAILED_DOWNLOADS_PATH.exists():
        print("No failed downloads found.")
        return

    failed = pd.read_csv(FAILED_DOWNLOADS_PATH)

    for _, row in failed.iterrows():
        try:
            document_type = row.get("document_type", "judgment")
            output_dir = get_output_dir(document_type)
            output_path = output_dir / row["file_name"]

            download_file(row["url"], output_path)

            print(f"RETRY SUCCESS | {row['document_id']}")

        except Exception as error:
            print(f"RETRY FAILED | {row['document_id']} | {error}")


if __name__ == "__main__":
    retry_failed_downloads()