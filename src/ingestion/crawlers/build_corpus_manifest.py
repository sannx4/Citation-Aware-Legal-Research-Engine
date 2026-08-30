import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from src.config import MANIFEST_PATH
from src.ingestion.crawlers.crawl_seed_urls import crawl_seed_urls
from src.ingestion.crawlers.manifest_builder import build_manifest, write_manifest


def main() -> None:
    seed_rows = crawl_seed_urls()

    manifest = build_manifest(seed_rows)

    write_manifest(manifest, MANIFEST_PATH)

    print("\nProduction Corpus Manifest Report")
    print("=" * 70)
    print(f"Total documents listed: {len(manifest)}")
    print(f"Manifest saved to: {MANIFEST_PATH}")
    print("=" * 70)

    for row in manifest[:10]:
        print(
            f"{row['document_id']} | "
            f"{row['source']} | "
            f"{row['document_type']} | "
            f"{row['file_name']}"
        )


if __name__ == "__main__":
    main()