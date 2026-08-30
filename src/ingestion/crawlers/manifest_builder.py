import csv
from pathlib import Path


FIELDNAMES = [
    "document_id",
    "source",
    "document_type",
    "title",
    "court",
    "year",
    "url",
    "file_name",
    "status",
]


def write_manifest(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def build_manifest(*row_groups: list[dict]) -> list[dict]:
    manifest = []
    seen_urls = set()

    for rows in row_groups:
        for row in rows:
            url = row["url"]

            if url in seen_urls:
                continue

            seen_urls.add(url)
            manifest.append(row)

    return manifest