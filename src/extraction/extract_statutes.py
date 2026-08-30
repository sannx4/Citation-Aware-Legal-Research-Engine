import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    CASE_CHUNKS_PATH,
    STATUTE_MENTIONS_PATH,
)

from src.extraction.patterns import extract_statute_mentions


def read_jsonl(input_path: Path) -> list[dict]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    records = []

    with input_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def build_statute_mentions(
    chunks: list[dict]
) -> list[dict]:

    rows = []

    for chunk in chunks:

        chunk_id = chunk["chunk_id"]
        case_id = chunk["case_id"]
        chunk_text = chunk["chunk_text"]

        mentions = extract_statute_mentions(
            chunk_text
        )

        for mention in mentions:

            rows.append(
                {
                    "case_id": case_id,
                    "chunk_id": chunk_id,
                    "mention_type": mention["mention_type"],
                    "mention_text": mention["mention_text"],
                    "reference_number": mention["reference_number"],
                    "start_char": mention["start_char"],
                    "end_char": mention["end_char"],
                }
            )

    return rows


def write_csv(
    rows: list[dict],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "case_id",
        "chunk_id",
        "mention_type",
        "mention_text",
        "reference_number",
        "start_char",
        "end_char",
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

        for row in rows:
            writer.writerow(row)


def main() -> None:

    chunks = read_jsonl(
        CASE_CHUNKS_PATH
    )

    rows = build_statute_mentions(
        chunks
    )

    write_csv(
        rows,
        STATUTE_MENTIONS_PATH,
    )

    print("\nDay 8 Statute Extraction Report")
    print("=" * 70)
    print(f"Chunks processed: {len(chunks)}")
    print(f"Statute mentions found: {len(rows)}")
    print(
        f"Output saved to: "
        f"{STATUTE_MENTIONS_PATH}"
    )
    print("=" * 70)

    for row in rows[:10]:
        print(
            f"{row['chunk_id']} | "
            f"{row['mention_type']} | "
            f"{row['mention_text']}"
        )


if __name__ == "__main__":
    main()