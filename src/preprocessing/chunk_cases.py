import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import CLEAN_CASES_PATH, CASE_CHUNKS_PATH


def read_jsonl(input_path: Path) -> list[dict]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = []

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_chunks(text: str, chunk_size: int, overlap: int) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "start_char": start,
                    "end_char": end,
                }
            )

        if end == text_length:
            break

        start = end - overlap

    return chunks


def build_case_chunks(records: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    all_chunks = []

    for record in records:
        case_id = record.get("case_id")
        title = record.get("title")
        court = record.get("court")
        year = record.get("year")
        cleaned_text = record.get("cleaned_text", "")

        chunks = create_chunks(cleaned_text, chunk_size, overlap)

        for index, chunk in enumerate(chunks, start=1):
            chunk_id = f"{case_id}_CHUNK_{index:04d}"

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "case_id": case_id,
                    "title": title,
                    "court": court,
                    "year": year,
                    "chunk_index": index,
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                    "chunk_text": chunk["text"],
                    "chunk_size": len(chunk["text"]),
                }
            )

    return all_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk cleaned legal judgments.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=900,
        help="Maximum number of characters per chunk.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=120,
        help="Number of overlapping characters between chunks.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = read_jsonl(CLEAN_CASES_PATH)
    chunks = build_case_chunks(records, args.chunk_size, args.overlap)

    write_jsonl(chunks, CASE_CHUNKS_PATH)

    print("\nDay 5 Chunking Report")
    print("=" * 70)
    print(f"Input cases: {len(records)}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Overlap: {args.overlap}")
    print(f"Output saved to: {CASE_CHUNKS_PATH}")
    print("=" * 70)

    for chunk in chunks[:5]:
        print(
            f"{chunk['chunk_id']} | "
            f"case={chunk['case_id']} | "
            f"chars={chunk['start_char']}-{chunk['end_char']} | "
            f"size={chunk['chunk_size']}"
        )


if __name__ == "__main__":
    main()