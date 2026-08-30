import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    ROLE_BASED_CHUNKS_PATH,
    CASE_CHUNKS_PATH,
    STATUTE_MENTIONS_PATH,
    LEGAL_CONCEPTS_PATH,
)

from src.extraction.legal_concepts import extract_concepts_from_text


ARTICLE_CONCEPT_LINKS = {
    "21": ["PERSONAL_LIBERTY", "PRIVACY", "DUE_PROCESS"],
    "14": ["ARBITRARINESS", "NATURAL_JUSTICE"],
    "19": ["REASONABLE_RESTRICTION"],
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def get_input_chunks() -> list[dict]:
    if ROLE_BASED_CHUNKS_PATH.exists():
        return read_jsonl(ROLE_BASED_CHUNKS_PATH)

    return read_jsonl(CASE_CHUNKS_PATH)


def load_statute_mentions() -> list[dict]:
    if not STATUTE_MENTIONS_PATH.exists():
        return []

    rows = []

    with STATUTE_MENTIONS_PATH.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    return rows


def build_chunk_concept_rows(chunks: list[dict]) -> list[dict]:
    rows = []

    for chunk in chunks:
        concepts = extract_concepts_from_text(chunk.get("chunk_text", ""))

        for concept in concepts:
            rows.append(
                {
                    "source_type": "CASE",
                    "source_id": chunk.get("case_id", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "title": chunk.get("title", ""),
                    "court": chunk.get("court", ""),
                    "year": chunk.get("year", ""),
                    "role": chunk.get("role", ""),
                    "concept": concept["concept"],
                    "matched_keywords": "; ".join(concept["matched_keywords"]),
                    "confidence": concept["confidence"],
                    "link_reason": "keyword_match",
                }
            )

    return rows


def build_article_concept_rows(statute_mentions: list[dict]) -> list[dict]:
    rows = []

    for mention in statute_mentions:
        if mention.get("mention_type") != "ARTICLE":
            continue

        article_number = str(mention.get("reference_number", "")).strip()
        concepts = ARTICLE_CONCEPT_LINKS.get(article_number, [])

        for concept in concepts:
            rows.append(
                {
                    "source_type": "ARTICLE",
                    "source_id": f"ARTICLE_{article_number}",
                    "chunk_id": mention.get("chunk_id", ""),
                    "title": "",
                    "court": "",
                    "year": "",
                    "role": "",
                    "concept": concept,
                    "matched_keywords": mention.get("mention_text", ""),
                    "confidence": 0.85,
                    "link_reason": "article_concept_mapping",
                }
            )

    return rows


def remove_duplicate_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    unique_rows = []

    for row in rows:
        key = (
            row["source_type"],
            row["source_id"],
            row["chunk_id"],
            row["concept"],
            row["link_reason"],
        )

        if key in seen:
            continue

        seen.add(key)
        unique_rows.append(row)

    return unique_rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_type",
        "source_id",
        "chunk_id",
        "title",
        "court",
        "year",
        "role",
        "concept",
        "matched_keywords",
        "confidence",
        "link_reason",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    chunks = get_input_chunks()
    statute_mentions = load_statute_mentions()

    rows = []
    rows.extend(build_chunk_concept_rows(chunks))
    rows.extend(build_article_concept_rows(statute_mentions))

    rows = remove_duplicate_rows(rows)

    write_csv(rows, LEGAL_CONCEPTS_PATH)

    print("\nLegal Concept Extraction Report")
    print("=" * 70)
    print(f"Chunks processed: {len(chunks)}")
    print(f"Statute mentions checked: {len(statute_mentions)}")
    print(f"Concept links created: {len(rows)}")
    print(f"Output saved to: {LEGAL_CONCEPTS_PATH}")
    print("=" * 70)

    for row in rows[:10]:
        print(
            f"{row['source_id']} -> {row['concept']} | "
            f"reason={row['link_reason']} | "
            f"confidence={row['confidence']}"
        )


if __name__ == "__main__":
    main()
