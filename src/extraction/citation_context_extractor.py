import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    ROLE_BASED_CHUNKS_PATH,
    CASE_CHUNKS_PATH,
    ADVANCED_CASE_CITATIONS_PATH,
    CITATION_CONTEXTS_PATH,
)

from src.extraction.citation_patterns import (
    CASE_NAME_PATTERN,
    clean_text,
    is_false_case_name,
)
from src.extraction.scc_citation_parser import parse_scc_citations
from src.extraction.neutral_citation_parser import (
    parse_air_citations,
    parse_neutral_citations,
)


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


def extract_case_name_citations(text: str) -> list[dict]:
    citations = []

    for match in CASE_NAME_PATTERN.finditer(text):
        party_1 = clean_text(match.group(1))
        connector = clean_text(match.group(2))
        party_2 = clean_text(match.group(3))

        if is_false_case_name(party_1, party_2):
            continue

        citation_text = f"{party_1} {connector} {party_2}"

        citations.append(
            {
                "citation_type": "CASE_NAME",
                "citation_text": citation_text,
                "year": "",
                "reporter": "",
                "volume": "",
                "court": "",
                "page_or_number": "",
                "party_1": party_1,
                "party_2": party_2,
                "connector": connector,
                "start_char": match.start(),
                "end_char": match.end(),
                "confidence": 0.90,
            }
        )

    return citations


def extract_context(text: str, start: int, end: int, window: int = 180) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)

    return " ".join(text[left:right].split())


def extract_all_citations(text: str) -> list[dict]:
    citations = []

    citations.extend(extract_case_name_citations(text))
    citations.extend(parse_scc_citations(text))
    citations.extend(parse_air_citations(text))
    citations.extend(parse_neutral_citations(text))

    citations.sort(key=lambda row: row["start_char"])

    return citations


def build_rows(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    citation_rows = []
    context_rows = []

    for chunk in chunks:
        chunk_text = chunk.get("chunk_text", "")
        citations = extract_all_citations(chunk_text)

        for index, citation in enumerate(citations, start=1):
            citation_id = (
                f"{chunk.get('chunk_id', 'CHUNK')}_ADV_CITATION_{index:04d}"
            )

            base_row = {
                "citation_id": citation_id,
                "case_id": chunk.get("case_id", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "title": chunk.get("title", ""),
                "court": chunk.get("court", ""),
                "year": chunk.get("year", ""),
                "role": chunk.get("role", ""),
                "citation_type": citation.get("citation_type", ""),
                "citation_text": citation.get("citation_text", ""),
                "citation_year": citation.get("year", ""),
                "reporter": citation.get("reporter", ""),
                "volume": citation.get("volume", ""),
                "citation_court": citation.get("court", ""),
                "page_or_number": citation.get("page_or_number", ""),
                "party_1": citation.get("party_1", ""),
                "party_2": citation.get("party_2", ""),
                "connector": citation.get("connector", ""),
                "start_char": citation.get("start_char", 0),
                "end_char": citation.get("end_char", 0),
                "confidence": citation.get("confidence", 0.0),
            }

            citation_rows.append(base_row)

            context_rows.append(
                {
                    "citation_id": citation_id,
                    "case_id": chunk.get("case_id", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "citation_text": citation.get("citation_text", ""),
                    "context": extract_context(
                        chunk_text,
                        citation.get("start_char", 0),
                        citation.get("end_char", 0),
                    ),
                }
            )

    return citation_rows, context_rows


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    chunks = get_input_chunks()
    citation_rows, context_rows = build_rows(chunks)

    citation_fields = [
        "citation_id",
        "case_id",
        "chunk_id",
        "title",
        "court",
        "year",
        "role",
        "citation_type",
        "citation_text",
        "citation_year",
        "reporter",
        "volume",
        "citation_court",
        "page_or_number",
        "party_1",
        "party_2",
        "connector",
        "start_char",
        "end_char",
        "confidence",
    ]

    context_fields = [
        "citation_id",
        "case_id",
        "chunk_id",
        "citation_text",
        "context",
    ]

    write_csv(citation_rows, ADVANCED_CASE_CITATIONS_PATH, citation_fields)
    write_csv(context_rows, CITATION_CONTEXTS_PATH, context_fields)

    print("\nAdvanced Citation Extraction Report")
    print("=" * 70)
    print(f"Chunks processed: {len(chunks)}")
    print(f"Advanced citations found: {len(citation_rows)}")
    print(f"Citation report saved to: {ADVANCED_CASE_CITATIONS_PATH}")
    print(f"Citation contexts saved to: {CITATION_CONTEXTS_PATH}")
    print("=" * 70)

    for row in citation_rows[:10]:
        print(
            f"{row['chunk_id']} | "
            f"{row['citation_type']} | "
            f"{row['citation_text']} | "
            f"confidence={row['confidence']}"
        )


if __name__ == "__main__":
    main()
