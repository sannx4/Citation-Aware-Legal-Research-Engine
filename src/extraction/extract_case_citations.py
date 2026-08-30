import csv
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import CASE_CHUNKS_PATH, CASE_CITATIONS_PATH


CASE_CITATION_PATTERN = re.compile(
    r"\b("
    r"[A-Z][A-Za-z.\s&'-]{1,80}?"
    r")\s+"
    r"(v\.|vs\.?|versus)\s+"
    r"("
    r"[A-Z][A-Za-z.\s&'-]{1,100}?"
    r")"
    r"(?=[,.;:\n)]|\s+Court|\s+held|\s+case|\s+under|\s+and|$)",
    flags=re.IGNORECASE,
)


FALSE_POSITIVE_TERMS = {
    "article",
    "section",
    "court",
    "petitioner",
    "respondent",
    "appellant",
    "issue",
    "facts",
    "reasoning",
    "decision",
}


def read_jsonl(input_path: Path) -> list[dict]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = []

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def clean_party_name(name: str) -> str:
    return " ".join(name.strip(" ,.;:\n\t").split())


def looks_like_false_positive(party_1: str, party_2: str) -> bool:
    p1 = party_1.lower().strip()
    p2 = party_2.lower().strip()

    if len(p1) < 2 or len(p2) < 2:
        return True

    if p1 in FALSE_POSITIVE_TERMS or p2 in FALSE_POSITIVE_TERMS:
        return True

    if p1.isdigit() or p2.isdigit():
        return True

    return False


def calculate_confidence(citation_text: str, party_1: str, party_2: str) -> float:
    confidence = 0.70

    if "." in party_1 or "." in party_2:
        confidence += 0.10

    if "Union of India" in citation_text:
        confidence += 0.10

    if "State of" in citation_text:
        confidence += 0.10

    if len(party_1.split()) >= 2 and len(party_2.split()) >= 2:
        confidence += 0.05

    return min(confidence, 0.95)


def extract_case_citations_from_text(text: str) -> list[dict]:
    citations = []

    for match in CASE_CITATION_PATTERN.finditer(text):
        party_1 = clean_party_name(match.group(1))
        connector = match.group(2)
        party_2 = clean_party_name(match.group(3))

        if looks_like_false_positive(party_1, party_2):
            continue

        citation_text = f"{party_1} {connector} {party_2}"
        confidence = calculate_confidence(citation_text, party_1, party_2)

        citations.append(
            {
                "citation_text": citation_text,
                "party_1": party_1,
                "party_2": party_2,
                "connector": connector,
                "confidence": confidence,
                "start_char": match.start(),
                "end_char": match.end(),
            }
        )

    return citations


def build_case_citation_rows(chunks: list[dict]) -> list[dict]:
    rows = []

    for chunk in chunks:
        case_id = chunk["case_id"]
        chunk_id = chunk["chunk_id"]
        chunk_text = chunk["chunk_text"]

        citations = extract_case_citations_from_text(chunk_text)

        for citation in citations:
            rows.append(
                {
                    "case_id": case_id,
                    "chunk_id": chunk_id,
                    "citation_text": citation["citation_text"],
                    "party_1": citation["party_1"],
                    "party_2": citation["party_2"],
                    "connector": citation["connector"],
                    "confidence": citation["confidence"],
                    "start_char": citation["start_char"],
                    "end_char": citation["end_char"],
                }
            )

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "chunk_id",
        "citation_text",
        "party_1",
        "party_2",
        "connector",
        "confidence",
        "start_char",
        "end_char",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main() -> None:
    chunks = read_jsonl(CASE_CHUNKS_PATH)
    rows = build_case_citation_rows(chunks)
    write_csv(rows, CASE_CITATIONS_PATH)

    print("\nDay 9 Case Citation Extraction Report")
    print("=" * 70)
    print(f"Chunks processed: {len(chunks)}")
    print(f"Case citations found: {len(rows)}")
    print(f"Output saved to: {CASE_CITATIONS_PATH}")
    print("=" * 70)

    for row in rows[:10]:
        print(
            f"{row['chunk_id']} | "
            f"{row['citation_text']} | "
            f"confidence={row['confidence']}"
        )


if __name__ == "__main__":
    main()