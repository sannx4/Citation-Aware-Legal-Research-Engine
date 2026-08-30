import argparse
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import QUERY_UNDERSTANDING_PATH


LEGAL_CONCEPT_MAP = {
    "phone": ["privacy", "digital surveillance", "personal data"],
    "mobile": ["privacy", "digital surveillance", "personal data"],
    "whatsapp": ["privacy", "digital surveillance", "personal data"],
    "surveillance": ["privacy", "digital surveillance"],
    "data": ["privacy", "personal data"],
    "passport": ["personal liberty", "fair procedure"],
    "detention": ["personal liberty", "preventive detention"],
    "arrest": ["personal liberty", "due process"],
    "speech": ["free speech", "reasonable restriction"],
}


CONCEPT_STATUTE_MAP = {
    "privacy": ["Article 21"],
    "digital surveillance": ["Article 21"],
    "personal data": ["Article 21"],
    "personal liberty": ["Article 21"],
    "fair procedure": ["Article 21"],
    "free speech": ["Article 19"],
    "reasonable restriction": ["Article 19"],
}


CONCEPT_CASE_MAP = {
    "privacy": ["K.S. Puttaswamy v. Union of India"],
    "digital surveillance": ["K.S. Puttaswamy v. Union of India"],
    "personal data": ["K.S. Puttaswamy v. Union of India"],
    "personal liberty": [
        "Maneka Gandhi v. Union of India",
        "A.K. Gopalan v. State of Madras",
    ],
    "fair procedure": ["Maneka Gandhi v. Union of India"],
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def detect_intent(query: str) -> str:
    q = query.lower()

    if any(word in q for word in ["can", "is it legal", "allowed", "right"]):
        return "legal_research"

    return "general_legal_search"


def extract_concepts(query: str) -> list[str]:
    tokens = tokenize(query)
    concepts = []

    for token in tokens:
        concepts.extend(LEGAL_CONCEPT_MAP.get(token, []))

    return sorted(set(concepts))


def map_statutes(concepts: list[str]) -> list[str]:
    statutes = []

    for concept in concepts:
        statutes.extend(CONCEPT_STATUTE_MAP.get(concept, []))

    return sorted(set(statutes))


def map_likely_cases(concepts: list[str]) -> list[str]:
    cases = []

    for concept in concepts:
        cases.extend(CONCEPT_CASE_MAP.get(concept, []))

    return sorted(set(cases))


def build_rewritten_query(
    original_query: str,
    concepts: list[str],
    statutes: list[str],
    likely_cases: list[str],
) -> str:
    parts = [original_query]
    parts.extend(concepts)
    parts.extend(statutes)
    parts.extend(likely_cases)

    return " ".join(parts)


def analyze_query(query: str) -> dict:
    concepts = extract_concepts(query)
    statutes = map_statutes(concepts)
    likely_cases = map_likely_cases(concepts)

    return {
        "original_query": query,
        "intent": detect_intent(query),
        "concepts": concepts,
        "statutes": statutes,
        "likely_cases": likely_cases,
        "rewritten_query": build_rewritten_query(
            query,
            concepts,
            statutes,
            likely_cases,
        ),
    }


def write_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Legal query understanding module."
    )

    parser.add_argument("query", type=str)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = analyze_query(args.query)

    write_json(result, QUERY_UNDERSTANDING_PATH)

    print("\nLegal Query Understanding Report")
    print("=" * 70)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 70)
    print(f"Output saved to: {QUERY_UNDERSTANDING_PATH}")


if __name__ == "__main__":
    main()
