import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    STATUTE_MENTIONS_PATH,
    AUTHORITY_SCORES_PATH,
    KG_REASONING_RESULT_PATH,
)


LEGAL_SIGNAL_MAP = {
    "phone": ["privacy", "digital surveillance", "personal data"],
    "mobile": ["privacy", "digital surveillance", "personal data"],
    "data": ["privacy", "personal data"],
    "permission": ["consent", "due process"],
    "police": ["state action", "search and seizure"],
    "search": ["search and seizure", "privacy"],
    "surveillance": ["digital surveillance", "privacy"],
    "passport": ["personal liberty", "fair procedure"],
    "detention": ["personal liberty", "preventive detention"],
}


CONCEPT_TO_STATUTE = {
    "privacy": ["Article 21"],
    "digital surveillance": ["Article 21"],
    "personal data": ["Article 21"],
    "consent": ["Article 21"],
    "due process": ["Article 21"],
    "search and seizure": ["Article 21"],
    "personal liberty": ["Article 21"],
    "fair procedure": ["Article 21"],
}


CONCEPT_TO_CASE_HINTS = {
    "privacy": ["K.S. Puttaswamy v. Union of India"],
    "digital surveillance": ["K.S. Puttaswamy v. Union of India"],
    "personal data": ["K.S. Puttaswamy v. Union of India"],
    "personal liberty": ["Maneka Gandhi v. Union of India"],
    "fair procedure": ["Maneka Gandhi v. Union of India"],
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def detect_intent(query: str) -> str:
    q = query.lower()

    if any(x in q for x in ["can", "allowed", "legal", "right", "permission"]):
        return "legal_research"

    return "legal_search"


def extract_concepts(query: str) -> list[str]:
    concepts = []

    for token in tokenize(query):
        concepts.extend(LEGAL_SIGNAL_MAP.get(token, []))

    return sorted(set(concepts))


def infer_statutes(concepts: list[str]) -> list[str]:
    statutes = []

    for concept in concepts:
        statutes.extend(CONCEPT_TO_STATUTE.get(concept, []))

    return sorted(set(statutes))


def infer_cases(concepts: list[str]) -> list[str]:
    cases = []

    for concept in concepts:
        cases.extend(CONCEPT_TO_CASE_HINTS.get(concept, []))

    if AUTHORITY_SCORES_PATH.exists():
        authority = pd.read_csv(AUTHORITY_SCORES_PATH)

        for _, row in authority.iterrows():
            title = str(row.get("title", ""))

            if "Puttaswamy" in title and "privacy" in concepts:
                cases.append(title)

            if "Maneka" in title and "personal liberty" in concepts:
                cases.append(title)

    return sorted(set(cases))


def reason_over_query(query: str) -> dict:
    concepts = extract_concepts(query)
    statutes = infer_statutes(concepts)
    likely_cases = infer_cases(concepts)

    rewritten_query = " ".join(
        [query] + concepts + statutes + likely_cases
    )

    return {
        "intent": detect_intent(query),
        "concepts": concepts,
        "statutes": statutes,
        "likely_cases": likely_cases,
        "rewritten_query": rewritten_query,
        "reasoning_trace": [
            "Detected legally relevant surface terms from user query.",
            "Mapped surface terms to legal concepts.",
            "Linked legal concepts to constitutional/statutory provisions.",
            "Linked concepts to likely precedent cases using authority/case metadata.",
        ],
    }


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    args = parser.parse_args()

    result = reason_over_query(args.query)
    write_json(result, KG_REASONING_RESULT_PATH)

    print("\nPhase 4 KG + LLM Reasoning Prototype")
    print("=" * 70)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 70)
    print(f"Saved to: {KG_REASONING_RESULT_PATH}")


if __name__ == "__main__":
    main()