import re

from src.extraction.concept_dictionary import LEGAL_CONCEPT_DICTIONARY


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def find_keyword_positions(text: str, keyword: str) -> list[dict]:
    matches = []
    pattern = re.compile(r"\b" + re.escape(keyword.lower()) + r"\b")

    for match in pattern.finditer(text.lower()):
        matches.append(
            {
                "keyword": keyword,
                "start_char": match.start(),
                "end_char": match.end(),
            }
        )

    return matches


def extract_concepts_from_text(text: str) -> list[dict]:
    concepts = []
    normalized = normalize_text(text)

    for concept, keywords in LEGAL_CONCEPT_DICTIONARY.items():
        matched_keywords = []

        for keyword in keywords:
            if keyword.lower() in normalized:
                matched_keywords.append(keyword)

        if not matched_keywords:
            continue

        confidence = min(0.60 + (0.10 * len(matched_keywords)), 0.95)

        concepts.append(
            {
                "concept": concept,
                "matched_keywords": matched_keywords,
                "confidence": round(confidence, 2),
            }
        )

    return concepts
