import re

from src.query.embedding_concept_matcher import find_embedding_concepts


STATUTE_PATTERNS = {
    "Article 21": [
        "privacy",
        "digital surveillance",
        "personal liberty",
        "search and seizure",
        "phone",
        "data",
        "consent",
    ],
    "Article 14": [
        "equality",
        "arbitrariness",
        "arbitrary",
    ],
    "Article 19": [
        "speech",
        "expression",
        "movement",
        "restriction",
    ],
}


def extract_explicit_statutes(query: str) -> list[str]:
    statutes = []

    for match in re.finditer(r"\bArticle\s+\d+[A-Z]?\b", query, re.IGNORECASE):
        statutes.append(match.group(0).title())

    for match in re.finditer(r"\bSection\s+\d+[A-Z]?\b", query, re.IGNORECASE):
        statutes.append(match.group(0).title())

    return statutes


def extract_concepts(query: str) -> list[str]:
    embedding_matches = find_embedding_concepts(
        query=query,
        threshold=0.30,
        top_k=5,
    )

    concepts = [row["concept"] for row in embedding_matches]

    return sorted(set(concepts))


def infer_statutes(query: str, concepts: list[str]) -> list[str]:
    text = query.lower()
    statutes = extract_explicit_statutes(query)

    combined = text + " " + " ".join(concepts)

    for statute, keywords in STATUTE_PATTERNS.items():
        if any(keyword in combined for keyword in keywords):
            statutes.append(statute)

    return sorted(set(statutes))
