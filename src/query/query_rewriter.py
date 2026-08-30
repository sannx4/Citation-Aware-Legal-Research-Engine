EXPANSION_TERMS = {
    "privacy": [
        "privacy",
        "right to privacy",
        "personal data",
        "informational privacy",
    ],
    "surveillance": [
        "digital surveillance",
        "phone data",
        "monitoring",
    ],
    "personal liberty": [
        "personal liberty",
        "consent",
        "permission",
    ],
    "search and seizure": [
        "search",
        "seizure",
        "warrant",
        "reasonable restriction",
    ],
    "due process": [
        "due process",
        "fair procedure",
    ],
    "natural justice": [
        "natural justice",
        "fair hearing",
    ],
}


def rewrite_query(original_query: str, concepts: list[str], statutes: list[str]) -> str:
    terms = []

    terms.extend(statutes)

    for concept in concepts:
        terms.extend(EXPANSION_TERMS.get(concept, [concept]))

    terms.append(original_query)

    seen = set()
    final_terms = []

    for term in terms:
        key = term.lower().strip()

        if key and key not in seen:
            seen.add(key)
            final_terms.append(term.strip())

    return " ".join(final_terms)
