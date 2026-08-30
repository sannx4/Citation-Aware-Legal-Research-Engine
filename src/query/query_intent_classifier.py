def classify_intent(query: str) -> str:
    text = query.lower()

    legal_keywords = [
        "police", "court", "case", "law", "legal", "rights",
        "article", "section", "bail", "arrest", "privacy",
        "permission", "search", "seizure", "detention",
        "contract", "murder", "passport", "government",
        "phone", "data", "surveillance", "warrant",
    ]

    if any(keyword in text for keyword in legal_keywords):
        return "legal_research"

    return "general_query"