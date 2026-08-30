from src.extraction.citation_patterns import AIR_PATTERN, NEUTRAL_CITATION_PATTERN


def parse_air_citations(text: str) -> list[dict]:
    citations = []

    for match in AIR_PATTERN.finditer(text):
        citations.append(
            {
                "citation_type": "AIR",
                "citation_text": match.group(0),
                "year": match.group(1),
                "reporter": "AIR",
                "volume": "",
                "court": match.group(2),
                "page_or_number": match.group(3),
                "start_char": match.start(),
                "end_char": match.end(),
                "confidence": 0.95,
            }
        )

    return citations


def parse_neutral_citations(text: str) -> list[dict]:
    citations = []

    for match in NEUTRAL_CITATION_PATTERN.finditer(text):
        citations.append(
            {
                "citation_type": "NEUTRAL",
                "citation_text": match.group(0),
                "year": match.group(1),
                "reporter": match.group(2).upper(),
                "volume": "",
                "court": "",
                "page_or_number": match.group(3),
                "start_char": match.start(),
                "end_char": match.end(),
                "confidence": 0.95,
            }
        )

    return citations
