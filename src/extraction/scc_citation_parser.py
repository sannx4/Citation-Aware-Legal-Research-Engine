from src.extraction.citation_patterns import SCC_PATTERN


def parse_scc_citations(text: str) -> list[dict]:
    citations = []

    for match in SCC_PATTERN.finditer(text):
        year = match.group(1)
        volume = match.group(2)
        page = match.group(3)

        citations.append(
            {
                "citation_type": "SCC",
                "citation_text": match.group(0),
                "year": year,
                "reporter": "SCC",
                "volume": volume,
                "court": "",
                "page_or_number": page,
                "start_char": match.start(),
                "end_char": match.end(),
                "confidence": 0.95,
            }
        )

    return citations