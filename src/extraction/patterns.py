import re

ARTICLE_PATTERN = re.compile(
    r"\bArticle\s+(\d+[A-Z]?)\b",
    flags=re.IGNORECASE,
)

SECTION_PATTERN = re.compile(
    r"\bSection\s+(\d+[A-Z]?)\b",
    flags=re.IGNORECASE,
)


def extract_articles(text: str) -> list[dict]:
    mentions = []

    for match in ARTICLE_PATTERN.finditer(text):
        mentions.append(
            {
                "mention_type": "ARTICLE",
                "mention_text": match.group(0),
                "reference_number": match.group(1),
                "start_char": match.start(),
                "end_char": match.end(),
            }
        )

    return mentions


def extract_sections(text: str) -> list[dict]:
    mentions = []

    for match in SECTION_PATTERN.finditer(text):
        mentions.append(
            {
                "mention_type": "SECTION",
                "mention_text": match.group(0),
                "reference_number": match.group(1),
                "start_char": match.start(),
                "end_char": match.end(),
            }
        )

    return mentions


def extract_statute_mentions(text: str) -> list[dict]:
    mentions = []

    mentions.extend(extract_articles(text))
    mentions.extend(extract_sections(text))

    mentions.sort(key=lambda x: x["start_char"])

    return mentions